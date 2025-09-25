"""Test fixtures and utilities for morphogenesis simulation testing.

Provides reusable test fixtures, mock environments, and data factories
for consistent and reliable testing.
"""

import tempfile
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import asyncio

from core.data.types import (
    CellID, Position, CellType, CellState, CellParameters, WorldParameters,
    ExperimentMetadata, create_cell_id, create_simulation_time
)
from core.data.state import CellData, SimulationState
from core.data.actions import CellAction, create_wait_action, create_move_action, create_swap_action


@dataclass
class MockEnvironment:
    """Mock environment for testing cell interactions."""
    cells: Dict[CellID, CellData] = field(default_factory=dict)
    world_parameters: WorldParameters = field(default_factory=WorldParameters)
    current_timestep: int = 0
    global_metrics: Dict[str, Any] = field(default_factory=dict)

    def add_cell(self, cell_data: CellData) -> None:
        """Add a cell to the environment."""
        self.cells[cell_data.cell_id] = cell_data

    def remove_cell(self, cell_id: CellID) -> None:
        """Remove a cell from the environment."""
        if cell_id in self.cells:
            del self.cells[cell_id]

    def get_neighbors(self, cell_id: CellID, radius: float = 5.0) -> Dict[CellID, CellData]:
        """Get neighboring cells within radius."""
        if cell_id not in self.cells:
            return {}

        center_cell = self.cells[cell_id]
        neighbors = {}

        for other_id, other_cell in self.cells.items():
            if other_id != cell_id:
                distance = center_cell.distance_to(other_cell)
                if distance <= radius:
                    neighbors[other_id] = other_cell

        return neighbors

    def advance_timestep(self) -> None:
        """Advance the simulation timestep."""
        self.current_timestep += 1

    def to_simulation_state(self) -> SimulationState:
        """Convert to SimulationState."""
        return SimulationState(
            timestep=create_simulation_time(self.current_timestep),
            cells=self.cells.copy(),
            world_parameters=self.world_parameters,
            global_metrics=self.global_metrics,
            metadata=None
        )


@dataclass
class TestScenario:
    """Predefined test scenario configuration."""
    name: str
    description: str
    cell_count: int
    world_size: Tuple[int, int]
    cell_positions: List[Position] = field(default_factory=list)
    cell_types: List[CellType] = field(default_factory=list)
    cell_values: List[float] = field(default_factory=list)
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)


class TestDataFactory:
    """Factory for creating test data with various patterns."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_random_cells(
        self,
        count: int,
        world_width: int = 100,
        world_height: int = 100,
        value_range: Tuple[float, float] = (0.0, 100.0)
    ) -> List[CellData]:
        """Create randomly positioned cells."""
        cells = []

        for i in range(count):
            cell_id = create_cell_id(i)
            position = Position(
                self.rng.uniform(0, world_width),
                self.rng.uniform(0, world_height)
            )
            sort_value = self.rng.uniform(*value_range)

            cell_data = CellData(
                cell_id=cell_id,
                position=position,
                cell_type=CellType.STANDARD,
                cell_state=CellState.ACTIVE,
                sort_value=sort_value,
                age=0
            )
            cells.append(cell_data)

        return cells

    def create_grid_cells(
        self,
        rows: int,
        cols: int,
        spacing: float = 5.0,
        value_pattern: str = "sequential"
    ) -> List[CellData]:
        """Create cells arranged in a grid pattern."""
        cells = []
        cell_id_counter = 0

        for row in range(rows):
            for col in range(cols):
                cell_id = create_cell_id(cell_id_counter)
                position = Position(col * spacing, row * spacing)

                # Determine sort value based on pattern
                if value_pattern == "sequential":
                    sort_value = float(cell_id_counter)
                elif value_pattern == "reverse":
                    sort_value = float(rows * cols - cell_id_counter - 1)
                elif value_pattern == "random":
                    sort_value = self.rng.uniform(0, 100)
                elif value_pattern == "checkerboard":
                    sort_value = float((row + col) % 2)
                else:
                    sort_value = 0.0

                cell_data = CellData(
                    cell_id=cell_id,
                    position=position,
                    cell_type=CellType.STANDARD,
                    cell_state=CellState.ACTIVE,
                    sort_value=sort_value,
                    age=0
                )
                cells.append(cell_data)
                cell_id_counter += 1

        return cells

    def create_clustered_cells(
        self,
        cluster_count: int,
        cells_per_cluster: int,
        world_width: int = 100,
        world_height: int = 100,
        cluster_radius: float = 10.0
    ) -> List[CellData]:
        """Create cells arranged in clusters."""
        cells = []
        cell_id_counter = 0

        # Create cluster centers
        cluster_centers = []
        for _ in range(cluster_count):
            center = Position(
                self.rng.uniform(cluster_radius, world_width - cluster_radius),
                self.rng.uniform(cluster_radius, world_height - cluster_radius)
            )
            cluster_centers.append(center)

        # Create cells around each center
        for cluster_idx, center in enumerate(cluster_centers):
            for _ in range(cells_per_cluster):
                # Position cell randomly within cluster radius
                angle = self.rng.uniform(0, 2 * 3.14159)
                distance = self.rng.uniform(0, cluster_radius)

                position = Position(
                    center.x + distance * self.rng.uniform(-1, 1),
                    center.y + distance * self.rng.uniform(-1, 1)
                )

                # Ensure position is within world bounds
                position = Position(
                    max(0, min(world_width, position.x)),
                    max(0, min(world_height, position.y))
                )

                cell_id = create_cell_id(cell_id_counter)
                sort_value = float(cluster_idx * cells_per_cluster + (cell_id_counter % cells_per_cluster))

                cell_data = CellData(
                    cell_id=cell_id,
                    position=position,
                    cell_type=CellType.STANDARD,
                    cell_state=CellState.ACTIVE,
                    sort_value=sort_value,
                    age=0
                )
                cells.append(cell_data)
                cell_id_counter += 1

        return cells

    def create_sorted_scenario(
        self,
        count: int,
        disorder_level: float = 0.1
    ) -> List[CellData]:
        """Create a mostly sorted scenario with some disorder."""
        cells = self.create_grid_cells(
            rows=int(count**0.5) + 1,
            cols=int(count**0.5) + 1,
            value_pattern="sequential"
        )[:count]

        # Add some disorder
        disorder_count = int(count * disorder_level)
        for _ in range(disorder_count):
            # Swap two random cells' sort values
            idx1, idx2 = self.rng.sample(range(count), 2)
            cells[idx1].sort_value, cells[idx2].sort_value = (
                cells[idx2].sort_value, cells[idx1].sort_value
            )

        return cells

    def create_performance_test_scenario(
        self,
        scale: str = "small"
    ) -> TestScenario:
        """Create scenarios for performance testing."""
        scenarios = {
            "small": TestScenario(
                name="small_performance",
                description="Small scale performance test",
                cell_count=50,
                world_size=(50, 50)
            ),
            "medium": TestScenario(
                name="medium_performance",
                description="Medium scale performance test",
                cell_count=200,
                world_size=(100, 100)
            ),
            "large": TestScenario(
                name="large_performance",
                description="Large scale performance test",
                cell_count=500,
                world_size=(200, 200)
            )
        }

        scenario = scenarios.get(scale, scenarios["small"])

        # Generate cells for the scenario
        cells = self.create_random_cells(
            scenario.cell_count,
            *scenario.world_size
        )

        scenario.cell_positions = [cell.position for cell in cells]
        scenario.cell_types = [cell.cell_type for cell in cells]
        scenario.cell_values = [cell.sort_value for cell in cells]

        return scenario


class TestFixtures:
    """Central fixture manager for simulation tests."""

    def __init__(self):
        self.temp_dirs: List[Path] = []
        self.mock_environments: List[MockEnvironment] = []
        self.data_factory = TestDataFactory()
        self.logger = logging.getLogger(__name__)

    def create_temp_dir(self, prefix: str = "morphogenesis_test_") -> Path:
        """Create a temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_mock_environment(
        self,
        cell_count: int = 10,
        world_size: Tuple[int, int] = (50, 50)
    ) -> MockEnvironment:
        """Create a mock environment with test cells."""
        env = MockEnvironment(
            world_parameters=WorldParameters(
                width=world_size[0],
                height=world_size[1]
            )
        )

        # Add test cells
        cells = self.data_factory.create_random_cells(
            cell_count, *world_size
        )

        for cell in cells:
            env.add_cell(cell)

        self.mock_environments.append(env)
        return env

    def create_test_files(self, temp_dir: Path) -> Dict[str, Path]:
        """Create common test files in a directory."""
        files = {}

        # Create configuration file
        config_data = {
            "simulation": {
                "world_width": 100,
                "world_height": 100,
                "cell_count": 50,
                "max_timesteps": 100
            },
            "test_mode": True
        }

        config_file = temp_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        files["config"] = config_file

        # Create test data file
        test_data = {
            "cells": [
                {
                    "id": i,
                    "position": {"x": i * 2.0, "y": i * 2.0},
                    "sort_value": float(i)
                }
                for i in range(10)
            ]
        }

        data_file = temp_dir / "test_data.json"
        with open(data_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        files["data"] = data_file

        # Create empty results directory
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        files["results_dir"] = results_dir

        return files

    def load_predefined_scenarios(self) -> Dict[str, TestScenario]:
        """Load predefined test scenarios."""
        scenarios = {}

        # Simple line scenario
        scenarios["line"] = TestScenario(
            name="line",
            description="Cells arranged in a line",
            cell_count=10,
            world_size=(100, 20),
            cell_positions=[Position(float(i * 10), 10.0) for i in range(10)],
            cell_values=list(range(10)),
            expected_outcomes={"final_order": "ascending"}
        )

        # Reverse line scenario
        scenarios["reverse_line"] = TestScenario(
            name="reverse_line",
            description="Cells in reverse order",
            cell_count=10,
            world_size=(100, 20),
            cell_positions=[Position(float(i * 10), 10.0) for i in range(10)],
            cell_values=list(range(9, -1, -1)),
            expected_outcomes={"sorting_required": True}
        )

        # Circle scenario
        import math
        circle_positions = []
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = 50 + 30 * math.cos(angle)
            y = 50 + 30 * math.sin(angle)
            circle_positions.append(Position(x, y))

        scenarios["circle"] = TestScenario(
            name="circle",
            description="Cells arranged in a circle",
            cell_count=12,
            world_size=(100, 100),
            cell_positions=circle_positions,
            cell_values=list(range(12)),
            expected_outcomes={"spatial_pattern": "circle"}
        )

        # Random scenario
        scenarios["random"] = TestScenario(
            name="random",
            description="Randomly placed cells",
            cell_count=25,
            world_size=(100, 100),
            expected_outcomes={"requires_sorting": True}
        )

        return scenarios

    def create_environment_from_scenario(self, scenario: TestScenario) -> MockEnvironment:
        """Create a mock environment from a test scenario."""
        env = MockEnvironment(
            world_parameters=WorldParameters(
                width=scenario.world_size[0],
                height=scenario.world_size[1]
            )
        )

        # Create cells based on scenario
        if scenario.cell_positions:
            # Use predefined positions
            for i in range(scenario.cell_count):
                cell_id = create_cell_id(i)
                position = scenario.cell_positions[i]
                cell_type = scenario.cell_types[i] if scenario.cell_types else CellType.STANDARD
                sort_value = scenario.cell_values[i] if scenario.cell_values else float(i)

                cell_data = CellData(
                    cell_id=cell_id,
                    position=position,
                    cell_type=cell_type,
                    cell_state=CellState.ACTIVE,
                    sort_value=sort_value,
                    age=0
                )
                env.add_cell(cell_data)
        else:
            # Generate random cells
            cells = self.data_factory.create_random_cells(
                scenario.cell_count,
                *scenario.world_size
            )
            for cell in cells:
                env.add_cell(cell)

        self.mock_environments.append(env)
        return env

    def create_benchmark_data(
        self,
        sizes: List[int] = None
    ) -> Dict[int, List[CellData]]:
        """Create benchmark datasets of various sizes."""
        if sizes is None:
            sizes = [10, 50, 100, 200, 500]

        benchmark_data = {}
        for size in sizes:
            cells = self.data_factory.create_random_cells(size)
            benchmark_data[size] = cells

        return benchmark_data

    def get_sample_actions(self, cell_id: CellID) -> List[CellAction]:
        """Get sample actions for testing."""
        timestep = create_simulation_time(1)

        actions = [
            create_wait_action(cell_id, timestep),
            create_move_action(cell_id, timestep, Position(10.0, 20.0)),
            create_swap_action(cell_id, timestep, create_cell_id(999))
        ]

        return actions

    def cleanup(self):
        """Clean up all created fixtures."""
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Cleaned up temp directory: {temp_dir}")

        # Clear mock environments
        self.mock_environments.clear()

        # Clear lists
        self.temp_dirs.clear()

        self.logger.info("All test fixtures cleaned up")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


# Utility functions for common test data patterns
def create_test_experiment_metadata(
    name: str = "test_experiment",
    seed: int = 42
) -> ExperimentMetadata:
    """Create test experiment metadata."""
    return ExperimentMetadata.create_new(
        name=name,
        description="Test experiment",
        seed=seed,
        researcher="test_framework"
    )


def create_test_cell_parameters(**overrides) -> CellParameters:
    """Create test cell parameters with optional overrides."""
    defaults = {
        'max_speed': 1.0,
        'movement_noise': 0.1,
        'interaction_radius': 5.0,
        'swap_probability': 0.1,
        'sorting_strength': 1.0,
        'max_age': 1000
    }
    defaults.update(overrides)
    return CellParameters(**defaults)


def create_test_world_parameters(**overrides) -> WorldParameters:
    """Create test world parameters with optional overrides."""
    defaults = {
        'width': 100,
        'height': 100,
        'periodic_boundaries': False,
        'viscosity': 0.1,
        'temperature': 1.0
    }
    defaults.update(overrides)
    return WorldParameters(**defaults)


# Example usage
if __name__ == "__main__":
    # Demonstrate fixture usage
    fixtures = TestFixtures()

    # Create test environment
    env = fixtures.create_mock_environment(cell_count=20)
    print(f"Created environment with {len(env.cells)} cells")

    # Create test scenarios
    scenarios = fixtures.load_predefined_scenarios()
    print(f"Available scenarios: {list(scenarios.keys())}")

    # Create temp directory and files
    temp_dir = fixtures.create_temp_dir()
    test_files = fixtures.create_test_files(temp_dir)
    print(f"Created test files: {list(test_files.keys())}")

    # Cleanup
    fixtures.cleanup()
    print("Fixtures cleaned up")