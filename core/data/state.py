"""Simulation state management for morphogenesis simulation.

This module manages the complete state of the simulation, including
cell data, world state, and immutable snapshots for analysis.
All state objects are designed to be immutable and thread-safe.
"""

import copy
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Dict, List, Optional, Tuple, Set, FrozenSet
from collections import defaultdict

from .types import (
    CellID, Position, CellState, CellType, SimulationTime,
    WorldDimensions, CellParameters, WorldParameters, ExperimentMetadata
)
from .actions import CellAction


@dataclass(frozen=True)
class CellData:
    """Immutable data structure representing a single cell.

    Contains all information about a cell including its position,
    state, properties, and history relevant for simulation.
    """
    cell_id: CellID
    position: Position
    cell_type: CellType
    state: CellState
    age: int = 0
    energy: float = 100.0

    # Sorting-specific properties
    sort_value: float = 0.0  # Value used for sorting algorithms
    target_position: Optional[Position] = None

    # Memory and learning
    memory: Dict[str, float] = field(default_factory=dict)
    experiences: List[str] = field(default_factory=list)

    # Social/interaction properties
    neighbors: FrozenSet[CellID] = frozenset()
    last_interaction_time: SimulationTime = SimulationTime(0)

    # Physical properties
    velocity: Position = Position(0.0, 0.0)
    acceleration: Position = Position(0.0, 0.0)

    # Behavioral parameters (can be individual)
    parameters: CellParameters = field(default_factory=CellParameters)

    def __post_init__(self):
        """Validate cell data after initialization."""
        if not self.is_valid():
            raise ValueError(f"Invalid cell data: {self}")

    def is_valid(self) -> bool:
        """Check if this cell data is valid."""
        checks = [
            isinstance(self.cell_id, int),
            isinstance(self.position, Position),
            isinstance(self.cell_type, CellType),
            isinstance(self.state, CellState),
            self.age >= 0,
            self.energy >= 0,
            isinstance(self.parameters, CellParameters),
            self.parameters.validate()
        ]
        return all(checks)

    def is_alive(self) -> bool:
        """Check if this cell is alive."""
        return self.state not in {CellState.DEAD, CellState.DYING}

    def is_active(self) -> bool:
        """Check if this cell can take actions."""
        return self.state in {CellState.ACTIVE, CellState.WAITING}

    def can_move(self) -> bool:
        """Check if this cell can move."""
        return self.is_active() and self.energy > 0

    def can_interact(self) -> bool:
        """Check if this cell can interact with others."""
        return self.is_active() and self.state != CellState.FROZEN

    def distance_to(self, other: 'CellData') -> float:
        """Calculate distance to another cell."""
        return self.position.distance_to(other.position)

    def with_position(self, new_position: Position) -> 'CellData':
        """Create a new CellData with updated position."""
        return dataclass_replace(self, position=new_position)

    def with_state(self, new_state: CellState) -> 'CellData':
        """Create a new CellData with updated state."""
        return dataclass_replace(self, state=new_state)

    def with_age(self, new_age: int) -> 'CellData':
        """Create a new CellData with updated age."""
        return dataclass_replace(self, age=new_age)

    def with_energy(self, new_energy: float) -> 'CellData':
        """Create a new CellData with updated energy."""
        return dataclass_replace(self, energy=max(0.0, new_energy))

    def with_neighbors(self, new_neighbors: Set[CellID]) -> 'CellData':
        """Create a new CellData with updated neighbors."""
        return dataclass_replace(self, neighbors=frozenset(new_neighbors))

    def add_experience(self, experience: str) -> 'CellData':
        """Create a new CellData with added experience."""
        new_experiences = list(self.experiences) + [experience]
        # Limit experience history to prevent unbounded growth
        if len(new_experiences) > 100:
            new_experiences = new_experiences[-100:]
        return dataclass_replace(self, experiences=new_experiences)

    def update_memory(self, key: str, value: float, decay: float = 0.99) -> 'CellData':
        """Create a new CellData with updated memory."""
        new_memory = dict(self.memory)

        # Apply decay to all existing memories
        for k in new_memory:
            new_memory[k] *= decay

        # Update specific memory
        new_memory[key] = value

        # Remove very weak memories to prevent unbounded growth
        new_memory = {k: v for k, v in new_memory.items() if abs(v) > 0.001}

        return dataclass_replace(self, memory=new_memory)


@dataclass(frozen=True)
class WorldState:
    """Immutable representation of the world environment state.

    Contains environmental factors that affect cell behavior,
    such as chemical gradients, physical constraints, and global parameters.
    """
    dimensions: WorldDimensions
    parameters: WorldParameters
    timestep: SimulationTime = SimulationTime(0)
    total_actions: int = 0

    # Environmental fields
    temperature_field: Optional[Dict[Position, float]] = None
    chemical_gradients: Dict[str, Dict[Position, float]] = field(default_factory=dict)
    obstacles: Set[Position] = field(default_factory=set)

    # Spatial indexing information
    occupied_positions: Set[Position] = field(default_factory=set)
    spatial_grid: Dict[Tuple[int, int], Set[CellID]] = field(default_factory=dict)

    def is_position_valid(self, position: Position) -> bool:
        """Check if a position is within world bounds."""
        width, height = self.dimensions
        return 0 <= position.x < width and 0 <= position.y < height

    def is_position_occupied(self, position: Position) -> bool:
        """Check if a position is occupied by a cell."""
        return position in self.occupied_positions

    def is_position_blocked(self, position: Position) -> bool:
        """Check if a position is blocked by obstacles."""
        return position in self.obstacles

    def get_grid_cell(self, position: Position) -> Tuple[int, int]:
        """Get the spatial grid cell for a position."""
        grid_size = self.parameters.grid_cell_size
        return (
            int(position.x // grid_size),
            int(position.y // grid_size)
        )

    def get_neighbors_in_grid(self, grid_cell: Tuple[int, int]) -> Set[CellID]:
        """Get all cells in a spatial grid cell."""
        return self.spatial_grid.get(grid_cell, set())


@dataclass(frozen=True)
class SimulationState:
    """Complete immutable state of the morphogenesis simulation.

    This is the authoritative state representation that includes
    all cells, world state, and simulation metadata.
    """
    timestep: SimulationTime
    cells: Dict[CellID, CellData]
    world_state: WorldState
    metadata: ExperimentMetadata

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Action history (for analysis)
    last_actions: Dict[CellID, CellAction] = field(default_factory=dict)

    def __post_init__(self):
        """Validate simulation state after initialization."""
        if not self.is_valid():
            raise ValueError(f"Invalid simulation state at timestep {self.timestep}")

    def is_valid(self) -> bool:
        """Check if this simulation state is valid."""
        try:
            # Check basic structure
            if not isinstance(self.timestep, int) or self.timestep < 0:
                return False

            if not isinstance(self.cells, dict):
                return False

            if not isinstance(self.world_state, WorldState):
                return False

            # Validate all cells
            for cell_id, cell_data in self.cells.items():
                if not isinstance(cell_data, CellData):
                    return False
                if cell_data.cell_id != cell_id:
                    return False
                if not cell_data.is_valid():
                    return False
                if not self.world_state.is_position_valid(cell_data.position):
                    return False

            # Check for position conflicts
            positions = [cell.position for cell in self.cells.values()]
            if len(positions) != len(set(positions)):
                # Multiple cells at same position (may be allowed in some cases)
                pass

            return True

        except Exception:
            return False

    @property
    def population_size(self) -> int:
        """Get current population size."""
        return len(self.cells)

    @property
    def living_cells(self) -> Dict[CellID, CellData]:
        """Get all living cells."""
        return {cid: cell for cid, cell in self.cells.items() if cell.is_alive()}

    @property
    def active_cells(self) -> Dict[CellID, CellData]:
        """Get all cells that can take actions."""
        return {cid: cell for cid, cell in self.cells.items() if cell.is_active()}

    def get_cell(self, cell_id: CellID) -> Optional[CellData]:
        """Get a specific cell by ID."""
        return self.cells.get(cell_id)

    def get_cells_by_type(self, cell_type: CellType) -> Dict[CellID, CellData]:
        """Get all cells of a specific type."""
        return {
            cid: cell for cid, cell in self.cells.items()
            if cell.cell_type == cell_type
        }

    def get_cells_by_state(self, state: CellState) -> Dict[CellID, CellData]:
        """Get all cells in a specific state."""
        return {
            cid: cell for cid, cell in self.cells.items()
            if cell.state == state
        }

    def get_cells_in_region(
        self,
        center: Position,
        radius: float
    ) -> Dict[CellID, CellData]:
        """Get all cells within a radius of a position."""
        return {
            cid: cell for cid, cell in self.cells.items()
            if cell.position.distance_to(center) <= radius
        }

    def get_neighbors(self, cell_id: CellID, radius: float) -> Dict[CellID, CellData]:
        """Get all neighbors of a cell within a radius."""
        cell = self.get_cell(cell_id)
        if cell is None:
            return {}

        return {
            cid: other_cell for cid, other_cell in self.cells.items()
            if cid != cell_id and cell.distance_to(other_cell) <= radius
        }

    def update_cell(self, cell_id: CellID, new_cell_data: CellData) -> 'SimulationState':
        """Create a new state with an updated cell."""
        new_cells = dict(self.cells)
        new_cells[cell_id] = new_cell_data
        return dataclass_replace(self, cells=new_cells)

    def add_cell(self, cell_data: CellData) -> 'SimulationState':
        """Create a new state with an additional cell."""
        new_cells = dict(self.cells)
        new_cells[cell_data.cell_id] = cell_data
        return dataclass_replace(self, cells=new_cells)

    def remove_cell(self, cell_id: CellID) -> 'SimulationState':
        """Create a new state with a cell removed."""
        new_cells = dict(self.cells)
        new_cells.pop(cell_id, None)
        return dataclass_replace(self, cells=new_cells)

    def with_timestep(self, new_timestep: SimulationTime) -> 'SimulationState':
        """Create a new state with updated timestep."""
        return dataclass_replace(self, timestep=new_timestep)

    def compute_sorting_metrics(self) -> Dict[str, float]:
        """Compute metrics related to sorting quality."""
        if not self.cells:
            return {'sorting_quality': 0.0, 'clustering_coefficient': 0.0}

        # Simple sorting quality based on sort_value ordering
        cells_by_position = sorted(self.cells.values(), key=lambda c: (c.position.x, c.position.y))
        sort_values = [cell.sort_value for cell in cells_by_position]

        # Calculate how well-sorted the values are
        inversions = 0
        n = len(sort_values)
        for i in range(n):
            for j in range(i + 1, n):
                if sort_values[i] > sort_values[j]:
                    inversions += 1

        max_inversions = n * (n - 1) // 2
        sorting_quality = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0

        # Simple clustering coefficient
        clustering_coeff = 0.0
        for cell in self.cells.values():
            neighbors = self.get_neighbors(cell.cell_id, cell.parameters.interaction_radius)
            if len(neighbors) > 1:
                same_type_neighbors = sum(
                    1 for neighbor in neighbors.values()
                    if neighbor.cell_type == cell.cell_type
                )
                clustering_coeff += same_type_neighbors / len(neighbors)

        if self.cells:
            clustering_coeff /= len(self.cells)

        return {
            'sorting_quality': sorting_quality,
            'clustering_coefficient': clustering_coeff,
            'population_size': len(self.cells),
            'inversions': inversions,
            'max_inversions': max_inversions
        }

    @staticmethod
    def empty(world_params: WorldParameters) -> 'SimulationState':
        """Create an empty simulation state."""
        from .types import ExperimentMetadata

        metadata = ExperimentMetadata.create_new(
            "empty_simulation",
            "Empty simulation state",
            42
        )

        world_state = WorldState(
            dimensions=world_params.dimensions,
            parameters=world_params
        )

        return SimulationState(
            timestep=SimulationTime(0),
            cells={},
            world_state=world_state,
            metadata=metadata
        )

    @staticmethod
    def create_initial_state(
        world_params: WorldParameters,
        cell_params: CellParameters,
        initial_cells: Optional[List[CellData]] = None
    ) -> 'SimulationState':
        """Create initial simulation state with optional cells."""
        from .types import ExperimentMetadata

        metadata = ExperimentMetadata.create_new(
            "initial_simulation",
            "Initial simulation state",
            42
        )

        cells_dict = {}
        occupied_positions = set()

        if initial_cells:
            for cell in initial_cells:
                cells_dict[cell.cell_id] = cell
                occupied_positions.add(cell.position)

        world_state = WorldState(
            dimensions=world_params.dimensions,
            parameters=world_params,
            occupied_positions=occupied_positions
        )

        return SimulationState(
            timestep=SimulationTime(0),
            cells=cells_dict,
            world_state=world_state,
            metadata=metadata
        )


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of simulation state for analysis.

    Lightweight representation that captures essential information
    for data analysis without storing the complete state.
    """
    timestep: SimulationTime
    timestamp: float  # Wall clock time when snapshot was taken

    # Basic population metrics
    population_size: int
    living_cells: int
    active_cells: int

    # Cell type distribution
    type_counts: Dict[CellType, int]
    state_counts: Dict[CellState, int]

    # Spatial metrics
    average_position: Position
    position_variance: float
    bounding_box: Tuple[Position, Position]  # (min, max)

    # Sorting metrics
    sorting_quality: float
    clustering_coefficient: float
    inversions: int

    # Performance metrics
    step_execution_time: float
    memory_usage: float

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_simulation_state(cls, state: SimulationState, execution_time: float = 0.0) -> 'StateSnapshot':
        """Create a snapshot from a complete simulation state."""
        cells = list(state.cells.values())

        # Calculate type and state distributions
        type_counts = defaultdict(int)
        state_counts = defaultdict(int)

        for cell in cells:
            type_counts[cell.cell_type] += 1
            state_counts[cell.state] += 1

        # Calculate spatial metrics
        positions = [cell.position for cell in cells]
        if positions:
            avg_x = sum(p.x for p in positions) / len(positions)
            avg_y = sum(p.y for p in positions) / len(positions)
            average_position = Position(avg_x, avg_y)

            # Position variance
            var_x = sum((p.x - avg_x) ** 2 for p in positions) / len(positions)
            var_y = sum((p.y - avg_y) ** 2 for p in positions) / len(positions)
            position_variance = (var_x + var_y) / 2

            # Bounding box
            min_pos = Position(
                min(p.x for p in positions),
                min(p.y for p in positions)
            )
            max_pos = Position(
                max(p.x for p in positions),
                max(p.y for p in positions)
            )
            bounding_box = (min_pos, max_pos)
        else:
            average_position = Position(0, 0)
            position_variance = 0.0
            bounding_box = (Position(0, 0), Position(0, 0))

        # Calculate sorting metrics
        sorting_metrics = state.compute_sorting_metrics()

        return cls(
            timestep=state.timestep,
            timestamp=time.time(),
            population_size=len(cells),
            living_cells=len([c for c in cells if c.is_alive()]),
            active_cells=len([c for c in cells if c.is_active()]),
            type_counts=dict(type_counts),
            state_counts=dict(state_counts),
            average_position=average_position,
            position_variance=position_variance,
            bounding_box=bounding_box,
            sorting_quality=sorting_metrics['sorting_quality'],
            clustering_coefficient=sorting_metrics['clustering_coefficient'],
            inversions=sorting_metrics['inversions'],
            step_execution_time=execution_time,
            memory_usage=0.0,  # Could be calculated if needed
        )


# Utility functions for state management
def create_initial_state(
    metadata: ExperimentMetadata,
    world_params: WorldParameters,
    initial_cells: List[CellData]
) -> SimulationState:
    """Create an initial simulation state."""
    world_state = WorldState(
        dimensions=world_params.dimensions,
        parameters=world_params,
        occupied_positions=set(cell.position for cell in initial_cells)
    )

    cells_dict = {cell.cell_id: cell for cell in initial_cells}

    return SimulationState(
        timestep=SimulationTime(0),
        cells=cells_dict,
        world_state=world_state,
        metadata=metadata
    )


def validate_state_transition(
    old_state: SimulationState,
    new_state: SimulationState
) -> List[str]:
    """Validate a state transition for consistency.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check timestep progression
    if new_state.timestep != old_state.timestep + 1:
        errors.append(f"Invalid timestep progression: {old_state.timestep} -> {new_state.timestep}")

    # Check that living cells don't disappear without dying
    old_living = set(old_state.living_cells.keys())
    new_living = set(new_state.living_cells.keys())
    disappeared = old_living - new_living

    for cell_id in disappeared:
        # Check if cell died in the last actions
        if cell_id in old_state.last_actions:
            action = old_state.last_actions[cell_id]
            if not hasattr(action, 'action_type') or action.action_type.value != 'die':
                errors.append(f"Cell {cell_id} disappeared without dying")

    # Check world consistency
    if new_state.world_state.dimensions != old_state.world_state.dimensions:
        errors.append("World dimensions changed during simulation")

    return errors