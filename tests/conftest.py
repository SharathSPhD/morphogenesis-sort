"""Pytest configuration and shared fixtures for morphogenesis tests."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List

from core.data.types import CellID, Position, CellType, CellState, CellParameters, WorldParameters
from core.data.state import CellData, SimulationState
from core.agents.cell_agent import CellBehaviorConfig
from core.agents.behaviors.config import AdaptiveConfig
from infrastructure.testing import TestFixtures, TestDataGenerator


@pytest.fixture
def event_loop():
    """Create an asyncio event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def basic_cell_parameters():
    """Create basic cell parameters for testing."""
    return CellParameters(
        initial_energy=100.0,
        movement_cost=1.0,
        swap_cost=2.0,
        interaction_radius=3.0,
        perception_radius=5.0,
        decision_threshold=0.5,
        cooperation_bonus=1.2,
        energy_decay_rate=0.01,
        age_decay_factor=0.001
    )


@pytest.fixture
def basic_world_parameters():
    """Create basic world parameters for testing."""
    return WorldParameters(
        width=100.0,
        height=100.0,
        periodic_boundaries=False,
        time_step_size=1.0,
        total_timesteps=1000,
        spatial_grid_resolution=1.0,
        interaction_radius=3.0,
        max_neighbors=10,
        environment_noise=0.0,
        external_forces=False
    )


@pytest.fixture
def basic_behavior_config():
    """Create basic behavior configuration for testing."""
    return CellBehaviorConfig(
        behavior_type="basic",
        decision_frequency=1,
        action_delay=0.0,
        movement_enabled=True,
        max_movement_distance=2.0,
        movement_randomness=0.1,
        interaction_enabled=True,
        cooperation_probability=0.5,
        learning_enabled=False,
        learning_rate=0.01,
        memory_capacity=100,
        error_recovery=True,
        max_retry_attempts=3,
        freeze_on_error=False
    )


@pytest.fixture
def adaptive_config():
    """Create adaptive configuration for testing."""
    return AdaptiveConfig(
        learning_strategy="reinforcement",
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        exploration_strategy="epsilon_greedy",
        reward_functions=["sorting_progress", "efficiency"],
        reward_weights=[1.0, 0.5],
        experience_replay=True,
        replay_batch_size=32,
        performance_window=100,
        imitation_enabled=True,
        imitation_threshold=0.7,
        intrinsic_motivation=True,
        curiosity_bonus=0.1,
        use_neural_network=False
    )


@pytest.fixture
def sample_cell_data():
    """Create sample cell data for testing."""
    return CellData(
        cell_id=CellID(1),
        position=Position(10.0, 20.0),
        cell_type=CellType.MORPHOGEN,
        cell_state=CellState.ACTIVE,
        sort_value=42,
        age=10
    )


@pytest.fixture
def sample_cells():
    """Create a collection of sample cells for testing."""
    cells = {}
    for i in range(5):
        cell_id = CellID(i)
        cells[cell_id] = CellData(
            cell_id=cell_id,
            position=Position(float(i * 10), float(i * 5)),
            cell_type=CellType.MORPHOGEN,
            cell_state=CellState.ACTIVE,
            sort_value=i * 10,
            age=i + 1
        )
    return cells


@pytest.fixture
def sample_simulation_state(sample_cells, basic_world_parameters):
    """Create a sample simulation state for testing."""
    return SimulationState(
        timestep=100,
        cells=sample_cells,
        world_parameters=basic_world_parameters,
        global_metrics={"total_energy": 500.0, "sorted_pairs": 80}
    )


@pytest.fixture
def test_fixtures():
    """Provide test fixtures utility."""
    return TestFixtures()


@pytest.fixture
def test_data_generator():
    """Provide test data generator utility."""
    return TestDataGenerator()


@pytest.fixture
def random_seed():
    """Provide consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def performance_benchmark_params():
    """Parameters for performance benchmarking tests."""
    return {
        'small_scale': {'cell_count': 10, 'timesteps': 100},
        'medium_scale': {'cell_count': 100, 'timesteps': 500},
        'large_scale': {'cell_count': 1000, 'timesteps': 1000}
    }