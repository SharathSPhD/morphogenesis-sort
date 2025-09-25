"""Unit tests for core data types and structures."""

import pytest
from dataclasses import asdict
from core.data.types import (
    CellID, Position, CellType, CellState, CellParameters,
    WorldParameters, ExperimentMetadata
)
from core.data.state import CellData, SimulationState


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating a Position."""
        pos = Position(10.5, 20.3)
        assert pos.x == 10.5
        assert pos.y == 20.3

    def test_position_distance(self):
        """Test distance calculation between positions."""
        pos1 = Position(0.0, 0.0)
        pos2 = Position(3.0, 4.0)
        assert pos1.distance_to(pos2) == 5.0

    def test_position_equality(self):
        """Test position equality comparison."""
        pos1 = Position(1.0, 2.0)
        pos2 = Position(1.0, 2.0)
        pos3 = Position(1.1, 2.0)

        assert pos1 == pos2
        assert pos1 != pos3


class TestCellData:
    """Test CellData dataclass."""

    def test_cell_data_creation(self, sample_cell_data):
        """Test creating CellData."""
        assert sample_cell_data.cell_id == CellID(1)
        assert sample_cell_data.position.x == 10.0
        assert sample_cell_data.position.y == 20.0
        assert sample_cell_data.sort_value == 42
        assert sample_cell_data.age == 10

    def test_cell_data_distance_to(self, sample_cell_data):
        """Test distance calculation between cells."""
        other_cell = CellData(
            cell_id=CellID(2),
            position=Position(13.0, 24.0),
            cell_type=CellType.MORPHOGEN,
            cell_state=CellState.ACTIVE,
            sort_value=50,
            age=5
        )
        distance = sample_cell_data.distance_to(other_cell)
        assert distance == 5.0

    def test_cell_data_serialization(self, sample_cell_data):
        """Test cell data can be converted to dict."""
        cell_dict = asdict(sample_cell_data)
        assert cell_dict['cell_id'] == 1
        assert cell_dict['position']['x'] == 10.0
        assert cell_dict['sort_value'] == 42


class TestCellParameters:
    """Test CellParameters dataclass."""

    def test_cell_parameters_defaults(self):
        """Test default cell parameters."""
        params = CellParameters()
        assert params.initial_energy > 0
        assert params.movement_cost >= 0
        assert params.interaction_radius > 0

    def test_cell_parameters_validation(self):
        """Test cell parameters validation."""
        # Valid parameters
        params = CellParameters(
            initial_energy=100.0,
            movement_cost=1.0,
            interaction_radius=3.0
        )
        assert params.initial_energy == 100.0

        # Negative values should work (they represent different behaviors)
        params_negative = CellParameters(movement_cost=-1.0)
        assert params_negative.movement_cost == -1.0


class TestWorldParameters:
    """Test WorldParameters dataclass."""

    def test_world_parameters_creation(self, basic_world_parameters):
        """Test creating world parameters."""
        assert basic_world_parameters.width == 100.0
        assert basic_world_parameters.height == 100.0
        assert basic_world_parameters.total_timesteps == 1000

    def test_world_parameters_boundary_validation(self):
        """Test world parameters with different boundary conditions."""
        params_periodic = WorldParameters(
            width=50.0,
            height=50.0,
            periodic_boundaries=True
        )
        assert params_periodic.periodic_boundaries is True

        params_non_periodic = WorldParameters(
            width=50.0,
            height=50.0,
            periodic_boundaries=False
        )
        assert params_non_periodic.periodic_boundaries is False


class TestSimulationState:
    """Test SimulationState dataclass."""

    def test_simulation_state_creation(self, sample_simulation_state):
        """Test creating a simulation state."""
        assert sample_simulation_state.timestep == 100
        assert len(sample_simulation_state.cells) == 5
        assert sample_simulation_state.global_metrics['total_energy'] == 500.0

    def test_simulation_state_add_cell(self, sample_simulation_state):
        """Test adding a cell to simulation state."""
        new_cell = CellData(
            cell_id=CellID(99),
            position=Position(100.0, 100.0),
            cell_type=CellType.ADAPTIVE,
            cell_state=CellState.ACTIVE,
            sort_value=999,
            age=1
        )

        # Add cell to state
        sample_simulation_state.cells[CellID(99)] = new_cell
        assert len(sample_simulation_state.cells) == 6
        assert CellID(99) in sample_simulation_state.cells

    def test_simulation_state_get_cells_by_type(self, sample_simulation_state):
        """Test filtering cells by type."""
        morphogen_cells = [
            cell for cell in sample_simulation_state.cells.values()
            if cell.cell_type == CellType.MORPHOGEN
        ]
        # All sample cells are MORPHOGEN type
        assert len(morphogen_cells) == 5

    def test_simulation_state_get_active_cells(self, sample_simulation_state):
        """Test filtering active cells."""
        active_cells = [
            cell for cell in sample_simulation_state.cells.values()
            if cell.cell_state == CellState.ACTIVE
        ]
        # All sample cells are ACTIVE state
        assert len(active_cells) == 5


class TestEnumTypes:
    """Test enum types."""

    def test_cell_type_enum(self):
        """Test CellType enum values."""
        assert CellType.MORPHOGEN.value == "morphogen"
        assert CellType.ADAPTIVE.value == "adaptive"
        assert CellType.SORTING.value == "sorting"

    def test_cell_state_enum(self):
        """Test CellState enum values."""
        assert CellState.ACTIVE.value == "active"
        assert CellState.FROZEN.value == "frozen"
        assert CellState.DEAD.value == "dead"

    def test_enum_comparison(self):
        """Test enum comparisons."""
        assert CellType.MORPHOGEN == CellType.MORPHOGEN
        assert CellType.MORPHOGEN != CellType.ADAPTIVE

        assert CellState.ACTIVE == CellState.ACTIVE
        assert CellState.ACTIVE != CellState.FROZEN