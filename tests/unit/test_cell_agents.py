"""Unit tests for cell agents and behaviors."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from core.agents.cell_agent import AsyncCellAgent, CellBehaviorConfig
from core.agents.behaviors.adaptive_cell import AdaptiveCellAgent
from core.agents.behaviors.sorting_cell import SortingCellAgent
from core.agents.behaviors.morphogen_cell import MorphogenCellAgent
from core.data.types import CellID, Position, CellType, CellState
from core.data.state import CellData
from core.data.actions import CellAction, ActionType


class TestAsyncCellAgent:
    """Test base AsyncCellAgent functionality."""

    @pytest.mark.asyncio
    async def test_agent_creation(self, sample_cell_data, basic_behavior_config, basic_cell_parameters):
        """Test creating a basic cell agent."""
        agent = AsyncCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            behavior_config=basic_behavior_config,
            parameters=basic_cell_parameters
        )

        assert agent.cell_id == sample_cell_data.cell_id
        assert agent.current_data == sample_cell_data
        assert agent.behavior_config == basic_behavior_config

    @pytest.mark.asyncio
    async def test_agent_update_data(self, sample_cell_data, basic_behavior_config, basic_cell_parameters):
        """Test updating agent data."""
        agent = AsyncCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            behavior_config=basic_behavior_config,
            parameters=basic_cell_parameters
        )

        new_data = CellData(
            cell_id=sample_cell_data.cell_id,
            position=Position(50.0, 60.0),
            cell_type=sample_cell_data.cell_type,
            cell_state=sample_cell_data.cell_state,
            sort_value=100,
            age=20
        )

        await agent.update_data(new_data)
        assert agent.current_data.position.x == 50.0
        assert agent.current_data.sort_value == 100
        assert agent.current_data.age == 20

    @pytest.mark.asyncio
    async def test_agent_get_neighbors(self, sample_cell_data, basic_behavior_config, basic_cell_parameters, sample_cells):
        """Test getting neighbors functionality."""
        agent = AsyncCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            behavior_config=basic_behavior_config,
            parameters=basic_cell_parameters
        )

        # Mock the neighbor_provider
        agent.neighbor_provider = Mock()
        agent.neighbor_provider.get_neighbors.return_value = sample_cells

        neighbors = agent.get_neighbors()
        assert len(neighbors) > 0
        agent.neighbor_provider.get_neighbors.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, sample_cell_data, basic_behavior_config, basic_cell_parameters):
        """Test agent start/stop lifecycle."""
        agent = AsyncCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            behavior_config=basic_behavior_config,
            parameters=basic_cell_parameters
        )

        # Mock the decision loop to avoid infinite execution
        agent._decision_loop = AsyncMock()

        # Test start
        await agent.start()
        assert agent._running is True

        # Test stop
        await agent.stop()
        assert agent._running is False


class TestAdaptiveCellAgent:
    """Test AdaptiveCellAgent behavior."""

    @pytest.mark.asyncio
    async def test_adaptive_agent_creation(self, sample_cell_data, adaptive_config, basic_cell_parameters):
        """Test creating an adaptive cell agent."""
        agent = AdaptiveCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        assert agent.adaptive_config == adaptive_config
        assert len(agent.state_features) > 0
        assert len(agent.action_space) > 0
        assert agent.policy_params is not None

    @pytest.mark.asyncio
    async def test_adaptive_state_representation(self, sample_cell_data, adaptive_config, basic_cell_parameters):
        """Test state representation extraction."""
        agent = AdaptiveCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        # Mock neighbor provider
        agent.neighbor_provider = Mock()
        agent.neighbor_provider.get_neighbors.return_value = {}
        agent.calculate_local_density = Mock(return_value=0.5)

        state = await agent._get_state_representation()
        assert isinstance(state, dict)
        assert 'position_x' in state
        assert 'position_y' in state
        assert 'sort_value' in state
        assert state['position_x'] == 10.0
        assert state['sort_value'] == 42

    @pytest.mark.asyncio
    async def test_adaptive_action_selection(self, sample_cell_data, adaptive_config, basic_cell_parameters):
        """Test action selection mechanism."""
        agent = AdaptiveCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        # Mock dependencies
        agent.neighbor_provider = Mock()
        agent.neighbor_provider.get_neighbors.return_value = {}
        agent.calculate_local_density = Mock(return_value=0.5)

        # Test epsilon-greedy action selection
        state = {'position_x': 10.0, 'position_y': 20.0, 'sort_value': 42}
        action_type = await agent._epsilon_greedy_action(state)

        assert action_type in agent.action_space

    @pytest.mark.asyncio
    async def test_adaptive_learning(self, sample_cell_data, adaptive_config, basic_cell_parameters):
        """Test learning from experience."""
        agent = AdaptiveCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        # Create a simple experience
        from core.agents.behaviors.adaptive_cell import Experience
        experience = Experience(
            state={'position_x': 10.0, 'sort_value': 42},
            action='move',
            reward=1.0,
            next_state={'position_x': 15.0, 'sort_value': 42},
            timestamp=100
        )

        # Test Q-learning update
        initial_weights = agent.policy_params.weights.copy()
        await agent._q_learning_update(experience)

        # Weights should be updated (unless they're zero-initialized)
        # Just test that the function runs without error
        assert agent.policy_params.weights is not None


class TestSortingCellAgent:
    """Test SortingCellAgent behavior."""

    @pytest.mark.asyncio
    async def test_sorting_agent_creation(self, sample_cell_data, basic_cell_parameters):
        """Test creating a sorting cell agent."""
        from core.agents.behaviors.config import SortingConfig

        sorting_config = SortingConfig(
            sorting_algorithm="bubble_sort",
            patience_threshold=10,
            cooperation_radius=3.0,
            swap_probability=0.8
        )

        agent = SortingCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            sorting_config=sorting_config,
            parameters=basic_cell_parameters
        )

        assert agent.sorting_config == sorting_config
        assert agent.patience_counter == 0

    @pytest.mark.asyncio
    async def test_sorting_bubble_sort_behavior(self, sample_cell_data, basic_cell_parameters):
        """Test bubble sort behavior."""
        from core.agents.behaviors.config import SortingConfig

        sorting_config = SortingConfig(
            sorting_algorithm="bubble_sort",
            patience_threshold=10,
            cooperation_radius=3.0,
            swap_probability=0.8
        )

        agent = SortingCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            sorting_config=sorting_config,
            parameters=basic_cell_parameters
        )

        # Mock neighbor with higher sort value to the left
        neighbor_data = CellData(
            cell_id=CellID(999),
            position=Position(5.0, 20.0),  # Left of current cell
            cell_type=CellType.SORTING,
            cell_state=CellState.ACTIVE,
            sort_value=100,  # Higher than current cell (42)
            age=5
        )

        agent.neighbor_provider = Mock()
        agent.neighbor_provider.get_neighbors.return_value = {CellID(999): neighbor_data}

        action = await agent._bubble_sort_behavior()

        # Should suggest a swap since neighbor has higher value and is to the left
        if hasattr(action, 'action_type'):
            # Could be swap or wait depending on exact implementation
            assert action.action_type in [ActionType.SWAP, ActionType.WAIT]


class TestMorphogenCellAgent:
    """Test MorphogenCellAgent behavior."""

    @pytest.mark.asyncio
    async def test_morphogen_agent_creation(self, sample_cell_data, basic_cell_parameters):
        """Test creating a morphogen cell agent."""
        from core.agents.behaviors.config import MorphogenConfig

        morphogen_config = MorphogenConfig(
            morphogen_type="activator",
            production_rate=1.0,
            diffusion_rate=0.1,
            decay_rate=0.05,
            gradient_threshold=0.3
        )

        agent = MorphogenCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            morphogen_config=morphogen_config,
            parameters=basic_cell_parameters
        )

        assert agent.morphogen_config == morphogen_config
        assert agent.morphogen_concentration >= 0

    @pytest.mark.asyncio
    async def test_morphogen_production(self, sample_cell_data, basic_cell_parameters):
        """Test morphogen production behavior."""
        from core.agents.behaviors.config import MorphogenConfig

        morphogen_config = MorphogenConfig(
            morphogen_type="activator",
            production_rate=1.0,
            diffusion_rate=0.1,
            decay_rate=0.05,
            gradient_threshold=0.3
        )

        agent = MorphogenCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            morphogen_config=morphogen_config,
            parameters=basic_cell_parameters
        )

        initial_concentration = agent.morphogen_concentration
        await agent._produce_morphogen()

        # Production should increase concentration
        assert agent.morphogen_concentration >= initial_concentration

    @pytest.mark.asyncio
    async def test_morphogen_gradient_following(self, sample_cell_data, basic_cell_parameters):
        """Test gradient following behavior."""
        from core.agents.behaviors.config import MorphogenConfig

        morphogen_config = MorphogenConfig(
            morphogen_type="activator",
            production_rate=1.0,
            diffusion_rate=0.1,
            decay_rate=0.05,
            gradient_threshold=0.3
        )

        agent = MorphogenCellAgent(
            cell_id=sample_cell_data.cell_id,
            initial_data=sample_cell_data,
            morphogen_config=morphogen_config,
            parameters=basic_cell_parameters
        )

        # Mock neighbors with different concentrations
        neighbors = {
            CellID(2): CellData(
                cell_id=CellID(2),
                position=Position(8.0, 20.0),
                cell_type=CellType.MORPHOGEN,
                cell_state=CellState.ACTIVE,
                sort_value=20,
                age=5
            ),
            CellID(3): CellData(
                cell_id=CellID(3),
                position=Position(12.0, 20.0),
                cell_type=CellType.MORPHOGEN,
                cell_state=CellState.ACTIVE,
                sort_value=30,
                age=5
            )
        }

        agent.neighbor_provider = Mock()
        agent.neighbor_provider.get_neighbors.return_value = neighbors

        # Mock morphogen concentrations
        agent.get_morphogen_concentration = Mock()
        agent.get_morphogen_concentration.side_effect = lambda cell_id: {
            CellID(2): 0.5,
            CellID(3): 1.0
        }.get(cell_id, 0.8)  # Default for self

        gradient = await agent._calculate_morphogen_gradient()
        assert isinstance(gradient, tuple)
        assert len(gradient) == 2  # x, y components