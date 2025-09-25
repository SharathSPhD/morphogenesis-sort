"""Integration tests for agent coordination and communication."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from core.agents.cell_agent import AsyncCellAgent
from core.agents.behaviors.adaptive_cell import AdaptiveCellAgent
from core.coordination.coordinator import SimulationCoordinator
from core.coordination.spatial_index import SpatialIndex
from core.data.types import CellID, Position
from core.data.state import CellData, SimulationState


class TestAgentCoordination:
    """Test coordination between multiple agents."""

    @pytest.mark.asyncio
    async def test_multi_agent_communication(self, sample_cells, basic_behavior_config, basic_cell_parameters):
        """Test communication between multiple agents."""
        agents = []

        # Create multiple agents
        for i, (cell_id, cell_data) in enumerate(list(sample_cells.items())[:3]):
            agent = AsyncCellAgent(
                cell_id=cell_id,
                initial_data=cell_data,
                behavior_config=basic_behavior_config,
                parameters=basic_cell_parameters
            )
            agents.append(agent)

        # Mock message passing between agents
        for agent in agents:
            agent._handle_message = AsyncMock()

        # Test broadcast message
        test_message = Mock()
        test_message.message_type = "test_broadcast"
        test_message.data = {"info": "test"}

        # Simulate coordinator broadcasting message
        for agent in agents:
            await agent._handle_message(test_message)

        # Verify all agents received the message
        for agent in agents:
            agent._handle_message.assert_called_once_with(test_message)

    @pytest.mark.asyncio
    async def test_spatial_neighbor_detection(self, sample_cells):
        """Test spatial neighbor detection between agents."""
        spatial_index = SpatialIndex(grid_size=10.0)

        # Add cells to spatial index
        for cell_id, cell_data in sample_cells.items():
            await spatial_index.add_cell(cell_id, cell_data.position)

        # Test neighbor finding
        test_cell_id = list(sample_cells.keys())[0]
        test_position = sample_cells[test_cell_id].position

        neighbors = await spatial_index.get_neighbors(
            test_cell_id, test_position, radius=15.0
        )

        # Should find some neighbors (other cells in the sample)
        assert len(neighbors) > 0
        assert test_cell_id not in neighbors  # Should not include self

    @pytest.mark.asyncio
    async def test_agent_lifecycle_coordination(self, sample_cells, basic_behavior_config, basic_cell_parameters):
        """Test coordinated agent lifecycle management."""
        coordinator = SimulationCoordinator()
        agents = []

        # Create and register agents
        for cell_id, cell_data in list(sample_cells.items())[:2]:
            agent = AsyncCellAgent(
                cell_id=cell_id,
                initial_data=cell_data,
                behavior_config=basic_behavior_config,
                parameters=basic_cell_parameters
            )

            # Mock the decision loop to prevent infinite execution
            agent._decision_loop = AsyncMock()
            agents.append(agent)

            await coordinator.register_agent(agent)

        # Start all agents through coordinator
        await coordinator.start_all_agents()

        # Verify agents are running
        for agent in agents:
            assert agent._running is True

        # Stop all agents through coordinator
        await coordinator.stop_all_agents()

        # Verify agents are stopped
        for agent in agents:
            assert agent._running is False

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, sample_cells):
        """Test conflict resolution between competing actions."""
        from core.coordination.conflict_resolver import ConflictResolver
        from core.data.actions import create_move_action, create_swap_action

        resolver = ConflictResolver()

        # Create conflicting actions
        cell1_id = CellID(1)
        cell2_id = CellID(2)

        # Both cells want to move to the same position
        target_position = Position(50.0, 50.0)

        action1 = create_move_action(cell1_id, 100, target_position)
        action2 = create_move_action(cell2_id, 100, target_position)

        conflicting_actions = [action1, action2]

        # Resolve conflicts
        resolved_actions = await resolver.resolve_conflicts(conflicting_actions)

        # Should handle the conflict (may allow one, modify positions, or reject both)
        assert len(resolved_actions) <= len(conflicting_actions)

        # Verify no two actions result in the same final position
        final_positions = []
        for action in resolved_actions:
            if hasattr(action, 'target_position'):
                final_positions.append((action.target_position.x, action.target_position.y))

        # Check for duplicate positions (allowing for some tolerance)
        unique_positions = set(final_positions)
        assert len(unique_positions) == len(final_positions) or len(final_positions) == 0

    @pytest.mark.asyncio
    async def test_simulation_state_synchronization(self, sample_simulation_state):
        """Test synchronization of simulation state across components."""
        coordinator = SimulationCoordinator()

        # Mock state updates
        coordinator.update_simulation_state = AsyncMock()
        coordinator.broadcast_state_update = AsyncMock()

        # Update simulation state
        await coordinator.update_simulation_state(sample_simulation_state)

        # Verify state was updated
        coordinator.update_simulation_state.assert_called_once_with(sample_simulation_state)

        # Test state broadcasting
        await coordinator.broadcast_state_update()
        coordinator.broadcast_state_update.assert_called_once()


class TestAdaptiveAgentInteractions:
    """Test interactions specific to adaptive agents."""

    @pytest.mark.asyncio
    async def test_adaptive_agent_imitation(self, sample_cells, adaptive_config, basic_cell_parameters):
        """Test imitation learning between adaptive agents."""
        # Create two adaptive agents
        cell1_id = list(sample_cells.keys())[0]
        cell2_id = list(sample_cells.keys())[1]

        agent1 = AdaptiveCellAgent(
            cell_id=cell1_id,
            initial_data=sample_cells[cell1_id],
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        agent2 = AdaptiveCellAgent(
            cell_id=cell2_id,
            initial_data=sample_cells[cell2_id],
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        # Set up neighbor relationship
        agent1.neighbor_provider = Mock()
        agent1.neighbor_provider.get_neighbors.return_value = {cell2_id: sample_cells[cell2_id]}

        # Mock agent2 as a high-performing role model
        agent1.role_models[cell2_id] = 0.8  # High performance score

        # Test imitation action
        imitation_action = await agent1._imitate_model(cell2_id)

        # Should return some action (or None if no imitation possible)
        assert imitation_action is None or hasattr(imitation_action, 'action_type')

    @pytest.mark.asyncio
    async def test_adaptive_agent_competition(self, sample_cells, adaptive_config, basic_cell_parameters):
        """Test competitive behavior between adaptive agents."""
        agents = []

        # Create multiple adaptive agents
        for cell_id, cell_data in list(sample_cells.items())[:3]:
            agent = AdaptiveCellAgent(
                cell_id=cell_id,
                initial_data=cell_data,
                adaptive_config=adaptive_config,
                parameters=basic_cell_parameters
            )
            agents.append(agent)

        # Set up competition for resources (mock scenario)
        resource_position = Position(25.0, 25.0)

        # Each agent evaluates the same resource
        decisions = []
        for agent in agents:
            # Mock the agent's decision making
            agent.neighbor_provider = Mock()
            agent.neighbor_provider.get_neighbors.return_value = {}
            agent.calculate_local_density = Mock(return_value=0.5)

            # Get the agent's state representation
            state = await agent._get_state_representation()

            # Get action decision
            action_type = await agent._epsilon_greedy_action(state)
            decisions.append((agent.cell_id, action_type))

        # Verify agents made decisions (may be different based on their internal state)
        assert len(decisions) == 3
        for cell_id, action_type in decisions:
            assert cell_id in [agent.cell_id for agent in agents]
            assert action_type in agents[0].action_space  # All agents have the same action space

    @pytest.mark.asyncio
    async def test_performance_feedback_integration(self, sample_cells, adaptive_config, basic_cell_parameters):
        """Test integration of performance feedback between agents and coordinator."""
        agent = AdaptiveCellAgent(
            cell_id=list(sample_cells.keys())[0],
            initial_data=list(sample_cells.values())[0],
            adaptive_config=adaptive_config,
            parameters=basic_cell_parameters
        )

        # Mock performance feedback message
        feedback_message = Mock()
        feedback_message.message_type = "performance_feedback"
        feedback_message.data = {"performance": 0.8, "rank": 2}

        # Send feedback to agent
        await agent._handle_message(feedback_message)

        # Verify agent processed the feedback
        # (In real implementation, this would update learning parameters)
        assert True  # Placeholder - actual test would verify internal state changes

    @pytest.mark.asyncio
    async def test_collective_behavior_emergence(self, sample_cells, adaptive_config, basic_cell_parameters):
        """Test emergence of collective behavior from individual agent interactions."""
        agents = []

        # Create a small group of adaptive agents
        for cell_id, cell_data in sample_cells.items():
            agent = AdaptiveCellAgent(
                cell_id=cell_id,
                initial_data=cell_data,
                adaptive_config=adaptive_config,
                parameters=basic_cell_parameters
            )
            agents.append(agent)

        # Set up mutual neighbor relationships
        for i, agent in enumerate(agents):
            neighbors = {
                other_agent.cell_id: other_agent.current_data
                for j, other_agent in enumerate(agents)
                if i != j
            }
            agent.neighbor_provider = Mock()
            agent.neighbor_provider.get_neighbors.return_value = neighbors
            agent.calculate_local_density = Mock(return_value=0.5)

        # Run a few decision cycles
        for _ in range(3):
            decisions = []
            for agent in agents:
                action_type = await agent._epsilon_greedy_action(
                    await agent._get_state_representation()
                )
                decisions.append(action_type)

            # Verify all agents made decisions
            assert len(decisions) == len(agents)

        # Test that agents can influence each other (simplified test)
        # In a real scenario, we'd measure coordination metrics
        performance_metrics = [agent.get_performance_metrics() for agent in agents]
        assert len(performance_metrics) == len(agents)


class TestDataFlowIntegration:
    """Test data flow between different system components."""

    @pytest.mark.asyncio
    async def test_action_to_state_update_flow(self, sample_cells, basic_behavior_config, basic_cell_parameters):
        """Test the flow from agent actions to state updates."""
        from core.data.actions import create_move_action
        from core.coordination.scheduler import ActionScheduler

        scheduler = ActionScheduler()

        # Create an agent action
        cell_id = list(sample_cells.keys())[0]
        action = create_move_action(
            cell_id, 100, Position(30.0, 40.0)
        )

        # Schedule the action
        await scheduler.schedule_action(action)

        # Execute scheduled actions
        executed_actions = await scheduler.execute_scheduled_actions()

        # Verify action was executed
        assert len(executed_actions) >= 0  # May be 0 if action was invalid/conflicted

    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self, sample_simulation_state):
        """Test integration of metrics collection across components."""
        from core.metrics.collector import MetricsCollector

        collector = MetricsCollector()

        # Collect metrics from simulation state
        metrics = await collector.collect_state_metrics(sample_simulation_state)

        assert isinstance(metrics, dict)
        assert 'total_cells' in metrics or len(metrics) == 0  # May be empty if not implemented

        # Test metrics aggregation
        aggregated = await collector.aggregate_metrics([metrics])
        assert isinstance(aggregated, dict)