"""DeterministicCoordinator - Central coordination engine for cell simulation.

This module implements the core coordination engine that manages async cell agents
in a deterministic, time-stepped manner. It eliminates the threading artifacts
present in the original implementation while ensuring scientific reproducibility.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, AsyncGenerator,
    Callable, Awaitable, Tuple, Union
)
from enum import Enum
import random

from ..agents.cell_agent import AsyncCellAgent, AgentMessage, AgentState
from ..data.types import (
    CellID, SimulationTime, Position, CellParameters, WorldParameters,
    create_simulation_time
)
from ..data.state import SimulationState, CellData, WorldState
from ..data.actions import (
    CellAction, ActionType, WaitAction, MoveAction, SwapAction,
    ActionResult, ActionStatus
)
from .spatial_index import SpatialIndex
from .scheduler import TimeStepScheduler
from .conflict_resolver import ConflictResolver


class CoordinatorState(Enum):
    """States of the coordinator lifecycle."""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class CoordinationMetrics:
    """Metrics for coordination engine performance."""
    total_timesteps: int = 0
    total_actions_processed: int = 0
    total_conflicts_resolved: int = 0
    average_timestep_duration: float = 0.0
    active_agents_count: int = 0
    failed_actions_count: int = 0

    # Performance metrics
    timestep_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    actions_per_timestep: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_timestep_performance(self, duration: float, action_count: int) -> None:
        """Update performance metrics after timestep completion."""
        self.timestep_durations.append(duration)
        self.actions_per_timestep.append(action_count)

        if self.timestep_durations:
            self.average_timestep_duration = sum(self.timestep_durations) / len(self.timestep_durations)


@dataclass
class CoordinatorConfig:
    """Configuration for the deterministic coordinator."""
    # Core execution parameters
    max_timesteps: int = 10000
    timestep_timeout: float = 1.0  # Max time per timestep in seconds
    max_concurrent_actions: int = 1000

    # Deterministic execution
    global_seed: int = 42
    action_ordering_seed: int = 123
    conflict_resolution_seed: int = 456

    # Performance tuning
    batch_size: int = 100  # Actions to process in each batch
    spatial_index_updates_per_timestep: int = 10
    metrics_collection_frequency: int = 10

    # Error handling
    max_action_retries: int = 3
    freeze_on_error: bool = True
    continue_on_agent_failure: bool = True

    # Memory management
    max_action_history: int = 1000
    gc_frequency: int = 100


class DeterministicCoordinator:
    """Central coordination engine for deterministic cell simulation.

    This class replaces the problematic threading approach with async-based
    coordination that ensures deterministic, reproducible execution. It manages
    all cell agents, coordinates their actions, and maintains simulation state.

    Key Features:
    - Deterministic time-stepped execution
    - Async cell agent management
    - Conflict resolution for competing actions
    - Spatial indexing for efficient neighbor queries
    - Performance monitoring and optimization
    - Complete elimination of race conditions
    """

    def __init__(
        self,
        world_params: WorldParameters,
        cell_params: CellParameters,
        config: CoordinatorConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.world_params = world_params
        self.cell_params = cell_params
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Core state
        self.state = CoordinatorState.CREATED
        self.current_timestep = create_simulation_time(0)
        self.simulation_state = SimulationState.empty(world_params)

        # Agent management
        self.agents: Dict[CellID, AsyncCellAgent] = {}
        self.agent_tasks: Dict[CellID, asyncio.Task] = {}
        self.active_agents: Set[CellID] = set()

        # Coordination infrastructure
        self.rng = random.Random(config.global_seed)
        self.spatial_index = SpatialIndex(world_params)
        self.scheduler = TimeStepScheduler(config.action_ordering_seed)
        self.conflict_resolver = ConflictResolver(config.conflict_resolution_seed)

        # Action management
        self.pending_actions: Dict[CellID, CellAction] = {}
        self.completed_actions: List[Tuple[CellAction, ActionResult]] = []
        self.action_history: deque = deque(maxlen=config.max_action_history)

        # Performance and metrics
        self.metrics = CoordinationMetrics()
        self.performance_callbacks: List[Callable[[CoordinationMetrics], Awaitable[None]]] = []

        # Control and synchronization
        self.shutdown_event = asyncio.Event()
        self.pause_event = asyncio.Event()
        self.timestep_completion_event = asyncio.Event()

        # Error handling
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.failed_agent_ids: Set[CellID] = set()

    async def initialize(self) -> None:
        """Initialize the coordinator and all subsystems."""
        self.logger.info("Initializing DeterministicCoordinator")
        self.state = CoordinatorState.INITIALIZING

        try:
            # Initialize spatial index
            await self.spatial_index.initialize()

            # Initialize scheduler
            await self.scheduler.initialize()

            # Initialize conflict resolver
            await self.conflict_resolver.initialize()

            # Create initial simulation state
            self.simulation_state = SimulationState.create_initial_state(
                self.world_params,
                self.cell_params
            )

            self.state = CoordinatorState.RUNNING
            self.logger.info("Coordinator initialization complete")

        except Exception as e:
            self.state = CoordinatorState.ERROR
            self.last_error = e
            self.logger.error(f"Coordinator initialization failed: {e}")
            raise

    async def add_agent(self, agent: AsyncCellAgent) -> None:
        """Add a new cell agent to the simulation."""
        if agent.cell_id in self.agents:
            raise ValueError(f"Agent with ID {agent.cell_id} already exists")

        self.agents[agent.cell_id] = agent
        self.active_agents.add(agent.cell_id)

        # Add to spatial index
        cell_data = agent.current_data
        await self.spatial_index.add_cell(agent.cell_id, cell_data.position)

        # Update simulation state
        self.simulation_state = self.simulation_state.add_cell(cell_data)

        # Start agent task
        task = asyncio.create_task(self._run_agent(agent))
        self.agent_tasks[agent.cell_id] = task

        self.logger.debug(f"Added agent {agent.cell_id}")

    async def remove_agent(self, cell_id: CellID) -> None:
        """Remove a cell agent from the simulation."""
        if cell_id not in self.agents:
            return

        # Shutdown agent gracefully
        agent = self.agents[cell_id]
        await agent.shutdown()

        # Cancel agent task
        if cell_id in self.agent_tasks:
            task = self.agent_tasks[cell_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.agent_tasks[cell_id]

        # Remove from spatial index
        await self.spatial_index.remove_cell(cell_id)

        # Update state
        self.active_agents.discard(cell_id)
        del self.agents[cell_id]
        self.simulation_state = self.simulation_state.remove_cell(cell_id)

        self.logger.debug(f"Removed agent {cell_id}")

    async def run_simulation(self, max_timesteps: Optional[int] = None) -> None:
        """Run the main simulation loop."""
        self.logger.info("Starting simulation execution")

        if self.state != CoordinatorState.RUNNING:
            raise RuntimeError(f"Cannot run simulation in state {self.state}")

        max_steps = max_timesteps or self.config.max_timesteps

        try:
            while (self.current_timestep < max_steps and
                   not self.shutdown_event.is_set() and
                   self.active_agents):

                # Check for pause
                if self.pause_event.is_set():
                    await self.pause_event.wait()

                # Execute one timestep
                await self._execute_timestep()

                # Update timestep
                self.current_timestep = create_simulation_time(self.current_timestep + 1)

                # Periodic maintenance
                if self.current_timestep % self.config.gc_frequency == 0:
                    await self._perform_maintenance()

            self.logger.info(f"Simulation completed after {self.current_timestep} timesteps")

        except Exception as e:
            self.state = CoordinatorState.ERROR
            self.last_error = e
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            await self._cleanup_simulation()

    async def _execute_timestep(self) -> None:
        """Execute a single deterministic timestep."""
        timestep_start = time.time()

        try:
            # Step 1: Collect actions from all agents
            actions = await self._collect_agent_actions()

            # Step 2: Order actions deterministically
            ordered_actions = await self.scheduler.order_actions(actions, self.current_timestep)

            # Step 3: Resolve conflicts
            resolved_actions = await self.conflict_resolver.resolve_conflicts(
                ordered_actions, self.simulation_state
            )

            # Step 4: Execute actions in batches
            action_results = await self._execute_actions_batch(resolved_actions)

            # Step 5: Update simulation state
            await self._update_simulation_state(action_results)

            # Step 6: Update spatial index
            await self._update_spatial_index()

            # Step 7: Notify agents of state changes
            await self._notify_agents_of_updates()

            # Step 8: Collect metrics
            await self._collect_timestep_metrics(action_results)

            # Record performance
            timestep_duration = time.time() - timestep_start
            self.metrics.update_timestep_performance(timestep_duration, len(actions))
            self.metrics.total_timesteps += 1

        except Exception as e:
            self.error_count += 1
            self.last_error = e
            self.logger.error(f"Timestep {self.current_timestep} execution failed: {e}")

            if not self.config.continue_on_agent_failure:
                raise

    async def _collect_agent_actions(self) -> List[CellAction]:
        """Collect actions from all active agents."""
        actions = []
        timeout = self.config.timestep_timeout / 2  # Give agents half the timestep time

        # Collect actions with timeout
        action_tasks = []
        for agent_id in list(self.active_agents):
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                task = asyncio.create_task(self._get_agent_action(agent, timeout))
                action_tasks.append(task)

        # Wait for all actions with timeout
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*action_tasks, return_exceptions=True),
                timeout=timeout
            )

            for result in completed_tasks:
                if isinstance(result, Exception):
                    self.logger.warning(f"Agent action collection failed: {result}")
                elif result is not None:
                    actions.append(result)

        except asyncio.TimeoutError:
            self.logger.warning(f"Action collection timed out at timestep {self.current_timestep}")
            # Cancel remaining tasks
            for task in action_tasks:
                if not task.done():
                    task.cancel()

        return actions

    async def _get_agent_action(self, agent: AsyncCellAgent, timeout: float) -> Optional[CellAction]:
        """Get action from a specific agent with timeout."""
        try:
            # Check if agent has an action ready
            if not agent.action_queue.empty():
                return await asyncio.wait_for(agent.action_queue.get(), timeout=timeout)
            else:
                # Agent might be deciding - give it a moment
                await asyncio.sleep(0.001)
                if not agent.action_queue.empty():
                    return await asyncio.wait_for(agent.action_queue.get(), timeout=timeout)

        except asyncio.TimeoutError:
            self.logger.debug(f"Agent {agent.cell_id} action timeout")
        except Exception as e:
            self.logger.warning(f"Error getting action from agent {agent.cell_id}: {e}")

            # Mark agent as failed if too many errors
            if agent.error_count > self.config.max_action_retries:
                self.failed_agent_ids.add(agent.cell_id)
                await self._handle_failed_agent(agent.cell_id)

        return None

    async def _execute_actions_batch(self, actions: List[CellAction]) -> List[Tuple[CellAction, ActionResult]]:
        """Execute actions in deterministic batches."""
        results = []
        batch_size = self.config.batch_size

        # Process actions in deterministic batches
        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]
            batch_results = await self._execute_action_batch(batch)
            results.extend(batch_results)

        return results

    async def _execute_action_batch(self, batch: List[CellAction]) -> List[Tuple[CellAction, ActionResult]]:
        """Execute a single batch of actions."""
        results = []

        for action in batch:
            try:
                result = await self._execute_single_action(action)
                results.append((action, result))
                self.metrics.total_actions_processed += 1

            except Exception as e:
                self.logger.error(f"Action execution failed: {action}, error: {e}")
                failed_result = ActionResult(
                    status=ActionStatus.FAILED,
                    error_message=str(e)
                )
                results.append((action, failed_result))
                self.metrics.failed_actions_count += 1

        return results

    async def _execute_single_action(self, action: CellAction) -> ActionResult:
        """Execute a single cell action."""
        if action.action_type == ActionType.WAIT:
            return ActionResult(status=ActionStatus.SUCCESS)

        elif action.action_type == ActionType.MOVE:
            return await self._execute_move_action(action)

        elif action.action_type == ActionType.SWAP:
            return await self._execute_swap_action(action)

        else:
            return ActionResult(
                status=ActionStatus.FAILED,
                error_message=f"Unknown action type: {action.action_type}"
            )

    async def _execute_move_action(self, action: MoveAction) -> ActionResult:
        """Execute a move action."""
        cell_id = action.cell_id
        new_position = action.target_position

        # Validate move
        if not self.world_params.validate():
            return ActionResult(
                status=ActionStatus.FAILED,
                error_message="Invalid world parameters"
            )

        # Check boundaries
        if not (0 <= new_position.x < self.world_params.width and
                0 <= new_position.y < self.world_params.height):
            return ActionResult(
                status=ActionStatus.FAILED,
                error_message="Move outside world boundaries"
            )

        # Update cell position
        cell_data = self.simulation_state.cells.get(cell_id)
        if cell_data is None:
            return ActionResult(
                status=ActionStatus.FAILED,
                error_message=f"Cell {cell_id} not found"
            )

        # Create updated cell data
        updated_cell = cell_data.with_position(new_position)

        # Update simulation state
        self.simulation_state = self.simulation_state.update_cell(cell_id, updated_cell)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            resulting_state=updated_cell
        )

    async def _execute_swap_action(self, action: SwapAction) -> ActionResult:
        """Execute a swap action between two cells."""
        cell_id1 = action.cell_id
        cell_id2 = action.target_cell_id

        # Get both cells
        cell1 = self.simulation_state.cells.get(cell_id1)
        cell2 = self.simulation_state.cells.get(cell_id2)

        if cell1 is None or cell2 is None:
            return ActionResult(
                status=ActionStatus.FAILED,
                error_message="One or both cells not found for swap"
            )

        # Swap positions
        updated_cell1 = cell1.with_position(cell2.position)
        updated_cell2 = cell2.with_position(cell1.position)

        # Update simulation state
        self.simulation_state = self.simulation_state.update_cell(cell_id1, updated_cell1)
        self.simulation_state = self.simulation_state.update_cell(cell_id2, updated_cell2)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            resulting_state=(updated_cell1, updated_cell2)
        )

    async def _update_simulation_state(self, action_results: List[Tuple[CellAction, ActionResult]]) -> None:
        """Update the global simulation state based on action results."""
        successful_actions = [
            (action, result) for action, result in action_results
            if result.status == ActionStatus.SUCCESS
        ]

        # Update world state
        self.simulation_state.world_state.timestep = self.current_timestep
        self.simulation_state.world_state.total_actions += len(successful_actions)

        # Store action history
        self.action_history.extend(successful_actions)
        self.completed_actions.extend(successful_actions)

    async def _update_spatial_index(self) -> None:
        """Update the spatial index with current cell positions."""
        cell_positions = {
            cell_id: cell_data.position
            for cell_id, cell_data in self.simulation_state.cells.items()
        }
        await self.spatial_index.update_all_positions(cell_positions)

    async def _notify_agents_of_updates(self) -> None:
        """Notify all agents of simulation state updates."""
        notification_tasks = []

        for agent_id in list(self.active_agents):
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                task = asyncio.create_task(self._notify_single_agent(agent))
                notification_tasks.append(task)

        # Execute notifications concurrently
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)

    async def _notify_single_agent(self, agent: AsyncCellAgent) -> None:
        """Notify a single agent of updates."""
        try:
            # Update agent's cell data
            cell_data = self.simulation_state.cells.get(agent.cell_id)
            if cell_data:
                update_message = AgentMessage(
                    sender_id=CellID(-1),  # Coordinator ID
                    message_type="update_state",
                    data={
                        "cell_data": cell_data,
                        "timestep": self.current_timestep
                    },
                    timestep=self.current_timestep
                )
                await agent.receive_message(update_message)

            # Update neighbor information
            neighbors = await self.spatial_index.find_neighbors(
                agent.cell_id,
                self.cell_params.interaction_radius
            )

            neighbor_data = {
                neighbor_id: self.simulation_state.cells[neighbor_id]
                for neighbor_id in neighbors
                if neighbor_id in self.simulation_state.cells
            }

            neighbor_message = AgentMessage(
                sender_id=CellID(-1),
                message_type="neighbor_update",
                data={"neighbors": neighbor_data},
                timestep=self.current_timestep
            )
            await agent.receive_message(neighbor_message)

        except Exception as e:
            self.logger.warning(f"Failed to notify agent {agent.cell_id}: {e}")

    async def _collect_timestep_metrics(self, action_results: List[Tuple[CellAction, ActionResult]]) -> None:
        """Collect metrics for this timestep."""
        self.metrics.active_agents_count = len(self.active_agents)

        # Update conflict resolution metrics
        conflicts_resolved = await self.conflict_resolver.get_resolved_conflicts_count()
        self.metrics.total_conflicts_resolved += conflicts_resolved

        # Call performance callbacks
        for callback in self.performance_callbacks:
            try:
                await callback(self.metrics)
            except Exception as e:
                self.logger.warning(f"Performance callback failed: {e}")

    async def _perform_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        # Remove failed agents
        for agent_id in list(self.failed_agent_ids):
            await self.remove_agent(agent_id)
            self.failed_agent_ids.discard(agent_id)

        # Cleanup completed actions
        if len(self.completed_actions) > self.config.max_action_history:
            self.completed_actions = self.completed_actions[-self.config.max_action_history//2:]

        # Force garbage collection
        import gc
        gc.collect()

    async def _handle_failed_agent(self, agent_id: CellID) -> None:
        """Handle a permanently failed agent."""
        self.logger.warning(f"Agent {agent_id} marked as permanently failed")

        if self.config.freeze_on_error:
            agent = self.agents.get(agent_id)
            if agent:
                agent.agent_state = AgentState.ERROR

    async def _run_agent(self, agent: AsyncCellAgent) -> None:
        """Run a single agent's lifecycle."""
        try:
            await agent.run()
        except Exception as e:
            self.logger.error(f"Agent {agent.cell_id} failed: {e}")
            self.failed_agent_ids.add(agent.cell_id)

    async def _cleanup_simulation(self) -> None:
        """Clean up simulation resources."""
        self.state = CoordinatorState.STOPPING

        # Shutdown all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            shutdown_tasks.append(asyncio.create_task(agent.shutdown()))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Cancel all agent tasks
        for task in self.agent_tasks.values():
            if not task.done():
                task.cancel()

        if self.agent_tasks:
            await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)

        self.state = CoordinatorState.STOPPED
        self.logger.info("Coordinator cleanup complete")

    # Public control methods
    async def pause(self) -> None:
        """Pause the simulation."""
        self.pause_event.set()
        self.logger.info("Simulation paused")

    async def resume(self) -> None:
        """Resume the simulation."""
        self.pause_event.clear()
        self.logger.info("Simulation resumed")

    async def shutdown(self) -> None:
        """Gracefully shutdown the coordinator."""
        self.logger.info("Coordinator shutdown requested")
        self.shutdown_event.set()
        await self._cleanup_simulation()

    # Metrics and monitoring
    def add_performance_callback(self, callback: Callable[[CoordinationMetrics], Awaitable[None]]) -> None:
        """Add a performance monitoring callback."""
        self.performance_callbacks.append(callback)

    def get_metrics(self) -> CoordinationMetrics:
        """Get current coordination metrics."""
        return self.metrics

    def get_simulation_state(self) -> SimulationState:
        """Get current simulation state."""
        return self.simulation_state

    # Agent management utilities
    def get_agent_count(self) -> int:
        """Get current number of active agents."""
        return len(self.active_agents)

    def get_agent_states(self) -> Dict[CellID, AgentState]:
        """Get the states of all agents."""
        return {
            agent_id: agent.agent_state
            for agent_id, agent in self.agents.items()
        }

    def is_running(self) -> bool:
        """Check if the coordinator is currently running."""
        return self.state == CoordinatorState.RUNNING and not self.shutdown_event.is_set()