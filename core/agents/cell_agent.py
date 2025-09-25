"""Async cell agent implementation for morphogenesis simulation.

This module provides the AsyncCellAgent base class that replaces the
problematic threading.Thread approach with async coroutines for
deterministic, scientifically valid cell simulation.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    AsyncGenerator, Optional, Dict, Any, List, Tuple, Set,
    Union, Callable, Awaitable
)
from enum import Enum

from ..data.types import (
    CellID, Position, CellState, CellType, SimulationTime,
    CellParameters
)
from ..data.actions import (
    CellAction, WaitAction, MoveAction, SwapAction, 
    create_wait_action, create_move_action, create_swap_action
)
from ..data.state import CellData, SimulationState


class AgentState(Enum):
    """States of the async cell agent lifecycle."""
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WAITING = "waiting"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message passed between agent and coordinator."""
    sender_id: CellID
    message_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestep: Optional[SimulationTime] = None


@dataclass
class CellBehaviorConfig:
    """Configuration for cell behavior parameters."""
    # Core behavior settings
    behavior_type: str = "standard"
    decision_frequency: int = 1  # How often to make decisions (timesteps)
    action_delay: float = 0.0  # Delay before executing actions
    
    # Sorting behavior
    sorting_enabled: bool = True
    sorting_algorithm: str = "bubble"  # bubble, selection, insertion
    comparison_method: str = "value"  # value, position, type
    
    # Movement behavior  
    movement_enabled: bool = True
    max_movement_distance: float = 1.0
    movement_randomness: float = 0.1
    
    # Interaction behavior
    interaction_enabled: bool = True
    cooperation_probability: float = 0.8
    swap_willingness: float = 0.5
    
    # Learning and adaptation
    learning_enabled: bool = False
    learning_rate: float = 0.01
    memory_capacity: int = 100
    
    # Error handling
    error_recovery: bool = True
    max_retry_attempts: int = 3
    freeze_on_error: bool = True


class AsyncCellAgent(ABC):
    """Base class for async cell agents.
    
    This replaces threading.Thread with async coroutines to eliminate
    race conditions and provide deterministic execution. Each cell
    runs as an independent coroutine that yields control cooperatively.
    """
    
    def __init__(
        self, 
        cell_id: CellID,
        initial_data: CellData,
        behavior_config: CellBehaviorConfig,
        parameters: CellParameters,
        random_seed: Optional[int] = None
    ):
        self.cell_id = cell_id
        self.current_data = initial_data
        self.behavior_config = behavior_config
        self.parameters = parameters
        
        # Agent state management
        self.agent_state = AgentState.CREATED
        self.last_update_timestep = SimulationTime(0)
        self.decision_counter = 0
        
        # Async coordination
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.action_queue: asyncio.Queue[CellAction] = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        
        # Deterministic random number generator
        self.rng = random.Random()
        if random_seed is not None:
            self.rng.seed(random_seed + cell_id)  # Unique seed per cell
        
        # Performance tracking
        self.performance_metrics = {
            'decisions_made': 0,
            'actions_taken': 0,
            'processing_time': 0.0,
            'errors_encountered': 0,
        }
        
        # Error handling
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        
        # Internal state
        self._neighbors_cache: Dict[CellID, CellData] = {}
        self._local_environment: Dict[str, Any] = {}
        self._action_history: List[CellAction] = []
    
    async def run(self) -> None:
        """Main async lifecycle method.
        
        This is the primary coroutine that manages the cell's lifecycle.
        It coordinates initialization, decision-making, action execution,
        and cleanup in a deterministic, cooperative manner.
        """
        try:
            await self._initialize()
            
            # Main lifecycle loop
            async for action in self._lifecycle_generator():
                if self.shutdown_event.is_set():
                    break
                    
                # Queue action for coordinator
                await self.action_queue.put(action)
                
                # Update performance metrics
                self.performance_metrics['actions_taken'] += 1
                
                # Yield control to event loop
                await asyncio.sleep(0)
                
        except Exception as e:
            await self._handle_error(e)
        finally:
            await self._cleanup()
    
    async def _lifecycle_generator(self) -> AsyncGenerator[CellAction, None]:
        """Generate sequence of actions for this cell.
        
        This is the core async generator that yields actions based on
        cell state, environment, and behavior configuration.
        """
        self.agent_state = AgentState.ACTIVE
        
        while not self.shutdown_event.is_set() and self.current_data.is_alive():
            try:
                # Process incoming messages
                await self._process_messages()
                
                # Update internal state
                await self._update_internal_state()
                
                # Make behavioral decision
                if self._should_make_decision():
                    action = await self._decide_action()
                    self.decision_counter += 1
                    self.performance_metrics['decisions_made'] += 1
                    
                    # Add to action history
                    self._action_history.append(action)
                    if len(self._action_history) > 100:  # Limit history size
                        self._action_history = self._action_history[-100:]
                    
                    yield action
                else:
                    # Yield wait action
                    yield create_wait_action(self.cell_id, self.last_update_timestep)
                
                # Age the cell
                await self._age_cell()
                
                # Cooperative yield
                await asyncio.sleep(0)
                
            except Exception as e:
                await self._handle_error(e)
                if self.error_count > self.behavior_config.max_retry_attempts:
                    break
    
    async def _initialize(self) -> None:
        """Initialize the cell agent."""
        self.agent_state = AgentState.INITIALIZING
        
        # Perform any setup needed by specific behaviors
        await self._setup_behavior()
        
        # Initialize local environment understanding
        await self._update_local_environment()
        
        self.agent_state = AgentState.ACTIVE
    
    async def _setup_behavior(self) -> None:
        """Setup behavior-specific initialization.
        
        Override in subclasses for specific behavior setup.
        """
        pass
    
    async def _process_messages(self) -> None:
        """Process messages from coordinator or other agents."""
        try:
            # Process all queued messages without blocking
            while not self.message_queue.empty():
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=0.001
                )
                await self._handle_message(message)
        except asyncio.TimeoutError:
            pass  # No messages to process
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle a specific message.
        
        Override in subclasses for message-specific handling.
        """
        if message.message_type == "update_state":
            await self._update_from_coordinator(message.data)
        elif message.message_type == "neighbor_update":
            await self._update_neighbors(message.data)
        elif message.message_type == "environment_update":
            await self._update_local_environment(message.data)
    
    async def _update_internal_state(self) -> None:
        """Update internal agent state."""
        # Update data from any external changes
        # This would typically come from coordinator updates
        pass
    
    def _should_make_decision(self) -> bool:
        """Determine if the cell should make a decision this timestep."""
        return (
            self.decision_counter % self.behavior_config.decision_frequency == 0 and
            self.current_data.is_active() and
            self.agent_state == AgentState.ACTIVE
        )
    
    @abstractmethod
    async def _decide_action(self) -> CellAction:
        """Decide what action to take based on current state.
        
        This is the core decision-making method that must be implemented
        by specific cell behavior classes.
        """
        pass
    
    async def _age_cell(self) -> None:
        """Age the cell and handle lifecycle transitions."""
        new_age = self.current_data.age + 1
        self.current_data = self.current_data.with_age(new_age)
        
        # Check for natural death
        if new_age >= self.parameters.max_age:
            if self.rng.random() < self.parameters.death_probability:
                self.current_data = self.current_data.with_state(CellState.DYING)
    
    async def _update_from_coordinator(self, update_data: Dict[str, Any]) -> None:
        """Update cell state from coordinator."""
        if 'cell_data' in update_data:
            self.current_data = update_data['cell_data']
        if 'timestep' in update_data:
            self.last_update_timestep = update_data['timestep']
    
    async def _update_neighbors(self, neighbor_data: Dict[str, Any]) -> None:
        """Update neighbor information."""
        if 'neighbors' in neighbor_data:
            self._neighbors_cache = neighbor_data['neighbors']
    
    async def _update_local_environment(self, env_data: Optional[Dict[str, Any]] = None) -> None:
        """Update local environment understanding."""
        if env_data:
            self._local_environment.update(env_data)
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors in cell execution."""
        self.error_count += 1
        self.last_error = error
        self.performance_metrics['errors_encountered'] += 1
        
        if self.behavior_config.freeze_on_error:
            self.agent_state = AgentState.ERROR
            self.current_data = self.current_data.with_state(CellState.FROZEN)
        
        # Log error (would integrate with logging system)
        print(f"Cell {self.cell_id} error: {error}")
    
    async def _cleanup(self) -> None:
        """Cleanup cell agent resources."""
        self.agent_state = AgentState.STOPPING
        
        # Clear queues
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.agent_state = AgentState.STOPPED
    
    # Communication methods
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to the coordinator."""
        # This would be handled by the coordinator interface
        pass
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from the coordinator."""
        await self.message_queue.put(message)
    
    # Utility methods for subclasses
    def get_neighbors(self) -> Dict[CellID, CellData]:
        """Get current neighbors from cache."""
        return self._neighbors_cache.copy()
    
    def get_neighbors_by_type(self, cell_type: CellType) -> Dict[CellID, CellData]:
        """Get neighbors of a specific type."""
        return {
            cid: cell for cid, cell in self._neighbors_cache.items()
            if cell.cell_type == cell_type
        }
    
    def get_closest_neighbor(self) -> Optional[Tuple[CellID, CellData]]:
        """Get the closest neighbor."""
        if not self._neighbors_cache:
            return None
        
        closest_id = None
        closest_distance = float('inf')
        
        for neighbor_id, neighbor_data in self._neighbors_cache.items():
            distance = self.current_data.distance_to(neighbor_data)
            if distance < closest_distance:
                closest_distance = distance
                closest_id = neighbor_id
        
        if closest_id is not None:
            return closest_id, self._neighbors_cache[closest_id]
        return None
    
    def calculate_local_density(self) -> float:
        """Calculate local cell density around this cell."""
        if not self._neighbors_cache:
            return 0.0
        
        area = 3.14159 * (self.parameters.interaction_radius ** 2)
        return len(self._neighbors_cache) / area
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            'agent_state': self.agent_state.value,
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'decision_counter': self.decision_counter,
            'neighbors_count': len(self._neighbors_cache),
        }
    
    # Shutdown handling
    async def shutdown(self) -> None:
        """Gracefully shutdown the cell agent."""
        self.shutdown_event.set()
        
        # Wait a bit for graceful shutdown
        try:
            await asyncio.wait_for(self._wait_for_shutdown(), timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Force shutdown
    
    async def _wait_for_shutdown(self) -> None:
        """Wait for agent to reach stopped state."""
        while self.agent_state not in {AgentState.STOPPED, AgentState.ERROR}:
            await asyncio.sleep(0.01)


class StandardCellAgent(AsyncCellAgent):
    """Standard cell agent with basic sorting behavior.
    
    This implements a simple cell that participates in sorting
    algorithms by comparing values with neighbors and deciding
    whether to swap positions.
    """
    
    async def _setup_behavior(self) -> None:
        """Setup standard cell behavior."""
        # Initialize sorting-specific state
        self._last_comparison_result: Optional[bool] = None
        self._swap_attempts = 0
        self._successful_swaps = 0
    
    async def _decide_action(self) -> CellAction:
        """Decide action based on sorting algorithm."""
        if not self.behavior_config.sorting_enabled:
            return create_wait_action(self.cell_id, self.last_update_timestep)
        
        # Get neighbors for comparison
        neighbors = self.get_neighbors()
        if not neighbors:
            return create_wait_action(self.cell_id, self.last_update_timestep)
        
        # Implement basic bubble sort behavior
        if self.behavior_config.sorting_algorithm == "bubble":
            return await self._bubble_sort_decision(neighbors)
        elif self.behavior_config.sorting_algorithm == "selection":
            return await self._selection_sort_decision(neighbors)
        else:
            return create_wait_action(self.cell_id, self.last_update_timestep)
    
    async def _bubble_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Make decision based on bubble sort algorithm."""
        # Find neighbor to potentially swap with
        best_swap_candidate = None
        best_improvement = 0.0
        
        for neighbor_id, neighbor_data in neighbors.items():
            # Calculate if swapping would improve sorting
            improvement = self._calculate_swap_benefit(neighbor_data)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap_candidate = neighbor_id
        
        # Decide whether to swap
        if (best_swap_candidate is not None and 
            best_improvement > 0 and 
            self.rng.random() < self.behavior_config.swap_willingness):
            
            self._swap_attempts += 1
            return create_swap_action(
                self.cell_id, 
                self.last_update_timestep, 
                best_swap_candidate,
                probability=min(1.0, best_improvement)
            )
        
        return create_wait_action(self.cell_id, self.last_update_timestep)
    
    async def _selection_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Make decision based on selection sort algorithm."""
        # Find the neighbor with the most extreme sort value
        if self.behavior_config.comparison_method == "value":
            best_neighbor = min(
                neighbors.items(),
                key=lambda item: item[1].sort_value
            )
            
            # If we should be before this neighbor, try to swap
            if self.current_data.sort_value > best_neighbor[1].sort_value:
                return create_swap_action(
                    self.cell_id,
                    self.last_update_timestep,
                    best_neighbor[0]
                )
        
        return create_wait_action(self.cell_id, self.last_update_timestep)
    
    def _calculate_swap_benefit(self, neighbor_data: CellData) -> float:
        """Calculate the benefit of swapping with a neighbor."""
        if self.behavior_config.comparison_method == "value":
            # Simple comparison based on sort values
            my_value = self.current_data.sort_value
            their_value = neighbor_data.sort_value
            
            # Benefit if we're out of order (higher value but should be lower position)
            if my_value > their_value:
                return abs(my_value - their_value)
        
        return 0.0