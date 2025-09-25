"""Async cellular agent implementation for morphogenesis research.

**Biological Foundation:**
Real cells are autonomous agents that sense their environment, make decisions,
and act based on genetic programs and local conditions. During morphogenesis
(the development of form and structure), cells coordinate their behavior to
create complex tissues and organs through purely local interactions.

**Scientific Innovation:**
This module implements computational cells as async agents, eliminating the
race conditions and non-deterministic behavior that plague traditional threaded
cell simulations. Each cell runs as an independent coroutine that makes
biologically-inspired decisions while maintaining scientific reproducibility.

**Key Biological Processes Modeled:**
- **Environmental Sensing**: Cells detect neighbors and local chemical gradients
- **Decision Making**: Cells choose actions based on developmental programs
- **Cooperative Behavior**: Cells coordinate through signaling and spatial cues
- **Self-Organization**: Emergent tissue patterns arise from individual cell actions

**Research Applications:**
- Studying how cells self-organize during embryonic development
- Modeling tissue repair and wound healing processes
- Understanding cancer cell migration and metastasis
- Investigating stem cell differentiation and tissue engineering

Example:
    >>> import asyncio
    >>> from core.agents.cell_agent import StandardCellAgent
    >>> from core.data.types import CellParameters, Position, CellID
    >>>
    >>> # Create a research-grade cellular agent
    >>> cell_params = CellParameters(interaction_radius=10.0, max_speed=2.0)
    >>> behavior_config = CellBehaviorConfig(
    ...     sorting_enabled=True,
    ...     cooperation_probability=0.8
    ... )
    >>> position = Position(x=50, y=50)
    >>> cell_data = CellData.create_initial(CellID(1), position, cell_params)
    >>>
    >>> # Create autonomous cell agent
    >>> cell_agent = StandardCellAgent(
    ...     cell_id=CellID(1),
    ...     initial_data=cell_data,
    ...     behavior_config=behavior_config,
    ...     parameters=cell_params,
    ...     random_seed=42
    ... )
    >>>
    >>> # Run cellular lifecycle simulation
    >>> async def simulate_cell_development():
    ...     await cell_agent.run()
    >>>
    >>> asyncio.run(simulate_cell_development())
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
    """Lifecycle states of autonomous cellular agents during morphogenesis.

    These states mirror the biological lifecycle of cells during development,
    from initial creation through active participation in tissue formation
    to eventual death or quiescence. Understanding these states is crucial
    for researchers analyzing cellular behavior and development patterns.

    States:
        CREATED: Cell agent instantiated but not yet initialized
        INITIALIZING: Cell setting up internal systems and sensing environment
        ACTIVE: Cell actively participating in morphogenetic processes
        WAITING: Cell temporarily inactive, awaiting environmental cues
        PROCESSING: Cell computing complex behavioral decisions
        PAUSED: Cell suspended by external control (useful for debugging)
        STOPPING: Cell beginning graceful shutdown process
        STOPPED: Cell completely terminated, all resources cleaned up
        ERROR: Cell encountered unrecoverable error, requires investigation

    Example:
        >>> cell_agent = StandardCellAgent(...)
        >>> print(cell_agent.agent_state)  # AgentState.CREATED
        >>> await cell_agent._initialize()
        >>> print(cell_agent.agent_state)  # AgentState.ACTIVE
    """
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
    """Communication message between cellular agents and the coordination system.

    In biological systems, cells communicate through biochemical signals,
    mechanical forces, and direct contact. This message system simulates
    that intercellular communication, enabling coordination and collective
    behavior during morphogenetic processes.

    The messaging system supports various types of cellular communication
    including environmental updates, neighbor notifications, and coordination
    signals that enable emergent tissue organization.

    Attributes:
        sender_id: Unique identifier of the sending cell (-1 for coordinator)
        message_type: Type of biological signal or coordination message
        data: Payload containing signal-specific information
        timestep: Simulation time when message was sent (for temporal analysis)

    Common Message Types:
        - "update_state": Cell state synchronization from coordinator
        - "neighbor_update": Information about nearby cells
        - "environment_update": Changes in local biochemical environment
        - "division_signal": Cell division coordination
        - "death_signal": Apoptosis notification to neighbors
        - "chemical_gradient": Morphogen concentration updates

    Example:
        >>> # Coordinator updating cell about its new position
        >>> update_msg = AgentMessage(
        ...     sender_id=CellID(-1),  # Coordinator
        ...     message_type="update_state",
        ...     data={"position": new_position, "cell_data": updated_data},
        ...     timestep=current_timestep
        ... )
        >>>
        >>> # Cell signaling division to neighbors
        >>> division_msg = AgentMessage(
        ...     sender_id=CellID(123),
        ...     message_type="division_signal",
        ...     data={"division_axis": axis_vector, "daughter_positions": positions},
        ...     timestep=current_timestep
        ... )
    """
    sender_id: CellID
    message_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestep: Optional[SimulationTime] = None


@dataclass
class CellBehaviorConfig:
    """Configuration parameters for cellular behavior during morphogenesis.

    This configuration defines how cells behave during development, controlling
    everything from basic movement patterns to complex cooperative behaviors.
    These parameters are inspired by real cellular biology, where cells exhibit
    diverse behaviors depending on their type, developmental stage, and environment.

    **Biological Relevance:**
    Different cell types exhibit distinct behaviors - neural cells migrate along
    specific paths, epithelial cells form tight sheets, and mesenchymal cells
    move individually. This configuration allows researchers to model these
    diverse cellular behaviors accurately.

    Attributes:
        behavior_type: Primary behavioral phenotype (e.g., "epithelial", "mesenchymal", "neural")
        decision_frequency: How often cells evaluate their environment (timesteps).
                          Mirrors biological decision-making cycles.
        action_delay: Delay between decision and action (seconds).
                     Models biological response latency.
        sorting_enabled: Whether cells participate in tissue sorting processes.
                        Critical for studying cell segregation and pattern formation.
        sorting_algorithm: Algorithm for cell sorting behavior ("bubble", "selection", "insertion").
                          Different algorithms model different biological sorting mechanisms.
        comparison_method: How cells compare themselves to neighbors ("value", "position", "type").
                         Reflects different biological sorting cues.
        movement_enabled: Whether cells can actively migrate.
                         Essential for studying cell migration during development.
        max_movement_distance: Maximum distance per timestep (biological units).
                              Reflects cellular locomotion capabilities.
        movement_randomness: Degree of random movement (0.0-1.0).
                           Models stochastic cellular motility.
        interaction_enabled: Whether cells respond to neighbors.
                           Critical for intercellular communication studies.
        cooperation_probability: Likelihood of cooperative behavior (0.0-1.0).
                               Models altruistic cellular behavior.
        swap_willingness: Probability of position swapping with neighbors (0.0-1.0).
                         Important for tissue rearrangement studies.
        learning_enabled: Whether cells adapt their behavior over time.
                         Models cellular memory and adaptation.
        learning_rate: Speed of behavioral adaptation (0.0-1.0).
                      Reflects plasticity in cellular responses.
        memory_capacity: Number of past events cells remember.
                        Models cellular memory systems.
        error_recovery: Whether cells can recover from errors.
                       Models biological robustness and repair mechanisms.
        max_retry_attempts: Maximum attempts to recover from errors.
                          Reflects cellular resilience limits.
        freeze_on_error: Whether to freeze cells on unrecoverable errors.
                        Useful for debugging developmental abnormalities.

    Example:
        >>> # Configuration for epithelial cells (form sheets, low motility)
        >>> epithelial_config = CellBehaviorConfig(
        ...     behavior_type="epithelial",
        ...     movement_enabled=True,
        ...     max_movement_distance=0.5,  # Low motility
        ...     cooperation_probability=0.9,  # Highly cooperative
        ...     sorting_enabled=True
        ... )
        >>>
        >>> # Configuration for neural cells (high motility, path-following)
        >>> neural_config = CellBehaviorConfig(
        ...     behavior_type="neural",
        ...     movement_enabled=True,
        ...     max_movement_distance=2.0,  # High motility
        ...     movement_randomness=0.05,  # Directed movement
        ...     learning_enabled=True  # Adaptive behavior
        ... )
    """
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
    """Base class for autonomous cellular agents in morphogenesis simulations.

    **Biological Inspiration:**
    Real cells are autonomous agents that continuously sense their environment,
    make decisions based on genetic programs and local conditions, and execute
    actions that contribute to tissue development. This class models that
    cellular autonomy while ensuring scientific reproducibility.

    **Scientific Innovation:**
    Traditional threaded cell simulations suffer from race conditions that make
    results non-reproducible - a fatal flaw for peer-reviewed research. This
    async-based design eliminates those issues while maintaining the autonomous
    nature of cellular behavior that drives morphogenetic processes.

    **Key Cellular Behaviors Modeled:**
    - **Environmental Sensing**: Continuous monitoring of local neighborhood
    - **Decision Making**: Biologically-inspired behavioral choices
    - **Action Execution**: Movement, division, signaling, and death
    - **Communication**: Intercellular signaling and coordination
    - **Adaptation**: Learning and behavioral modification over time
    - **Lifecycle Management**: Birth, growth, reproduction, and death

    **Research Applications:**
    - Developmental biology studies of tissue formation
    - Cancer research modeling metastasis and tumor growth
    - Regenerative medicine simulating wound healing
    - Tissue engineering optimizing scaffold designs
    - Systems biology understanding emergent properties

    Args:
        cell_id: Unique identifier for this cellular agent
        initial_data: Complete biological state data for the cell
        behavior_config: Configuration controlling cellular behavior patterns
        parameters: Physical and biological parameters for the cell type
        random_seed: Seed for deterministic random behavior (scientific reproducibility)

    Example:
        >>> # Create a custom cell agent for tissue morphogenesis research
        >>> class NeuralCellAgent(AsyncCellAgent):
        ...     async def _decide_action(self) -> CellAction:
        ...         # Neural cells migrate along chemical gradients
        ...         neighbors = self.get_neighbors()
        ...         gradient_direction = self._detect_gradient(neighbors)
        ...         if gradient_direction:
        ...             target_pos = self.current_data.position + gradient_direction
        ...             return create_move_action(self.cell_id, self.last_update_timestep, target_pos)
        ...         return create_wait_action(self.cell_id, self.last_update_timestep)
        >>>
        >>> # Create and run neural development simulation
        >>> neural_cell = NeuralCellAgent(
        ...     cell_id=CellID(1),
        ...     initial_data=neural_cell_data,
        ...     behavior_config=neural_behavior_config,
        ...     parameters=neural_cell_parameters,
        ...     random_seed=12345
        ... )
        >>>
        >>> # Run complete cellular lifecycle
        >>> await neural_cell.run()

    Note:
        Subclasses must implement the _decide_action() method to define specific
        cellular behaviors. The async architecture ensures deterministic execution
        essential for reproducible morphogenesis research.
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