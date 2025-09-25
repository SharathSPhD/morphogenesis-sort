Core API Reference
==================

The Core API provides the fundamental building blocks for morphogenesis simulations, including cellular agents, coordination mechanisms, data types, and metrics collection.

Overview
--------

The core module contains:

* **Agents**: Cellular agents with various behaviors (sorting, adaptive, morphogen-based)
* **Coordination**: Systems for managing agent interactions and spatial organization
* **Data**: Type definitions, state management, and serialization
* **Metrics**: Performance monitoring and data collection

Core Coordination
------------------

The coordination system manages agent interactions, scheduling, and spatial organization.

Coordinator
~~~~~~~~~~~

.. automodule:: core.coordination.coordinator
   :members:
   :undoc-members:
   :show-inheritance:

The main coordinator manages the overall simulation state and agent interactions.

**Key Features:**

* Deterministic agent execution order
* Conflict resolution for spatial overlaps
* State synchronization across agents
* Performance optimization through spatial indexing

**Example Usage:**

.. code-block:: python

   from core.coordination.coordinator import DeterministicCoordinator

   coordinator = DeterministicCoordinator(
       grid_size=(100, 100),
       max_agents=1000
   )

   # Add agents to the simulation
   await coordinator.add_agent(cell_agent)

   # Run simulation step
   await coordinator.step()

Scheduler
~~~~~~~~~

.. automodule:: core.coordination.scheduler
   :members:
   :undoc-members:
   :show-inheritance:

The scheduler determines the order of agent actions within each timestep.

**Scheduling Strategies:**

* **Deterministic**: Fixed order based on agent IDs
* **Random**: Randomized order with fixed seed for reproducibility
* **Priority**: Order based on agent priority levels

Spatial Index
~~~~~~~~~~~~~

.. automodule:: core.coordination.spatial_index
   :members:
   :undoc-members:
   :show-inheritance:

Efficient spatial indexing for neighbor queries and collision detection.

**Spatial Data Structures:**

* **Grid-based**: O(1) lookups for regular grids
* **Quadtree**: Adaptive spatial partitioning for irregular distributions
* **R-tree**: Efficient range queries for complex shapes

Conflict Resolver
~~~~~~~~~~~~~~~~~

.. automodule:: core.coordination.conflict_resolver
   :members:
   :undoc-members:
   :show-inheritance:

Handles conflicts when multiple agents attempt to occupy the same space.

**Conflict Resolution Strategies:**

* **First-come-first-served**: Based on agent execution order
* **Priority-based**: Higher priority agents win conflicts
* **Negotiation**: Agents can negotiate alternative positions

Core Agents
-----------

Cellular agents are the fundamental units of morphogenesis simulations.

Base Cell Agent
~~~~~~~~~~~~~~~

.. automodule:: core.agents.cell_agent
   :members:
   :undoc-members:
   :show-inheritance:

The base cell agent provides common functionality for all cellular behaviors.

**Agent Lifecycle:**

1. **Initialization**: Set initial position, state, and parameters
2. **Perception**: Gather information about local environment and neighbors
3. **Decision**: Choose actions based on behavior rules and local state
4. **Action**: Execute movement, communication, or state changes
5. **Update**: Update internal state and prepare for next timestep

**Example Cell Agent:**

.. code-block:: python

   from core.agents.cell_agent import AsyncCellAgent

   class CustomCell(AsyncCellAgent):
       async def perceive(self):
           # Gather local information
           neighbors = await self.get_neighbors()
           local_density = len(neighbors)
           return {'neighbors': neighbors, 'density': local_density}

       async def decide(self, perception):
           # Make decisions based on perception
           if perception['density'] > 5:
               return {'action': 'move', 'direction': 'away_from_crowd'}
           return {'action': 'stay'}

       async def act(self, decision):
           # Execute the decision
           if decision['action'] == 'move':
               await self.move(decision['direction'])

Sorting Cell Behavior
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: core.agents.behaviors.sorting_cell
   :members:
   :undoc-members:
   :show-inheritance:

Cells that sort themselves based on type similarity.

**Sorting Algorithm:**

1. Identify neighbors of different types
2. Calculate local disorder (mixing) level
3. If disorder exceeds threshold, move toward similar cells
4. Update position to reduce local mixing

**Parameters:**

* ``sorting_strength``: How strongly cells prefer similar neighbors (0.0-1.0)
* ``movement_speed``: Maximum distance per timestep
* ``perception_radius``: How far cells can sense neighbors

Adaptive Cell Behavior
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: core.agents.behaviors.adaptive_cell
   :members:
   :undoc-members:
   :show-inheritance:

Cells that adapt their behavior based on local conditions and learning.

**Adaptive Features:**

* **Learning**: Update behavior rules based on experience
* **Memory**: Remember past states and decisions
* **Plasticity**: Adjust parameters based on environmental feedback
* **Anticipation**: Predict future states and plan accordingly

Morphogen Cell Behavior
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: core.agents.behaviors.morphogen_cell
   :members:
   :undoc-members:
   :show-inheritance:

Cells that respond to morphogen gradients for pattern formation.

**Morphogen System:**

* **Production**: Cells can produce morphogens
* **Diffusion**: Morphogens spread through the environment
* **Response**: Cells change behavior based on morphogen concentration
* **Degradation**: Morphogens decay over time

Core Data Types
---------------

Standard data types and structures used throughout the system.

Types
~~~~~

.. automodule:: core.data.types
   :members:
   :undoc-members:
   :show-inheritance:

Fundamental data types for positions, states, and actions.

**Core Types:**

* ``Position2D``: 2D coordinates with validation
* ``CellType``: Enumeration of cellular types
* ``AgentState``: Complete agent state representation
* ``ActionType``: Available agent actions

State Management
~~~~~~~~~~~~~~~~

.. automodule:: core.data.state
   :members:
   :undoc-members:
   :show-inheritance:

State management and persistence for agents and simulations.

**State Features:**

* **Immutable states**: Prevent accidental modifications
* **State transitions**: Validated state changes
* **History tracking**: Maintain state evolution over time
* **Checkpointing**: Save and restore simulation states

Actions
~~~~~~~

.. automodule:: core.data.actions
   :members:
   :undoc-members:
   :show-inheritance:

Action definitions and validation for agent behaviors.

**Action Categories:**

* **Movement**: Spatial position changes
* **Communication**: Information exchange between agents
* **State Change**: Internal state modifications
* **Environment**: Interactions with the environment

Serialization
~~~~~~~~~~~~~

.. automodule:: core.data.serialization
   :members:
   :undoc-members:
   :show-inheritance:

Efficient serialization and deserialization of simulation data.

**Supported Formats:**

* **JSON**: Human-readable format for configuration
* **Pickle**: Python-native binary format
* **HDF5**: High-performance scientific data format
* **Parquet**: Columnar format for analysis

Core Metrics
------------

Performance monitoring and data collection systems.

Metrics Collector
~~~~~~~~~~~~~~~~~

.. automodule:: core.metrics.collector
   :members:
   :undoc-members:
   :show-inheritance:

Central collection point for all simulation metrics.

**Metric Types:**

* **Performance**: Timing, memory usage, throughput
* **Scientific**: Emergence, order parameters, correlations
* **System**: Error rates, resource utilization
* **Custom**: User-defined domain-specific metrics

Snapshots
~~~~~~~~~

.. automodule:: core.metrics.snapshots
   :members:
   :undoc-members:
   :show-inheritance:

Point-in-time snapshots of complete simulation state.

**Snapshot Features:**

* **Full state capture**: All agent positions and states
* **Compressed storage**: Efficient storage of large states
* **Incremental snapshots**: Only store changes between timesteps
* **Restoration**: Restart simulations from any snapshot

Exporters
~~~~~~~~~

.. automodule:: core.metrics.exporters
   :members:
   :undoc-members:
   :show-inheritance:

Export metrics and data to various formats for analysis.

**Export Formats:**

* **CSV**: Tabular data for spreadsheet analysis
* **JSON**: Structured data for web applications
* **HDF5**: High-performance scientific computing
* **Parquet**: Big data analysis with Pandas/Dask

Counters
~~~~~~~~

.. automodule:: core.metrics.counters
   :members:
   :undoc-members:
   :show-inheritance:

High-performance counters for tracking events and quantities.

**Counter Types:**

* **Event counters**: Track discrete events (collisions, births, deaths)
* **Accumulating counters**: Track cumulative quantities (total distance moved)
* **Histogram counters**: Track distributions (agent speeds, neighbor counts)
* **Time series counters**: Track values over time

Example: Building a Custom Simulation
--------------------------------------

Here's a complete example showing how to use the Core API to build a custom morphogenesis simulation:

.. code-block:: python

   import asyncio
   from core.coordination.coordinator import DeterministicCoordinator
   from core.agents.behaviors.sorting_cell import SortingCell
   from core.data.types import Position2D, CellType
   from core.metrics.collector import MetricsCollector

   async def run_custom_simulation():
       # Initialize coordinator
       coordinator = DeterministicCoordinator(
           grid_size=(50, 50),
           max_agents=200
       )

       # Initialize metrics collection
       metrics = MetricsCollector(
           collect_snapshots=True,
           collect_performance=True
       )
       coordinator.set_metrics_collector(metrics)

       # Create mixed population of cells
       for i in range(100):
           cell_type = CellType.TYPE_A if i < 50 else CellType.TYPE_B
           position = Position2D(
               x=random.randint(0, 49),
               y=random.randint(0, 49)
           )

           cell = SortingCell(
               agent_id=i,
               initial_position=position,
               cell_type=cell_type,
               sorting_strength=0.8
           )

           await coordinator.add_agent(cell)

       # Run simulation
       for timestep in range(1000):
           await coordinator.step()

           if timestep % 100 == 0:
               # Take snapshot every 100 timesteps
               await metrics.take_snapshot(f"step_{timestep}")

       # Analyze results
       final_state = await coordinator.get_state()
       sorting_quality = calculate_sorting_quality(final_state)

       print(f"Final sorting quality: {sorting_quality:.3f}")

       # Export data for further analysis
       await metrics.export_to_csv("simulation_results.csv")
       await metrics.export_to_hdf5("simulation_snapshots.h5")

   # Run the simulation
   asyncio.run(run_custom_simulation())

Performance Considerations
--------------------------

**Optimization Guidelines:**

1. **Batch Operations**: Group similar operations together
2. **Spatial Indexing**: Use appropriate spatial data structures
3. **Memory Management**: Reuse objects when possible
4. **Async Patterns**: Avoid blocking operations in agent code
5. **Profiling**: Use built-in performance monitoring

**Scaling Recommendations:**

* **Small simulations** (< 100 agents): Any configuration works
* **Medium simulations** (100-1000 agents): Enable spatial indexing
* **Large simulations** (1000+ agents): Use optimized data structures and consider distributed computing

**Common Pitfalls:**

* Avoid synchronous I/O operations in agent code
* Don't create new objects in tight loops
* Use appropriate data structures for spatial queries
* Monitor memory usage for long-running simulations