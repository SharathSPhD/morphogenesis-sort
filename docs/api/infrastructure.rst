Infrastructure API Reference
=============================

The Infrastructure API provides essential support systems for morphogenesis simulations, including logging, performance monitoring, testing infrastructure, and system utilities.

Overview
--------

The infrastructure module contains:

* **Logging**: Comprehensive logging system for debugging and analysis
* **Performance Monitoring**: Real-time performance tracking and optimization
* **Testing Infrastructure**: Tools for automated testing and validation
* **System Utilities**: Platform-specific optimizations and resource management

Logging System
--------------

Comprehensive logging infrastructure for debugging, monitoring, and analysis.

Logger
~~~~~~

.. automodule:: infrastructure.logging.logger
   :members:
   :undoc-members:
   :show-inheritance:

Main logging interface providing structured logging capabilities.

**Key Features:**

* **Structured Logging**: JSON-formatted logs with consistent schema
* **Multiple Outputs**: Console, file, database, and remote logging
* **Performance Tracking**: Built-in timing and performance metrics
* **Context Management**: Automatic context propagation across async calls
* **Log Aggregation**: Centralized collection from distributed components

**Basic Usage:**

.. code-block:: python

   from infrastructure.logging.logger import MorphogenesisLogger

   # Create logger instance
   logger = MorphogenesisLogger(
       name="cellular_simulation",
       level="INFO",
       enable_performance_tracking=True
   )

   # Basic logging
   logger.info("Starting cellular simulation")
   logger.warning("Population size exceeds recommended limits")
   logger.error("Agent collision detected", agent_id=123, position=(45, 67))

   # Structured logging with context
   with logger.context(experiment_id="exp_001", simulation_step=500):
       logger.info("Processing agent interactions")
       logger.debug("Agent state update", agent_id=456, new_state="moving")

**Advanced Logging Features:**

.. code-block:: python

   # Performance timing
   with logger.timer("simulation_step"):
       await run_simulation_step()

   # Metric tracking
   logger.metric("agent_count", 1000)
   logger.metric("convergence_rate", 0.85)
   logger.metric("memory_usage_mb", 245.6)

   # Custom log levels
   logger.log_level("EMERGENCE", level=25)  # Between INFO and WARNING
   logger.emergence("Collective behavior detected", emergence_score=0.78)

   # Batch logging for performance
   log_buffer = logger.create_buffer()
   for agent in agents:
       log_buffer.debug("Agent position update",
                       agent_id=agent.id,
                       position=agent.position)
   await log_buffer.flush()

**Configuration:**

.. code-block:: python

   # Configure from dictionary
   logger_config = {
       'version': 1,
       'formatters': {
           'structured': {
               'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
               'class': 'infrastructure.logging.formatters.StructuredFormatter'
           }
       },
       'handlers': {
           'console': {
               'class': 'logging.StreamHandler',
               'formatter': 'structured',
               'level': 'INFO'
           },
           'file': {
               'class': 'logging.FileHandler',
               'filename': 'morphogenesis.log',
               'formatter': 'structured',
               'level': 'DEBUG'
           }
       },
       'loggers': {
           'morphogenesis': {
               'handlers': ['console', 'file'],
               'level': 'DEBUG',
               'propagate': False
           }
       }
   }

   logger = MorphogenesisLogger.from_config(logger_config)

Async Logger
~~~~~~~~~~~~

.. automodule:: infrastructure.logging.async_logger
   :members:
   :undoc-members:
   :show-inheritance:

High-performance asynchronous logging optimized for concurrent simulations.

**Async Features:**

* **Non-blocking**: Logging doesn't block simulation execution
* **Batched Writes**: Efficiently batch log messages for I/O
* **Queue Management**: Handle backpressure and memory limits
* **Context Propagation**: Maintain context across async tasks

**Usage:**

.. code-block:: python

   from infrastructure.logging.async_logger import AsyncLogger

   # Create async logger
   async_logger = AsyncLogger(
       batch_size=1000,
       flush_interval=1.0,  # seconds
       max_queue_size=10000
   )

   await async_logger.start()

   # Async logging (non-blocking)
   await async_logger.ainfo("Async simulation step started")
   await async_logger.adebug("Agent update", agent_id=789)

   # Context manager for automatic flushing
   async with async_logger.ensure_flushed():
       for i in range(10000):
           await async_logger.adebug(f"Processing item {i}")
   # Automatically flushed when exiting context

   await async_logger.stop()

**Performance Optimization:**

.. code-block:: python

   # Configure for high-throughput scenarios
   high_perf_logger = AsyncLogger(
       batch_size=5000,  # Larger batches
       flush_interval=0.1,  # More frequent flushes
       compression=True,  # Compress log data
       async_write=True  # Asynchronous disk writes
   )

   # Monitor performance
   stats = await high_perf_logger.get_performance_stats()
   print(f"Messages/second: {stats.messages_per_second}")
   print(f"Queue utilization: {stats.queue_utilization:.2%}")
   print(f"Average latency: {stats.avg_latency_ms:.2f}ms")

Performance Monitoring
----------------------

Real-time performance tracking and system resource monitoring.

Performance Monitor
~~~~~~~~~~~~~~~~~~~

.. automodule:: infrastructure.performance.monitor
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensive performance monitoring system.

**Monitoring Capabilities:**

* **CPU Usage**: Per-process and system-wide CPU utilization
* **Memory Usage**: RAM, swap, and memory allocations
* **I/O Performance**: Disk read/write rates and latency
* **Network**: Bandwidth usage and connection counts
* **Application Metrics**: Custom performance indicators

**Basic Usage:**

.. code-block:: python

   from infrastructure.performance.monitor import PerformanceMonitor

   monitor = PerformanceMonitor(
       sampling_interval=0.1,  # Sample every 100ms
       history_length=1000,    # Keep 1000 samples
       alert_thresholds={
           'cpu_percent': 80,
           'memory_percent': 90,
           'disk_usage_percent': 95
       }
   )

   # Start monitoring
   await monitor.start()

   # Monitor specific functions
   @monitor.profile
   async def expensive_simulation_step():
       # Expensive computation here
       pass

   await expensive_simulation_step()

   # Get performance report
   report = monitor.get_performance_report()
   print(f"Average CPU: {report.avg_cpu_percent:.1f}%")
   print(f"Peak memory: {report.peak_memory_mb:.1f}MB")
   print(f"Function calls: {report.function_call_count}")

**Real-time Monitoring:**

.. code-block:: python

   # Real-time dashboard
   dashboard = monitor.create_dashboard(
       update_interval=1.0,
       show_plots=True,
       save_screenshots=True
   )

   # Start dashboard server
   await dashboard.start_server(port=8080)

   # Alert system
   @monitor.on_alert
   async def handle_performance_alert(alert):
       if alert.metric == 'memory_percent' and alert.value > 90:
           logger.warning(f"High memory usage: {alert.value:.1f}%")
           # Trigger garbage collection or cleanup
           await cleanup_resources()

**Profiling Integration:**

.. code-block:: python

   # Profile specific simulation components
   cell_profiler = monitor.create_profiler("cellular_agents")
   coordination_profiler = monitor.create_profiler("coordination_system")

   with cell_profiler:
       for agent in agents:
           await agent.step()

   with coordination_profiler:
       await coordinator.resolve_conflicts()

   # Generate profiling report
   profiling_report = monitor.generate_profiling_report([
       cell_profiler, coordination_profiler
   ])

**Custom Metrics:**

.. code-block:: python

   # Define custom metrics
   monitor.define_metric(
       name="emergence_score",
       description="Quantification of emergent behavior",
       unit="score",
       aggregation="average"
   )

   monitor.define_metric(
       name="agent_interactions_per_second",
       description="Rate of agent-agent interactions",
       unit="interactions/sec",
       aggregation="sum"
   )

   # Record custom metrics
   monitor.record_metric("emergence_score", 0.75)
   monitor.record_metric("agent_interactions_per_second", 1250)

**Performance Optimization Recommendations:**

.. code-block:: python

   # Get optimization recommendations
   optimizer = monitor.get_optimizer()
   recommendations = optimizer.analyze_performance(
       simulation_data,
       focus_areas=['cpu', 'memory', 'io']
   )

   for recommendation in recommendations:
       print(f"Issue: {recommendation.issue}")
       print(f"Recommendation: {recommendation.solution}")
       print(f"Expected improvement: {recommendation.expected_improvement}")

Testing Infrastructure
----------------------

Comprehensive testing framework for morphogenesis simulations.

Test Fixtures
~~~~~~~~~~~~~

.. automodule:: infrastructure.testing.fixtures
   :members:
   :undoc-members:
   :show-inheritance:

Reusable test fixtures and mock objects for testing.

**Available Fixtures:**

* **Mock Agents**: Configurable cellular agents for testing
* **Test Environments**: Pre-configured simulation environments
* **Data Generators**: Synthetic data for testing algorithms
* **Performance Benchmarks**: Standard benchmarks for comparison

**Basic Usage:**

.. code-block:: python

   import pytest
   from infrastructure.testing.fixtures import create_test_environment

   @pytest.fixture
   async def simulation_env():
       """Create a standard test environment."""
       env = await create_test_environment(
           population_size=100,
           grid_size=(50, 50),
           cell_types=['A', 'B'],
           random_seed=42
       )
       yield env
       await env.cleanup()

   async def test_cellular_sorting(simulation_env):
       # Run sorting simulation
       results = await simulation_env.run_sorting_experiment(
           simulation_time=1000
       )

       # Validate results
       assert results.sorting_quality > 0.8
       assert results.convergence_achieved
       assert results.execution_time < 30.0  # seconds

**Mock Objects:**

.. code-block:: python

   from infrastructure.testing.fixtures import MockCellAgent, MockCoordinator

   # Create mock agent with predictable behavior
   mock_agent = MockCellAgent(
       agent_id=1,
       behavior_pattern='circular_movement',
       movement_speed=1.0
   )

   # Configure mock responses
   mock_agent.set_response_to_neighbors(
       neighbor_count=5,
       response='move_away'
   )

   # Mock coordinator for testing
   mock_coordinator = MockCoordinator(
       grid_size=(10, 10),
       enable_collision_detection=True
   )

   # Verify interactions
   await mock_agent.step()
   assert mock_coordinator.get_call_count('resolve_collision') == 0

**Data Generators:**

.. code-block:: python

   from infrastructure.testing.fixtures import DataGenerator

   generator = DataGenerator(seed=42)

   # Generate synthetic agent positions
   positions = generator.generate_clustered_positions(
       n_agents=200,
       n_clusters=5,
       cluster_spread=10.0
   )

   # Generate synthetic time series data
   time_series = generator.generate_emergence_time_series(
       length=1000,
       emergence_point=500,
       noise_level=0.1
   )

   # Generate synthetic interaction networks
   network = generator.generate_interaction_network(
       n_agents=100,
       connection_probability=0.1,
       network_type='small_world'
   )

Test Case Base
~~~~~~~~~~~~~~

.. automodule:: infrastructure.testing.test_case
   :members:
   :undoc-members:
   :show-inheritance:

Base classes and utilities for writing morphogenesis tests.

**Test Categories:**

* **Unit Tests**: Test individual components in isolation
* **Integration Tests**: Test component interactions
* **Performance Tests**: Validate performance requirements
* **Property Tests**: Property-based testing with Hypothesis
* **Regression Tests**: Ensure consistency across versions

**Base Test Class:**

.. code-block:: python

   from infrastructure.testing.test_case import MorphogenesisTestCase

   class TestCellularSorting(MorphogenesisTestCase):

       async def setUp(self):
           self.env = await self.create_test_environment()
           self.metrics_collector = self.create_metrics_collector()

       async def test_sorting_convergence(self):
           """Test that cells sort within expected time."""
           results = await self.run_sorting_simulation(
               population_size=200,
               max_time=2000
           )

           # Use assertion helpers
           self.assertConverged(results, tolerance=0.05)
           self.assertWithinTimeLimit(results, max_time=1500)
           self.assertEmergenceDetected(results, min_strength=0.7)

       async def test_performance_requirements(self):
           """Test that simulation meets performance requirements."""
           with self.performance_monitor():
               results = await self.run_sorting_simulation(
                   population_size=1000
               )

           self.assertPerformanceWithinLimits(
               max_memory_mb=500,
               max_cpu_percent=80,
               max_execution_time=60
           )

**Property-Based Testing:**

.. code-block:: python

   from hypothesis import strategies as st
   from infrastructure.testing.test_case import PropertyBasedTest

   class TestSortingProperties(PropertyBasedTest):

       @given(
           population_size=st.integers(min_value=10, max_value=1000),
           sorting_strength=st.floats(min_value=0.1, max_value=1.0),
           grid_density=st.floats(min_value=0.1, max_value=0.9)
       )
       async def test_sorting_invariants(self, population_size, sorting_strength, grid_density):
           """Test that sorting maintains certain invariants."""
           results = await self.run_sorting_simulation(
               population_size=population_size,
               sorting_strength=sorting_strength,
               grid_density=grid_density
           )

           # Invariant: Total agent count remains constant
           self.assertEqual(results.final_agent_count, population_size)

           # Invariant: Higher sorting strength leads to better sorting
           if sorting_strength > 0.8:
               self.assertGreater(results.sorting_quality, 0.7)

**Performance Testing:**

.. code-block:: python

   from infrastructure.testing.test_case import PerformanceTestCase

   class TestPerformance(PerformanceTestCase):

       def test_scaling_performance(self):
           """Test performance scaling with population size."""
           population_sizes = [100, 500, 1000, 2000, 5000]
           results = []

           for size in population_sizes:
               with self.measure_performance() as perf:
                   result = await self.run_simulation(population_size=size)

               results.append({
                   'population_size': size,
                   'execution_time': perf.execution_time,
                   'memory_usage': perf.peak_memory_mb,
                   'sorting_quality': result.sorting_quality
               })

           # Analyze scaling characteristics
           self.assertLinearScaling(results, 'execution_time', tolerance=0.2)
           self.assertSublinearScaling(results, 'memory_usage')

**Regression Testing:**

.. code-block:: python

   from infrastructure.testing.test_case import RegressionTestCase

   class TestRegression(RegressionTestCase):

       def test_algorithm_consistency(self):
           """Ensure algorithms produce consistent results across versions."""
           reference_config = self.load_reference_config('v1.0.0')

           current_results = await self.run_simulation(reference_config)
           reference_results = self.load_reference_results('v1.0.0')

           self.assertResultsEquivalent(
               current_results,
               reference_results,
               tolerance=0.01
           )

System Utilities
----------------

Platform-specific optimizations and system integration utilities.

**Resource Management:**

.. code-block:: python

   from infrastructure.system.resource_manager import ResourceManager

   resource_manager = ResourceManager()

   # Memory management
   await resource_manager.optimize_memory_usage()
   current_usage = resource_manager.get_memory_usage()

   # CPU affinity for performance
   resource_manager.set_cpu_affinity([0, 1, 2, 3])  # Use specific cores

   # Disk I/O optimization
   resource_manager.optimize_disk_io(
       buffer_size='64KB',
       sync_frequency=10  # seconds
   )

**Platform Detection:**

.. code-block:: python

   from infrastructure.system.platform_detection import PlatformInfo

   platform = PlatformInfo()

   if platform.is_linux():
       # Linux-specific optimizations
       platform.enable_transparent_huge_pages()
   elif platform.is_windows():
       # Windows-specific optimizations
       platform.set_process_priority('high')
   elif platform.is_macos():
       # macOS-specific optimizations
       platform.optimize_for_metal_performance()

**Configuration Management:**

.. code-block:: python

   from infrastructure.system.config_manager import ConfigManager

   config = ConfigManager()

   # Load configuration from multiple sources
   config.load_from_file('morphogenesis.yaml')
   config.load_from_environment(prefix='MORPH_')
   config.load_from_command_line(sys.argv)

   # Access configuration with type safety
   population_size = config.get_int('population_size', default=1000)
   debug_mode = config.get_bool('debug_mode', default=False)
   output_dir = config.get_path('output_dir', default='./results')

Error Handling and Recovery
---------------------------

Robust error handling and recovery mechanisms.

**Error Recovery:**

.. code-block:: python

   from infrastructure.error_handling import ErrorRecoveryManager

   recovery_manager = ErrorRecoveryManager()

   # Configure retry strategies
   recovery_manager.add_retry_strategy(
       exception_type=ResourceExhaustedException,
       max_retries=3,
       backoff_strategy='exponential',
       recovery_action=cleanup_resources
   )

   # Checkpoint-based recovery
   @recovery_manager.with_checkpointing(interval=100)
   async def long_running_simulation():
       for step in range(10000):
           await simulation_step()
           if step % 100 == 0:
               recovery_manager.create_checkpoint(step)

**Graceful Shutdown:**

.. code-block:: python

   from infrastructure.lifecycle import LifecycleManager

   lifecycle = LifecycleManager()

   # Register cleanup handlers
   lifecycle.register_cleanup_handler(save_simulation_state)
   lifecycle.register_cleanup_handler(close_database_connections)
   lifecycle.register_cleanup_handler(shutdown_monitoring)

   # Handle shutdown signals
   lifecycle.handle_shutdown_signals(['SIGINT', 'SIGTERM'])

   # Ensure graceful shutdown
   async with lifecycle:
       await run_simulation()
   # Cleanup handlers automatically called

Integration Examples
--------------------

**Complete Infrastructure Setup:**

.. code-block:: python

   import asyncio
   from infrastructure.logging.logger import MorphogenesisLogger
   from infrastructure.performance.monitor import PerformanceMonitor
   from infrastructure.testing.fixtures import create_test_environment

   async def setup_infrastructure():
       # Configure logging
       logger = MorphogenesisLogger(
           name="morphogenesis_simulation",
           level="INFO",
           handlers=['console', 'file', 'async_buffer']
       )

       # Start performance monitoring
       monitor = PerformanceMonitor(
           sampling_interval=0.1,
           enable_alerts=True,
           dashboard_port=8080
       )
       await monitor.start()

       # Create test environment if in test mode
       if os.getenv('TESTING', False):
           test_env = await create_test_environment()
           return logger, monitor, test_env

       return logger, monitor

   async def main():
       logger, monitor = await setup_infrastructure()

       # Your simulation code here
       logger.info("Infrastructure setup complete")

       # Cleanup
       await monitor.stop()

   if __name__ == "__main__":
       asyncio.run(main())

**Production Deployment:**

.. code-block:: python

   from infrastructure.deployment import ProductionSetup

   async def production_deployment():
       setup = ProductionSetup()

       # Configure for production
       await setup.configure_production_logging()
       await setup.setup_monitoring_dashboard()
       await setup.enable_error_reporting()
       await setup.optimize_system_resources()

       # Health checks
       health_checker = setup.create_health_checker([
           'database_connection',
           'memory_usage',
           'disk_space',
           'simulation_responsiveness'
       ])

       await health_checker.start()

       logger.info("Production environment ready")

Performance Considerations
--------------------------

**Optimization Guidelines:**

1. **Logging**: Use async logging for high-throughput scenarios
2. **Monitoring**: Balance monitoring frequency with performance impact
3. **Testing**: Run performance tests in isolated environments
4. **Resource Management**: Monitor and limit resource usage
5. **Error Handling**: Implement efficient error recovery strategies

**Scaling Recommendations:**

* **Development**: Full logging and monitoring enabled
* **Testing**: Lightweight monitoring with comprehensive test coverage
* **Production**: Optimized logging with essential monitoring only

**Common Pitfalls:**

* Excessive logging in tight loops
* Not cleaning up monitoring resources
* Blocking operations in async logging
* Insufficient error handling in distributed systems