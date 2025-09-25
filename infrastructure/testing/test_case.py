"""Base test case classes for morphogenesis simulation testing.

Provides specialized test case classes with simulation-specific functionality,
async support, and common testing utilities.
"""

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass
import logging
import time
import sys
from contextlib import asynccontextmanager

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.data.types import (
    CellID, Position, CellType, CellState, CellParameters, WorldParameters,
    create_cell_id, create_simulation_time
)
from core.data.state import CellData, SimulationState
from core.data.actions import CellAction, create_wait_action
from core.agents.cell_agent import AsyncCellAgent, CellBehaviorConfig

from .fixtures import TestFixtures
from .assertions import SimulationAssertions


@dataclass
class TestConfig:
    """Configuration for simulation tests."""
    # Test environment
    temp_dir_cleanup: bool = True
    log_level: str = "WARNING"
    capture_logs: bool = True

    # Simulation parameters
    default_cell_count: int = 10
    default_world_size: tuple = (50, 50)
    default_timesteps: int = 100
    default_seed: int = 42

    # Performance thresholds
    max_test_duration: float = 30.0
    memory_limit_mb: float = 100.0

    # Async settings
    async_timeout: float = 10.0
    event_loop_policy: Optional[str] = None


class SimulationTestCase(unittest.TestCase):
    """Base test case for synchronous simulation tests."""

    test_config = TestConfig()

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        super().setUpClass()

        # Setup logging
        cls._setup_test_logging()

        # Create shared fixtures
        cls.fixtures = TestFixtures()

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        # Clean up fixtures
        if hasattr(cls, 'fixtures'):
            cls.fixtures.cleanup()

        super().tearDownClass()

    @classmethod
    def _setup_test_logging(cls):
        """Setup logging for tests."""
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, cls.test_config.log_level))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add test handler
        if cls.test_config.capture_logs:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)

    def setUp(self):
        """Set up individual test."""
        super().setUp()

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="morphogenesis_test_"))
        self.addCleanup(self._cleanup_temp_dir)

        # Initialize assertions helper
        self.sim_assert = SimulationAssertions()

        # Track test start time for performance monitoring
        self._test_start_time = time.time()

    def tearDown(self):
        """Clean up individual test."""
        # Check test duration
        duration = time.time() - self._test_start_time
        if duration > self.test_config.max_test_duration:
            self.logger.warning(
                f"Test {self._testMethodName} took {duration:.2f}s "
                f"(limit: {self.test_config.max_test_duration}s)"
            )

        super().tearDown()

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.test_config.temp_dir_cleanup and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @property
    def logger(self) -> logging.Logger:
        """Get test logger."""
        return logging.getLogger(f"test.{self.__class__.__name__}")

    # Helper methods for creating test data
    def create_test_cell_data(
        self,
        cell_id: Optional[CellID] = None,
        position: Optional[Position] = None,
        cell_type: CellType = CellType.STANDARD,
        cell_state: CellState = CellState.ACTIVE,
        sort_value: float = 0.0,
        age: int = 0
    ) -> CellData:
        """Create test cell data with defaults."""
        if cell_id is None:
            cell_id = create_cell_id(1)
        if position is None:
            position = Position(0.0, 0.0)

        return CellData(
            cell_id=cell_id,
            position=position,
            cell_type=cell_type,
            cell_state=cell_state,
            sort_value=sort_value,
            age=age
        )

    def create_test_simulation_state(
        self,
        cell_count: Optional[int] = None,
        world_width: int = 50,
        world_height: int = 50
    ) -> SimulationState:
        """Create test simulation state."""
        if cell_count is None:
            cell_count = self.test_config.default_cell_count

        # Create cells
        cells = {}
        for i in range(cell_count):
            cell_id = create_cell_id(i)
            position = Position(
                float(i % world_width),
                float(i // world_width)
            )
            cells[cell_id] = self.create_test_cell_data(
                cell_id=cell_id,
                position=position,
                sort_value=float(i)
            )

        # Create world parameters
        world_params = WorldParameters(
            width=world_width,
            height=world_height
        )

        return SimulationState(
            timestep=create_simulation_time(0),
            cells=cells,
            world_parameters=world_params,
            global_metrics={},
            metadata=None
        )

    def create_test_behavior_config(self, **kwargs) -> CellBehaviorConfig:
        """Create test behavior configuration."""
        defaults = {
            'behavior_type': 'test',
            'decision_frequency': 1,
            'action_delay': 0.0,
            'sorting_enabled': True,
            'movement_enabled': True,
            'interaction_enabled': True,
            'error_recovery': True,
            'freeze_on_error': False
        }
        defaults.update(kwargs)
        return CellBehaviorConfig(**defaults)

    def create_test_cell_parameters(self, **kwargs) -> CellParameters:
        """Create test cell parameters."""
        defaults = {
            'max_speed': 1.0,
            'interaction_radius': 3.0,
            'swap_probability': 0.1,
            'max_age': 1000
        }
        defaults.update(kwargs)
        return CellParameters(**defaults)

    # Assertion helpers
    def assertCellDataEqual(self, cell1: CellData, cell2: CellData, msg: Optional[str] = None):
        """Assert that two CellData objects are equal."""
        self.sim_assert.assert_cell_data_equal(cell1, cell2, msg)

    def assertPositionClose(
        self,
        pos1: Position,
        pos2: Position,
        delta: float = 0.001,
        msg: Optional[str] = None
    ):
        """Assert that two positions are close."""
        self.sim_assert.assert_position_close(pos1, pos2, delta, msg)

    def assertMetricsValid(self, metrics: Dict[str, Any], msg: Optional[str] = None):
        """Assert that metrics dictionary is valid."""
        self.sim_assert.assert_metrics_valid(metrics, msg)


class AsyncSimulationTestCase(SimulationTestCase):
    """Base test case for asynchronous simulation tests."""

    def setUp(self):
        """Set up async test."""
        super().setUp()

        # Get or create event loop
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Set event loop policy if specified
        if self.test_config.event_loop_policy:
            policy_class = getattr(asyncio, self.test_config.event_loop_policy)
            asyncio.set_event_loop_policy(policy_class())

    def tearDown(self):
        """Clean up async test."""
        # Close event loop if we created it
        if hasattr(self, 'loop') and not self.loop.is_closed():
            # Cancel any remaining tasks
            pending_tasks = asyncio.all_tasks(self.loop)
            for task in pending_tasks:
                task.cancel()

            # Wait for tasks to complete
            if pending_tasks:
                self.loop.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )

        super().tearDown()

    def run_async(self, coro, timeout: Optional[float] = None):
        """Run an async coroutine in the test event loop."""
        if timeout is None:
            timeout = self.test_config.async_timeout

        try:
            return self.loop.run_until_complete(
                asyncio.wait_for(coro, timeout=timeout)
            )
        except asyncio.TimeoutError:
            self.fail(f"Async operation timed out after {timeout}s")

    @asynccontextmanager
    async def async_test_context(self):
        """Async context manager for test setup/teardown."""
        # Setup
        start_time = time.time()

        try:
            yield
        finally:
            # Teardown
            duration = time.time() - start_time
            if duration > self.test_config.max_test_duration:
                self.logger.warning(
                    f"Async test took {duration:.2f}s "
                    f"(limit: {self.test_config.max_test_duration}s)"
                )

    async def create_test_cell_agent(
        self,
        cell_id: Optional[CellID] = None,
        **kwargs
    ) -> AsyncCellAgent:
        """Create a test cell agent."""
        if cell_id is None:
            cell_id = create_cell_id(1)

        cell_data = self.create_test_cell_data(cell_id=cell_id)
        behavior_config = self.create_test_behavior_config(**kwargs)
        parameters = self.create_test_cell_parameters()

        # Create a simple test agent implementation
        class TestCellAgent(AsyncCellAgent):
            async def _decide_action(self) -> CellAction:
                return create_wait_action(self.cell_id, self.last_update_timestep)

        return TestCellAgent(cell_id, cell_data, behavior_config, parameters)

    async def run_cell_agent_briefly(
        self,
        agent: AsyncCellAgent,
        duration: float = 0.1
    ) -> List[CellAction]:
        """Run a cell agent for a brief time and collect actions."""
        actions = []

        # Create action collection task
        async def collect_actions():
            async for action in agent._lifecycle_generator():
                actions.append(action)
                if len(actions) >= 10:  # Limit actions collected
                    break

        # Run with timeout
        try:
            await asyncio.wait_for(collect_actions(), timeout=duration)
        except asyncio.TimeoutError:
            pass  # Expected for brief runs

        return actions

    # Async assertion helpers
    async def assert_eventually(
        self,
        condition: Callable[[], Union[bool, Any]],
        timeout: float = 5.0,
        check_interval: float = 0.1,
        msg: Optional[str] = None
    ):
        """Assert that a condition eventually becomes true."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = condition()
                if result:
                    return result
            except Exception:
                pass  # Ignore exceptions during checking

            await asyncio.sleep(check_interval)

        # Final check with proper error reporting
        result = condition()
        if not result:
            error_msg = msg or f"Condition not met within {timeout}s"
            self.fail(error_msg)

    async def assert_async_raises(
        self,
        exception_type: type,
        coro,
        msg: Optional[str] = None
    ):
        """Assert that an async operation raises a specific exception."""
        with self.assertRaises(exception_type, msg=msg):
            await coro


class PerformanceTestCase(AsyncSimulationTestCase):
    """Test case with performance monitoring and assertions."""

    def setUp(self):
        """Set up performance test."""
        super().setUp()
        self._performance_start = time.time()
        self._memory_start = self._get_memory_usage()

    def tearDown(self):
        """Clean up performance test."""
        duration = time.time() - self._performance_start
        memory_end = self._get_memory_usage()
        memory_delta = memory_end - self._memory_start

        # Log performance metrics
        self.logger.info(
            f"Performance: duration={duration:.3f}s, "
            f"memory_delta={memory_delta:.1f}MB"
        )

        # Check performance thresholds
        if duration > self.test_config.max_test_duration:
            self.fail(
                f"Test exceeded duration limit: {duration:.2f}s > "
                f"{self.test_config.max_test_duration}s"
            )

        if memory_delta > self.test_config.memory_limit_mb:
            self.fail(
                f"Test exceeded memory limit: {memory_delta:.1f}MB > "
                f"{self.test_config.memory_limit_mb}MB"
            )

        super().tearDown()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def assert_performance_within_limits(
        self,
        max_duration: Optional[float] = None,
        max_memory_mb: Optional[float] = None
    ):
        """Assert that performance is within specified limits."""
        current_duration = time.time() - self._performance_start
        current_memory = self._get_memory_usage() - self._memory_start

        if max_duration and current_duration > max_duration:
            self.fail(
                f"Operation too slow: {current_duration:.2f}s > {max_duration}s"
            )

        if max_memory_mb and current_memory > max_memory_mb:
            self.fail(
                f"Memory usage too high: {current_memory:.1f}MB > {max_memory_mb}MB"
            )


# Test utilities and decorators
def skip_if_no_module(module_name: str, reason: Optional[str] = None):
    """Skip test if module is not available."""
    try:
        __import__(module_name)
        return lambda func: func
    except ImportError:
        skip_reason = reason or f"Module '{module_name}' not available"
        return unittest.skip(skip_reason)


def timeout(seconds: float):
    """Decorator to add timeout to async test methods."""
    def decorator(func):
        if not asyncio.iscoroutinefunction(func):
            raise ValueError("@timeout can only be used on async functions")

        async def wrapper(self, *args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(self, *args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                self.fail(f"Test timed out after {seconds}s")

        return wrapper
    return decorator


def repeat(times: int):
    """Decorator to repeat a test multiple times."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for i in range(times):
                try:
                    if asyncio.iscoroutinefunction(func):
                        self.run_async(func(self, *args, **kwargs))
                    else:
                        func(self, *args, **kwargs)
                except Exception as e:
                    self.fail(f"Test failed on iteration {i+1}/{times}: {e}")

        return wrapper
    return decorator


# Example test class
class ExampleSimulationTest(AsyncSimulationTestCase):
    """Example test class showing usage patterns."""

    async def test_basic_cell_creation(self):
        """Test basic cell data creation."""
        cell_data = self.create_test_cell_data()
        self.assertEqual(cell_data.cell_id, create_cell_id(1))
        self.assertEqual(cell_data.position, Position(0.0, 0.0))

    async def test_cell_agent_lifecycle(self):
        """Test cell agent lifecycle."""
        async with self.async_test_context():
            agent = await self.create_test_cell_agent()

            # Test that agent is created properly
            self.assertEqual(agent.cell_id, create_cell_id(1))
            self.assertIsNotNone(agent.current_data)

            # Test brief execution
            actions = await self.run_cell_agent_briefly(agent, duration=0.1)
            self.assertGreater(len(actions), 0)

    @timeout(5.0)
    async def test_with_timeout(self):
        """Test with timeout decorator."""
        await asyncio.sleep(0.1)  # Should complete quickly
        self.assertTrue(True)

    @repeat(3)
    def test_repeated(self):
        """Test with repeat decorator."""
        self.assertTrue(True)


if __name__ == "__main__":
    # Run example tests
    unittest.main()