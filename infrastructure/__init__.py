"""Infrastructure components for morphogenesis simulation.

This package provides core infrastructure components including:
- Logging: Comprehensive logging system with multiple handlers and formatters
- Performance: Performance monitoring and profiling utilities
- Testing: Testing utilities and fixtures for simulation components
"""

__version__ = "1.0.0"
__author__ = "Morphogenesis Research Team"

from .logging import (
    setup_logging, get_logger, LoggingConfig,
    PerformanceLogger, AsyncLogger
)
from .performance import (
    PerformanceMonitor, ProfilerManager, MetricsCollector,
    BenchmarkRunner, ResourceTracker
)
from .testing import (
    SimulationTestCase, AsyncTestRunner, TestFixtures,
    MockCellAgent, TestDataGenerator
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'LoggingConfig',
    'PerformanceLogger',
    'AsyncLogger',

    # Performance
    'PerformanceMonitor',
    'ProfilerManager',
    'MetricsCollector',
    'BenchmarkRunner',
    'ResourceTracker',

    # Testing
    'SimulationTestCase',
    'AsyncTestRunner',
    'TestFixtures',
    'MockCellAgent',
    'TestDataGenerator',
]