"""Logging infrastructure for morphogenesis simulation.

Provides comprehensive logging capabilities including:
- Structured logging with JSON output
- Performance logging for profiling
- Async logging for high-performance scenarios
- Multiple output handlers (console, file, remote)
- Log filtering and formatting
"""

from .logger import (
    setup_logging, get_logger, LoggingConfig,
    create_file_handler, create_console_handler, create_json_handler
)
from .performance_logger import PerformanceLogger, TimingContext
from .async_logger import AsyncLogger, AsyncLogHandler
from .formatters import (
    JSONFormatter, ColoredFormatter, PerformanceFormatter
)

__all__ = [
    'setup_logging',
    'get_logger',
    'LoggingConfig',
    'create_file_handler',
    'create_console_handler',
    'create_json_handler',
    'PerformanceLogger',
    'TimingContext',
    'AsyncLogger',
    'AsyncLogHandler',
    'JSONFormatter',
    'ColoredFormatter',
    'PerformanceFormatter',
]