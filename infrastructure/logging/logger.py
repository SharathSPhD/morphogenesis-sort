"""Core logging system for morphogenesis simulation.

This module provides a comprehensive logging system with structured logging,
multiple output formats, and performance optimization for simulation environments.
"""

import logging
import logging.config
import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue

from .formatters import JSONFormatter, ColoredFormatter, PerformanceFormatter


@dataclass
class LoggingConfig:
    """Configuration for logging system."""

    # Basic configuration
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # File logging
    file_logging_enabled: bool = True
    log_file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Console logging
    console_logging_enabled: bool = True
    console_colored: bool = True
    console_level: str = "INFO"

    # JSON logging
    json_logging_enabled: bool = False
    json_file_path: Optional[str] = None

    # Performance logging
    performance_logging_enabled: bool = True
    performance_file_path: Optional[str] = None

    # Remote logging (future extension)
    remote_logging_enabled: bool = False
    remote_endpoint: Optional[str] = None

    # Async logging
    async_logging_enabled: bool = False
    log_queue_size: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class ThreadSafeLogger:
    """Thread-safe logger implementation."""

    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self._logger = logging.getLogger(name)
        self._lock = threading.RLock()

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Thread-safe logging method."""
        with self._lock:
            self._logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.log(logging.CRITICAL, message, *args, **kwargs)


def setup_logging(
    config: Optional[LoggingConfig] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> None:
    """Setup comprehensive logging system.

    Args:
        config: Logging configuration
        log_dir: Directory for log files
    """
    if config is None:
        config = LoggingConfig()

    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Set default paths if not provided
    if config.log_file_path is None:
        config.log_file_path = str(log_dir / "morphogenesis.log")
    if config.json_file_path is None:
        config.json_file_path = str(log_dir / "morphogenesis.json")
    if config.performance_file_path is None:
        config.performance_file_path = str(log_dir / "performance.log")

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root level
    root_logger.setLevel(getattr(logging, config.level.upper()))

    handlers = []

    # Console handler
    if config.console_logging_enabled:
        console_handler = create_console_handler(config)
        handlers.append(console_handler)

    # File handler
    if config.file_logging_enabled:
        file_handler = create_file_handler(config)
        handlers.append(file_handler)

    # JSON handler
    if config.json_logging_enabled:
        json_handler = create_json_handler(config)
        handlers.append(json_handler)

    # Performance handler
    if config.performance_logging_enabled:
        performance_handler = create_performance_handler(config)
        handlers.append(performance_handler)

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized with config: {config.to_dict()}")


def create_console_handler(config: LoggingConfig) -> logging.StreamHandler:
    """Create console logging handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, config.console_level.upper()))

    if config.console_colored:
        formatter = ColoredFormatter(
            fmt=config.format,
            datefmt=config.date_format
        )
    else:
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt=config.date_format
        )

    handler.setFormatter(formatter)
    return handler


def create_file_handler(config: LoggingConfig) -> logging.handlers.RotatingFileHandler:
    """Create file logging handler."""
    import logging.handlers

    handler = logging.handlers.RotatingFileHandler(
        config.log_file_path,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    handler.setLevel(getattr(logging, config.level.upper()))

    formatter = logging.Formatter(
        fmt=config.format,
        datefmt=config.date_format
    )
    handler.setFormatter(formatter)
    return handler


def create_json_handler(config: LoggingConfig) -> logging.FileHandler:
    """Create JSON logging handler."""
    handler = logging.FileHandler(config.json_file_path, encoding='utf-8')
    handler.setLevel(getattr(logging, config.level.upper()))

    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    return handler


def create_performance_handler(config: LoggingConfig) -> logging.FileHandler:
    """Create performance logging handler."""
    handler = logging.FileHandler(config.performance_file_path, encoding='utf-8')
    handler.setLevel(logging.DEBUG)  # Capture all performance data

    formatter = PerformanceFormatter()
    handler.setFormatter(formatter)

    # Add filter to only capture performance-related logs
    handler.addFilter(PerformanceLogFilter())

    return handler


class PerformanceLogFilter(logging.Filter):
    """Filter to capture only performance-related log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter performance-related records."""
        # Check if record has performance-related attributes
        performance_indicators = [
            'duration', 'execution_time', 'memory_usage', 'cpu_usage',
            'throughput', 'latency', 'benchmark', 'profile'
        ]

        # Check record attributes
        for indicator in performance_indicators:
            if hasattr(record, indicator):
                return True

        # Check message content
        message = record.getMessage().lower()
        for indicator in performance_indicators:
            if indicator in message:
                return True

        # Check logger name
        if 'performance' in record.name.lower():
            return True

        return False


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Get logger instance with optional custom configuration.

    Args:
        name: Logger name
        config: Optional custom configuration

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If custom config provided, create custom handler
    if config and config != LoggingConfig():
        # Create custom handler for this logger
        if config.file_logging_enabled and config.log_file_path:
            file_handler = create_file_handler(config)
            logger.addHandler(file_handler)

        if config.console_logging_enabled:
            console_handler = create_console_handler(config)
            logger.addHandler(console_handler)

        logger.setLevel(getattr(logging, config.level.upper()))
        logger.propagate = False  # Don't propagate to root logger

    return logger


class LoggingContext:
    """Context manager for enhanced logging with additional context."""

    def __init__(
        self,
        logger: logging.Logger,
        context: Dict[str, Any],
        level: int = logging.INFO
    ):
        self.logger = logger
        self.context = context
        self.level = level
        self.old_context = {}

    def __enter__(self):
        # Store old context and add new context
        for key, value in self.context.items():
            if hasattr(self.logger, key):
                self.old_context[key] = getattr(self.logger, key)
            setattr(self.logger, key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        for key in self.context.keys():
            if key in self.old_context:
                setattr(self.logger, key, self.old_context[key])
            else:
                delattr(self.logger, key)


def create_simulation_logger(
    simulation_name: str,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """Create logger specifically for simulation runs.

    Args:
        simulation_name: Name of the simulation
        log_dir: Directory for log files

    Returns:
        Configured logger for the simulation
    """
    if log_dir is None:
        log_dir = Path("logs") / "simulations"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create custom config for this simulation
    config = LoggingConfig(
        log_file_path=str(log_dir / f"{simulation_name}.log"),
        json_file_path=str(log_dir / f"{simulation_name}.json"),
        performance_file_path=str(log_dir / f"{simulation_name}_performance.log")
    )

    logger = get_logger(f"simulation.{simulation_name}", config)
    logger.info(f"Simulation logger created for: {simulation_name}")

    return logger


def configure_library_loggers(level: str = "WARNING") -> None:
    """Configure third-party library loggers to reduce noise.

    Args:
        level: Logging level for library loggers
    """
    # Common noisy libraries
    noisy_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'numba',
        'asyncio'
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


class MemoryHandler(logging.handlers.MemoryHandler):
    """Enhanced memory handler for buffered logging."""

    def __init__(self, capacity: int = 1000, flush_level: int = logging.ERROR):
        super().__init__(capacity, flush_level)
        self._records_by_level = {
            logging.DEBUG: [],
            logging.INFO: [],
            logging.WARNING: [],
            logging.ERROR: [],
            logging.CRITICAL: []
        }

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record with level tracking."""
        super().emit(record)

        # Track records by level
        level = record.levelno
        if level in self._records_by_level:
            self._records_by_level[level].append(record)

            # Limit memory usage
            if len(self._records_by_level[level]) > 100:
                self._records_by_level[level] = self._records_by_level[level][-100:]

    def get_records_by_level(self, level: int) -> List[logging.LogRecord]:
        """Get records by logging level."""
        return self._records_by_level.get(level, []).copy()

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts by level."""
        return {
            logging.getLevelName(level): len(records)
            for level, records in self._records_by_level.items()
            if records
        }


def setup_development_logging() -> None:
    """Setup logging configuration optimized for development."""
    config = LoggingConfig(
        level="DEBUG",
        console_logging_enabled=True,
        console_colored=True,
        console_level="DEBUG",
        file_logging_enabled=True,
        json_logging_enabled=False,
        performance_logging_enabled=True
    )

    setup_logging(config)
    configure_library_loggers("INFO")


def setup_production_logging(log_dir: Path) -> None:
    """Setup logging configuration optimized for production."""
    config = LoggingConfig(
        level="INFO",
        console_logging_enabled=True,
        console_colored=False,
        console_level="WARNING",
        file_logging_enabled=True,
        json_logging_enabled=True,
        performance_logging_enabled=True,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )

    setup_logging(config, log_dir)
    configure_library_loggers("ERROR")