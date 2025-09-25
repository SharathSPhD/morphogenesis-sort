"""Custom logging formatters for morphogenesis simulation.

Provides specialized formatters for different logging needs including
JSON output, colored console output, and performance-specific formatting.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_extra: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add process/thread info
        if record.process:
            log_entry['process'] = record.process
        if record.thread:
            log_entry['thread'] = record.thread

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra attributes if enabled
        if self.include_extra:
            # Get custom attributes (those not in standard LogRecord)
            standard_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message'
            }

            extra_attrs = {}
            for key, value in record.__dict__.items():
                if key not in standard_attrs:
                    # Ensure value is JSON serializable
                    try:
                        json.dumps(value)
                        extra_attrs[key] = value
                    except (TypeError, ValueError):
                        extra_attrs[key] = str(value)

            if extra_attrs:
                log_entry['extra'] = extra_attrs

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',        # Bold
        'DIM': '\033[2m',         # Dim
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        # Check if we're running in a terminal that supports colors
        return (
            hasattr(sys.stdout, 'isatty') and
            sys.stdout.isatty() and
            'TERM' in os.environ and
            os.environ['TERM'] != 'dumb'
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format record with colors."""
        if not self.use_colors:
            return super().format(record)

        # Get base formatted message
        message = super().format(record)

        # Apply colors based on log level
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])

        # Color the level name
        colored_level = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        message = message.replace(record.levelname, colored_level, 1)

        # Highlight logger name
        if hasattr(record, 'name') and record.name:
            logger_parts = record.name.split('.')
            if len(logger_parts) > 2:
                # Dim the package parts, highlight the module
                package_part = '.'.join(logger_parts[:-1])
                module_part = logger_parts[-1]
                colored_name = (
                    f"{self.COLORS['DIM']}{package_part}.{self.COLORS['RESET']}"
                    f"{self.COLORS['BOLD']}{module_part}{self.COLORS['RESET']}"
                )
            else:
                colored_name = f"{self.COLORS['BOLD']}{record.name}{self.COLORS['RESET']}"

            message = message.replace(record.name, colored_name, 1)

        # Highlight error messages
        if record.levelno >= logging.ERROR:
            message = f"{self.COLORS['BOLD']}{message}{self.COLORS['RESET']}"

        return message


class PerformanceFormatter(logging.Formatter):
    """Specialized formatter for performance logging."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = (
                "%(asctime)s [PERF] %(name)s:%(funcName)s:%(lineno)d - "
                "%(message)s %(performance_data)s"
            )
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format performance record with metrics."""
        # Collect performance-related attributes
        perf_attrs = {}
        performance_keys = [
            'duration', 'execution_time', 'memory_usage', 'cpu_usage',
            'throughput', 'latency', 'operations_count', 'memory_delta',
            'peak_memory', 'avg_cpu', 'cache_hits', 'cache_misses'
        ]

        for key in performance_keys:
            if hasattr(record, key):
                value = getattr(record, key)
                if isinstance(value, float):
                    # Format float values nicely
                    if value < 0.001:
                        perf_attrs[key] = f"{value*1000:.3f}µs"
                    elif value < 1:
                        perf_attrs[key] = f"{value*1000:.2f}ms"
                    else:
                        perf_attrs[key] = f"{value:.3f}s"
                elif isinstance(value, int) and key.endswith('_usage'):
                    # Memory usage in bytes
                    if value < 1024:
                        perf_attrs[key] = f"{value}B"
                    elif value < 1024**2:
                        perf_attrs[key] = f"{value/1024:.1f}KB"
                    elif value < 1024**3:
                        perf_attrs[key] = f"{value/1024**2:.1f}MB"
                    else:
                        perf_attrs[key] = f"{value/1024**3:.1f}GB"
                else:
                    perf_attrs[key] = str(value)

        # Format performance data as key=value pairs
        if perf_attrs:
            perf_data = ' '.join(f"{k}={v}" for k, v in perf_attrs.items())
            record.performance_data = f"[{perf_data}]"
        else:
            record.performance_data = ""

        return super().format(record)


class CompactFormatter(logging.Formatter):
    """Compact formatter for minimal output."""

    def __init__(self):
        super().__init__("%(levelname)s:%(name)s:%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Format record in compact form."""
        # Shorten logger names
        name_parts = record.name.split('.')
        if len(name_parts) > 2:
            # Keep first and last parts
            short_name = f"{name_parts[0]}...{name_parts[-1]}"
        else:
            short_name = record.name

        original_name = record.name
        record.name = short_name
        result = super().format(record)
        record.name = original_name

        return result


class SimulationFormatter(logging.Formatter):
    """Specialized formatter for simulation logging."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_simulation_context: bool = True
    ):
        if fmt is None:
            fmt = (
                "%(asctime)s [%(levelname)s] %(simulation_context)s "
                "%(name)s: %(message)s"
            )
        super().__init__(fmt, datefmt)
        self.include_simulation_context = include_simulation_context

    def format(self, record: logging.LogRecord) -> str:
        """Format record with simulation context."""
        if self.include_simulation_context:
            context_parts = []

            # Add timestep if available
            if hasattr(record, 'timestep'):
                context_parts.append(f"t={record.timestep}")

            # Add cell ID if available
            if hasattr(record, 'cell_id'):
                context_parts.append(f"cell={record.cell_id}")

            # Add experiment name if available
            if hasattr(record, 'experiment'):
                context_parts.append(f"exp={record.experiment}")

            # Add phase if available
            if hasattr(record, 'phase'):
                context_parts.append(f"phase={record.phase}")

            if context_parts:
                record.simulation_context = f"[{' '.join(context_parts)}]"
            else:
                record.simulation_context = ""
        else:
            record.simulation_context = ""

        return super().format(record)


class MultilineFormatter(logging.Formatter):
    """Formatter that handles multiline messages properly."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        indent: str = "    "
    ):
        super().__init__(fmt, datefmt)
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        """Format record with proper multiline indentation."""
        # Get formatted message
        formatted = super().format(record)

        # Split into lines
        lines = formatted.split('\n')

        if len(lines) <= 1:
            return formatted

        # Indent continuation lines
        result = [lines[0]]  # First line unchanged
        for line in lines[1:]:
            if line.strip():  # Don't indent empty lines
                result.append(self.indent + line)
            else:
                result.append(line)

        return '\n'.join(result)


# Import os after class definitions to avoid issues
import os