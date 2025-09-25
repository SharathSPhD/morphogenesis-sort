"""Performance logging utilities for morphogenesis simulation.

Provides specialized logging for performance monitoring, profiling,
and benchmarking of simulation components.
"""

import logging
import time
import psutil
import threading
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
import asyncio


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # Memory metrics
    memory_start: int = 0
    memory_end: int = 0
    memory_peak: int = 0
    memory_delta: int = 0

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_time_user: float = 0.0
    cpu_time_system: float = 0.0

    # Operation metrics
    operations_count: int = 0
    throughput: float = 0.0  # operations per second

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'duration': self.duration,
            'memory_delta': self.memory_delta,
            'memory_peak': self.memory_peak,
            'cpu_percent': self.cpu_percent,
            'operations_count': self.operations_count,
            'throughput': self.throughput
        }
        result.update(self.custom_metrics)
        return result


class PerformanceLogger:
    """High-performance logger for performance monitoring."""

    def __init__(
        self,
        name: str = "performance",
        log_file: Optional[Path] = None,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(f"performance.{name}")
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking

        # Performance tracking state
        self._active_timers: Dict[str, PerformanceMetrics] = {}
        self._timer_lock = threading.RLock()

        # Process handle for system metrics
        self._process = psutil.Process() if (enable_memory_tracking or enable_cpu_tracking) else None

        # Setup file handler if specified
        if log_file:
            self._setup_file_handler(log_file)

    def _setup_file_handler(self, log_file: Path) -> None:
        """Setup dedicated performance log file."""
        import logging.handlers
        from .formatters import PerformanceFormatter

        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(PerformanceFormatter())

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't propagate to root logger

    def start_timer(self, operation_name: str) -> str:
        """Start a performance timer.

        Args:
            operation_name: Name of the operation being timed

        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{operation_name}_{int(time.time() * 1000000)}"

        metrics = PerformanceMetrics()
        metrics.start_time = time.perf_counter()

        if self.enable_memory_tracking and self._process:
            try:
                memory_info = self._process.memory_info()
                metrics.memory_start = memory_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if self.enable_cpu_tracking and self._process:
            try:
                cpu_times = self._process.cpu_times()
                metrics.cpu_time_user = cpu_times.user
                metrics.cpu_time_system = cpu_times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        with self._timer_lock:
            self._active_timers[timer_id] = metrics

        return timer_id

    def stop_timer(
        self,
        timer_id: str,
        operations_count: int = 1,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """Stop a performance timer and log results.

        Args:
            timer_id: Timer ID returned by start_timer
            operations_count: Number of operations performed
            custom_metrics: Additional custom metrics

        Returns:
            Performance metrics
        """
        end_time = time.perf_counter()

        with self._timer_lock:
            if timer_id not in self._active_timers:
                raise ValueError(f"Timer {timer_id} not found")

            metrics = self._active_timers.pop(timer_id)

        # Calculate timing
        metrics.end_time = end_time
        metrics.duration = end_time - metrics.start_time
        metrics.operations_count = operations_count

        if metrics.duration > 0:
            metrics.throughput = operations_count / metrics.duration

        # Calculate memory metrics
        if self.enable_memory_tracking and self._process:
            try:
                memory_info = self._process.memory_info()
                metrics.memory_end = memory_info.rss
                metrics.memory_delta = metrics.memory_end - metrics.memory_start

                # Track peak memory during operation (approximate)
                metrics.memory_peak = max(metrics.memory_start, metrics.memory_end)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Calculate CPU metrics
        if self.enable_cpu_tracking and self._process:
            try:
                cpu_times = self._process.cpu_times()
                user_delta = cpu_times.user - metrics.cpu_time_user
                system_delta = cpu_times.system - metrics.cpu_time_system
                total_cpu_time = user_delta + system_delta

                if metrics.duration > 0:
                    metrics.cpu_percent = (total_cpu_time / metrics.duration) * 100
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Add custom metrics
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)

        # Log performance data
        self._log_performance(timer_id, metrics)

        return metrics

    def _log_performance(self, operation_id: str, metrics: PerformanceMetrics) -> None:
        """Log performance metrics."""
        # Create log record with performance data
        extra_data = {
            'operation_id': operation_id,
            'duration': metrics.duration,
            'memory_delta': metrics.memory_delta,
            'memory_peak': metrics.memory_peak,
            'cpu_percent': metrics.cpu_percent,
            'operations_count': metrics.operations_count,
            'throughput': metrics.throughput
        }

        # Add custom metrics
        extra_data.update(metrics.custom_metrics)

        # Log with extra data
        self.logger.info(
            f"Operation completed: {operation_id}",
            extra=extra_data
        )

    def log_instant_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an instantaneous metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Optional unit of measurement
            context: Additional context information
        """
        extra_data = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit or '',
            'timestamp': time.time()
        }

        if context:
            extra_data.update(context)

        self.logger.info(
            f"Instant metric: {metric_name} = {value}{unit or ''}",
            extra=extra_data
        )

    def log_memory_snapshot(self, label: str = "memory_snapshot") -> None:
        """Log current memory usage snapshot."""
        if not self.enable_memory_tracking or not self._process:
            return

        try:
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()

            extra_data = {
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms,
                'memory_percent': memory_percent,
                'memory_available': psutil.virtual_memory().available
            }

            self.logger.info(f"Memory snapshot: {label}", extra=extra_data)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Could not get memory info: {e}")

    def get_active_timers(self) -> List[str]:
        """Get list of currently active timer IDs."""
        with self._timer_lock:
            return list(self._active_timers.keys())

    def clear_active_timers(self) -> None:
        """Clear all active timers (useful for cleanup)."""
        with self._timer_lock:
            self._active_timers.clear()


@contextmanager
def TimingContext(
    logger: PerformanceLogger,
    operation_name: str,
    operations_count: int = 1,
    custom_metrics: Optional[Dict[str, Any]] = None
):
    """Context manager for timing operations.

    Args:
        logger: Performance logger instance
        operation_name: Name of the operation being timed
        operations_count: Number of operations performed
        custom_metrics: Additional custom metrics

    Yields:
        PerformanceMetrics object that can be updated during execution
    """
    timer_id = logger.start_timer(operation_name)
    temp_metrics = PerformanceMetrics()

    try:
        yield temp_metrics
    finally:
        # Merge any metrics set during execution
        final_custom_metrics = custom_metrics or {}
        if temp_metrics.custom_metrics:
            final_custom_metrics.update(temp_metrics.custom_metrics)

        final_operations_count = temp_metrics.operations_count or operations_count

        logger.stop_timer(timer_id, final_operations_count, final_custom_metrics)


def performance_monitor(
    logger: Optional[PerformanceLogger] = None,
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False
):
    """Decorator for monitoring function performance.

    Args:
        logger: Performance logger instance
        operation_name: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func: Callable) -> Callable:
        perf_logger = logger or PerformanceLogger()
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_id = perf_logger.start_timer(op_name)

            custom_metrics = {}
            if log_args:
                custom_metrics['args_count'] = len(args)
                custom_metrics['kwargs_count'] = len(kwargs)

            try:
                result = func(*args, **kwargs)

                if log_result and result is not None:
                    if hasattr(result, '__len__'):
                        custom_metrics['result_length'] = len(result)
                    custom_metrics['result_type'] = type(result).__name__

                perf_logger.stop_timer(timer_id, 1, custom_metrics)
                return result

            except Exception as e:
                custom_metrics['exception'] = str(e)
                custom_metrics['exception_type'] = type(e).__name__
                perf_logger.stop_timer(timer_id, 1, custom_metrics)
                raise

        return wrapper
    return decorator


async def async_performance_monitor(
    logger: Optional[PerformanceLogger] = None,
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False
):
    """Decorator for monitoring async function performance."""
    def decorator(func: Callable) -> Callable:
        perf_logger = logger or PerformanceLogger()
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timer_id = perf_logger.start_timer(op_name)

            custom_metrics = {}
            if log_args:
                custom_metrics['args_count'] = len(args)
                custom_metrics['kwargs_count'] = len(kwargs)

            try:
                result = await func(*args, **kwargs)

                if log_result and result is not None:
                    if hasattr(result, '__len__'):
                        custom_metrics['result_length'] = len(result)
                    custom_metrics['result_type'] = type(result).__name__

                perf_logger.stop_timer(timer_id, 1, custom_metrics)
                return result

            except Exception as e:
                custom_metrics['exception'] = str(e)
                custom_metrics['exception_type'] = type(e).__name__
                perf_logger.stop_timer(timer_id, 1, custom_metrics)
                raise

        return wrapper
    return decorator


class BenchmarkSuite:
    """Suite for running performance benchmarks."""

    def __init__(self, name: str, logger: Optional[PerformanceLogger] = None):
        self.name = name
        self.logger = logger or PerformanceLogger(f"benchmark.{name}")
        self.benchmarks: Dict[str, Callable] = {}
        self.results: Dict[str, PerformanceMetrics] = {}

    def add_benchmark(self, name: str, func: Callable) -> None:
        """Add a benchmark function."""
        self.benchmarks[name] = func

    def benchmark(self, name: str):
        """Decorator to add a benchmark function."""
        def decorator(func: Callable) -> Callable:
            self.add_benchmark(name, func)
            return func
        return decorator

    def run_benchmark(self, name: str, iterations: int = 1) -> PerformanceMetrics:
        """Run a specific benchmark."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark {name} not found")

        func = self.benchmarks[name]

        with TimingContext(self.logger, f"{self.name}.{name}", iterations) as metrics:
            for _ in range(iterations):
                func()
            metrics.operations_count = iterations

        # Get the metrics from the last logged operation
        # This is a simplified approach - in practice you'd want to capture the actual metrics
        self.results[name] = metrics
        return metrics

    def run_all_benchmarks(self, iterations: int = 1) -> Dict[str, PerformanceMetrics]:
        """Run all benchmarks in the suite."""
        results = {}
        for name in self.benchmarks:
            results[name] = self.run_benchmark(name, iterations)
        return results

    def get_results_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of benchmark results."""
        summary = {}
        for name, metrics in self.results.items():
            summary[name] = {
                'avg_duration_ms': metrics.duration * 1000 / metrics.operations_count,
                'throughput_ops_sec': metrics.throughput,
                'memory_delta_mb': metrics.memory_delta / (1024 * 1024),
                'cpu_percent': metrics.cpu_percent
            }
        return summary