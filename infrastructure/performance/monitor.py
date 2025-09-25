"""System performance monitoring for morphogenesis simulation.

Provides real-time monitoring of system resources, performance metrics,
and alerts for optimal simulation performance.
"""

import psutil
import threading
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging


class SystemMetrics(NamedTuple):
    """System performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]


@dataclass
class PerformanceAlert:
    """Performance alert configuration and state."""
    name: str
    metric: str
    threshold: float
    comparison: str  # "greater", "less", "equal"
    duration_seconds: float = 5.0
    enabled: bool = True
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    callback: Optional[Callable[[float, 'PerformanceAlert'], None]] = None


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    collection_interval: float = 1.0
    history_size: int = 3600  # 1 hour at 1s intervals
    enable_network_monitoring: bool = True
    enable_disk_monitoring: bool = True
    enable_process_monitoring: bool = True
    log_file: Optional[Path] = None
    alert_cooldown_seconds: float = 60.0


class PerformanceMonitor:
    """Real-time system performance monitor."""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Performance history
        self.metrics_history: deque = deque(maxlen=self.config.history_size)

        # Alert management
        self.alerts: Dict[str, PerformanceAlert] = {}
        self._alert_states: Dict[str, List[datetime]] = {}

        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Network baseline for delta calculations
        self._last_network_stats = None

        # Logger
        self.logger = logging.getLogger(__name__)
        if self.config.log_file:
            self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Setup file logging for performance metrics."""
        handler = logging.FileHandler(self.config.log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Check alerts
                self._check_alerts(metrics)

                # Log metrics if configured
                if self.config.log_file:
                    self._log_metrics(metrics)

                # Wait for next collection
                self._stop_event.wait(self.config.collection_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.collection_interval)

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)

        # Network metrics
        network_bytes_sent = 0
        network_bytes_recv = 0
        if self.config.enable_network_monitoring:
            network = psutil.net_io_counters()
            if self._last_network_stats:
                network_bytes_sent = network.bytes_sent - self._last_network_stats.bytes_sent
                network_bytes_recv = network.bytes_recv - self._last_network_stats.bytes_recv
            self._last_network_stats = network

        # Process count
        process_count = len(psutil.pids()) if self.config.enable_process_monitoring else 0

        # Load average (Unix-like systems)
        load_average = []
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            load_average = [0.0, 0.0, 0.0]

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count,
            load_average=load_average
        )

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        if not self.metrics_history:
            return self.collect_metrics()
        return self.metrics_history[-1]

    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[SystemMetrics]:
        """Get historical metrics.

        Args:
            duration_seconds: How far back to look (None for all history)

        Returns:
            List of SystemMetrics within the specified duration
        """
        if duration_seconds is None:
            return list(self.metrics_history)

        cutoff_time = time.time() - duration_seconds
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

    def get_average_metrics(self, duration_seconds: float = 60.0) -> Optional[SystemMetrics]:
        """Get average metrics over a time period.

        Args:
            duration_seconds: Time period to average over

        Returns:
            SystemMetrics with averaged values
        """
        history = self.get_metrics_history(duration_seconds)
        if not history:
            return None

        # Calculate averages
        count = len(history)
        avg_timestamp = sum(m.timestamp for m in history) / count
        avg_cpu = sum(m.cpu_percent for m in history) / count
        avg_memory_percent = sum(m.memory_percent for m in history) / count
        avg_memory_used = sum(m.memory_used_mb for m in history) / count
        avg_memory_available = sum(m.memory_available_mb for m in history) / count
        avg_disk_usage = sum(m.disk_usage_percent for m in history) / count
        avg_disk_free = sum(m.disk_free_gb for m in history) / count

        # For network, use total over period
        total_bytes_sent = sum(m.network_bytes_sent for m in history)
        total_bytes_recv = sum(m.network_bytes_recv for m in history)

        avg_process_count = sum(m.process_count for m in history) / count

        # Average load averages
        avg_load_1 = sum(m.load_average[0] for m in history) / count
        avg_load_5 = sum(m.load_average[1] for m in history) / count
        avg_load_15 = sum(m.load_average[2] for m in history) / count

        return SystemMetrics(
            timestamp=avg_timestamp,
            cpu_percent=avg_cpu,
            memory_percent=avg_memory_percent,
            memory_used_mb=avg_memory_used,
            memory_available_mb=avg_memory_available,
            disk_usage_percent=avg_disk_usage,
            disk_free_gb=avg_disk_free,
            network_bytes_sent=total_bytes_sent,
            network_bytes_recv=total_bytes_recv,
            process_count=int(avg_process_count),
            load_average=[avg_load_1, avg_load_5, avg_load_15]
        )

    def add_alert(self, alert: PerformanceAlert) -> None:
        """Add a performance alert.

        Args:
            alert: PerformanceAlert configuration
        """
        self.alerts[alert.name] = alert
        self._alert_states[alert.name] = []
        self.logger.info(f"Added performance alert: {alert.name}")

    def remove_alert(self, alert_name: str) -> None:
        """Remove a performance alert.

        Args:
            alert_name: Name of the alert to remove
        """
        if alert_name in self.alerts:
            del self.alerts[alert_name]
            del self._alert_states[alert_name]
            self.logger.info(f"Removed performance alert: {alert_name}")

    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check all alerts against current metrics."""
        current_time = datetime.now()

        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue

            # Get metric value
            metric_value = getattr(metrics, alert.metric, None)
            if metric_value is None:
                continue

            # Check threshold
            triggered = self._evaluate_threshold(metric_value, alert)

            # Track trigger state
            alert_history = self._alert_states[alert_name]

            if triggered:
                alert_history.append(current_time)

            # Clean old triggers (outside duration window)
            cutoff_time = current_time - timedelta(seconds=alert.duration_seconds)
            alert_history[:] = [t for t in alert_history if t > cutoff_time]

            # Check if alert should fire
            should_fire = len(alert_history) > 0 and not alert.triggered
            should_clear = len(alert_history) == 0 and alert.triggered

            if should_fire:
                self._fire_alert(alert, metric_value, current_time)
            elif should_clear:
                self._clear_alert(alert, current_time)

    def _evaluate_threshold(self, value: float, alert: PerformanceAlert) -> bool:
        """Evaluate if a metric value triggers an alert threshold."""
        if alert.comparison == "greater":
            return value > alert.threshold
        elif alert.comparison == "less":
            return value < alert.threshold
        elif alert.comparison == "equal":
            return abs(value - alert.threshold) < 0.001  # Float equality
        else:
            return False

    def _fire_alert(self, alert: PerformanceAlert, value: float, trigger_time: datetime) -> None:
        """Fire a performance alert."""
        alert.triggered = True
        alert.trigger_time = trigger_time

        message = (
            f"Performance alert '{alert.name}' triggered: "
            f"{alert.metric} = {value:.2f} (threshold: {alert.threshold:.2f})"
        )

        self.logger.warning(message)

        # Call callback if provided
        if alert.callback:
            try:
                alert.callback(value, alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def _clear_alert(self, alert: PerformanceAlert, clear_time: datetime) -> None:
        """Clear a performance alert."""
        alert.triggered = False
        alert.trigger_time = None

        message = f"Performance alert '{alert.name}' cleared"
        self.logger.info(message)

    def _log_metrics(self, metrics: SystemMetrics) -> None:
        """Log metrics to file."""
        log_data = {
            'timestamp': metrics.timestamp,
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'memory_used_mb': metrics.memory_used_mb,
            'disk_usage_percent': metrics.disk_usage_percent,
            'load_average_1m': metrics.load_average[0],
        }

        self.logger.info(f"METRICS: {json.dumps(log_data)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance status."""
        current = self.get_current_metrics()
        if not current:
            return {"error": "No metrics available"}

        # Calculate recent averages
        recent_avg = self.get_average_metrics(60.0)  # Last minute

        # Count active alerts
        active_alerts = sum(1 for alert in self.alerts.values() if alert.triggered)

        summary = {
            'current': {
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'memory_used_mb': current.memory_used_mb,
                'disk_usage_percent': current.disk_usage_percent,
                'load_average': current.load_average,
            },
            'recent_average': {},
            'alerts': {
                'total': len(self.alerts),
                'active': active_alerts,
                'triggered': [
                    name for name, alert in self.alerts.items()
                    if alert.triggered
                ]
            },
            'monitoring': {
                'active': self._monitoring_active,
                'history_size': len(self.metrics_history),
                'collection_interval': self.config.collection_interval,
            }
        }

        if recent_avg:
            summary['recent_average'] = {
                'cpu_percent': recent_avg.cpu_percent,
                'memory_percent': recent_avg.memory_percent,
                'memory_used_mb': recent_avg.memory_used_mb,
                'disk_usage_percent': recent_avg.disk_usage_percent,
            }

        return summary

    def export_metrics_to_file(self, file_path: Path, format: str = "json") -> None:
        """Export metrics history to file.

        Args:
            file_path: Path to export file
            format: Export format ("json" or "csv")
        """
        if format == "json":
            self._export_json(file_path)
        elif format == "csv":
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, file_path: Path) -> None:
        """Export metrics as JSON."""
        data = {
            'export_time': datetime.now().isoformat(),
            'config': {
                'collection_interval': self.config.collection_interval,
                'history_size': self.config.history_size,
            },
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_mb': m.memory_used_mb,
                    'memory_available_mb': m.memory_available_mb,
                    'disk_usage_percent': m.disk_usage_percent,
                    'disk_free_gb': m.disk_free_gb,
                    'network_bytes_sent': m.network_bytes_sent,
                    'network_bytes_recv': m.network_bytes_recv,
                    'process_count': m.process_count,
                    'load_average': m.load_average,
                }
                for m in self.metrics_history
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics exported to {file_path}")

    def _export_csv(self, file_path: Path) -> None:
        """Export metrics as CSV."""
        import csv

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_mb',
                'memory_available_mb', 'disk_usage_percent', 'disk_free_gb',
                'network_bytes_sent', 'network_bytes_recv', 'process_count',
                'load_avg_1m', 'load_avg_5m', 'load_avg_15m'
            ])

            # Data
            for m in self.metrics_history:
                writer.writerow([
                    m.timestamp, m.cpu_percent, m.memory_percent,
                    m.memory_used_mb, m.memory_available_mb,
                    m.disk_usage_percent, m.disk_free_gb,
                    m.network_bytes_sent, m.network_bytes_recv,
                    m.process_count,
                    m.load_average[0], m.load_average[1], m.load_average[2]
                ])

        self.logger.info(f"Metrics exported to {file_path}")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Convenience functions for common alert configurations
def create_cpu_alert(name: str, threshold: float, callback: Optional[Callable] = None) -> PerformanceAlert:
    """Create a CPU usage alert."""
    return PerformanceAlert(
        name=name,
        metric="cpu_percent",
        threshold=threshold,
        comparison="greater",
        callback=callback
    )


def create_memory_alert(name: str, threshold: float, callback: Optional[Callable] = None) -> PerformanceAlert:
    """Create a memory usage alert."""
    return PerformanceAlert(
        name=name,
        metric="memory_percent",
        threshold=threshold,
        comparison="greater",
        callback=callback
    )


def create_disk_alert(name: str, threshold: float, callback: Optional[Callable] = None) -> PerformanceAlert:
    """Create a disk usage alert."""
    return PerformanceAlert(
        name=name,
        metric="disk_usage_percent",
        threshold=threshold,
        comparison="greater",
        callback=callback
    )


async def async_monitoring_example():
    """Example of async integration with performance monitoring."""
    config = MonitoringConfig(collection_interval=0.5)

    with PerformanceMonitor(config) as monitor:
        # Add some alerts
        monitor.add_alert(create_cpu_alert("high_cpu", 80.0))
        monitor.add_alert(create_memory_alert("high_memory", 85.0))

        # Simulate some work
        for i in range(10):
            await asyncio.sleep(1)
            metrics = monitor.get_current_metrics()
            print(f"Step {i+1}: CPU={metrics.cpu_percent:.1f}%, Memory={metrics.memory_percent:.1f}%")

        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"Performance summary: {summary}")


if __name__ == "__main__":
    # Run async example
    asyncio.run(async_monitoring_example())