"""Async logging utilities for high-performance logging in async environments.

Provides non-blocking logging capabilities optimized for async applications
with queue-based buffering and batch processing.
"""

import asyncio
import logging
import threading
import queue
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from .formatters import JSONFormatter


@dataclass
class AsyncLogRecord:
    """Async-friendly log record."""
    timestamp: float
    level: int
    level_name: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    extra_data: Dict[str, Any]
    exception_info: Optional[str] = None


class AsyncLogHandler(logging.Handler):
    """Async-compatible log handler with queue-based processing."""

    def __init__(
        self,
        queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0
    ):
        super().__init__()
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Thread-safe queue for log records
        self.log_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.shutdown_event = threading.Event()

        # Background thread for processing logs
        self._processor_thread = threading.Thread(
            target=self._process_logs,
            daemon=True
        )
        self._processor_thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the queue."""
        try:
            # Convert to async log record
            async_record = AsyncLogRecord(
                timestamp=record.created,
                level=record.levelno,
                level_name=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                extra_data=getattr(record, '__dict__', {}),
                exception_info=self.formatException(record.exc_info) if record.exc_info else None
            )

            # Add to queue (non-blocking)
            self.log_queue.put_nowait(async_record)

        except queue.Full:
            # Queue is full, drop the record or handle overflow
            self.handleError(record)

    def _process_logs(self) -> None:
        """Process log records in background thread."""
        batch = []
        last_flush = datetime.now()

        while not self.shutdown_event.is_set():
            try:
                # Get records with timeout
                try:
                    record = self.log_queue.get(timeout=0.1)
                    batch.append(record)
                except queue.Empty:
                    record = None

                # Check if we should flush
                now = datetime.now()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (now - last_flush).total_seconds() >= self.flush_interval)
                )

                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                # Log processing error (avoid infinite recursion)
                print(f"Async log handler error: {e}")

        # Final flush
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[AsyncLogRecord]) -> None:
        """Flush a batch of log records."""
        # Override in subclasses to implement actual flushing
        pass

    def close(self) -> None:
        """Close the handler and cleanup resources."""
        self.shutdown_event.set()
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)
        super().close()


class AsyncFileHandler(AsyncLogHandler):
    """Async file handler that writes logs to files asynchronously."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = 'a',
        encoding: str = 'utf-8',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filename = Path(filename)
        self.mode = mode
        self.encoding = encoding
        self.formatter = JSONFormatter()

        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def _flush_batch(self, batch: List[AsyncLogRecord]) -> None:
        """Flush batch to file."""
        try:
            lines = []
            for record in batch:
                # Format record
                formatted = self._format_record(record)
                lines.append(formatted + '\n')

            # Write all lines at once
            with open(self.filename, self.mode, encoding=self.encoding) as f:
                f.writelines(lines)

        except Exception as e:
            print(f"Error writing log batch to file: {e}")

    def _format_record(self, record: AsyncLogRecord) -> str:
        """Format an async log record."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.timestamp).isoformat(),
            'level': record.level_name,
            'logger': record.logger_name,
            'message': record.message,
            'module': record.module,
            'function': record.function,
            'line': record.line_number,
        }

        if record.exception_info:
            log_data['exception'] = record.exception_info

        # Add extra data
        if record.extra_data:
            log_data['extra'] = record.extra_data

        return json.dumps(log_data, ensure_ascii=False)


class AsyncLogger:
    """High-performance async logger with queue-based processing."""

    def __init__(
        self,
        name: str,
        handlers: Optional[List[AsyncLogHandler]] = None,
        level: int = logging.INFO
    ):
        self.name = name
        self.level = level
        self.handlers = handlers or []

        # Create underlying logger for compatibility
        self._logger = logging.getLogger(f"async.{name}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        # Add async handlers to the logger
        for handler in self.handlers:
            self._logger.addHandler(handler)

    async def log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log a message asynchronously."""
        if level < self.level:
            return

        # Create log record
        record = self._logger.makeRecord(
            self.name, level, __file__, 0, message, (), exc_info, extra=extra
        )

        # Emit to all handlers (non-blocking)
        for handler in self.handlers:
            if level >= handler.level:
                handler.emit(record)

    async def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log debug message."""
        await self.log(logging.DEBUG, message, extra)

    async def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log info message."""
        await self.log(logging.INFO, message, extra)

    async def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log warning message."""
        await self.log(logging.WARNING, message, extra)

    async def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log error message."""
        await self.log(logging.ERROR, message, extra, exc_info)

    async def critical(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log critical message."""
        await self.log(logging.CRITICAL, message, extra, exc_info)

    def add_handler(self, handler: AsyncLogHandler) -> None:
        """Add a log handler."""
        self.handlers.append(handler)
        self._logger.addHandler(handler)

    def remove_handler(self, handler: AsyncLogHandler) -> None:
        """Remove a log handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            self._logger.removeHandler(handler)

    async def close(self) -> None:
        """Close all handlers."""
        for handler in self.handlers:
            handler.close()


class AsyncBufferedHandler(AsyncLogHandler):
    """Async handler that buffers logs in memory with periodic flushing."""

    def __init__(
        self,
        buffer_size: int = 1000,
        auto_flush_interval: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.auto_flush_interval = auto_flush_interval
        self.buffer: List[AsyncLogRecord] = []
        self._last_flush = datetime.now()
        self._buffer_lock = threading.RLock()

    def _flush_batch(self, batch: List[AsyncLogRecord]) -> None:
        """Add batch to buffer and flush if needed."""
        with self._buffer_lock:
            self.buffer.extend(batch)

            # Check if we need to flush
            now = datetime.now()
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                (now - self._last_flush).total_seconds() >= self.auto_flush_interval
            )

            if should_flush:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the internal buffer."""
        if not self.buffer:
            return

        # Override in subclasses to implement actual flushing
        # For base class, just clear the buffer
        self.buffer.clear()
        self._last_flush = datetime.now()

    def get_buffered_records(self) -> List[AsyncLogRecord]:
        """Get copy of buffered records."""
        with self._buffer_lock:
            return self.buffer.copy()

    def clear_buffer(self) -> None:
        """Clear the buffer."""
        with self._buffer_lock:
            self.buffer.clear()


class AsyncAggregateHandler(AsyncLogHandler):
    """Handler that aggregates log statistics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats = {
            'total_records': 0,
            'records_by_level': {
                'DEBUG': 0,
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0
            },
            'records_by_logger': {},
            'errors_per_minute': [],
            'start_time': datetime.now()
        }
        self._stats_lock = threading.RLock()

    def _flush_batch(self, batch: List[AsyncLogRecord]) -> None:
        """Update statistics with batch records."""
        with self._stats_lock:
            for record in batch:
                self.stats['total_records'] += 1
                self.stats['records_by_level'][record.level_name] += 1

                logger_name = record.logger_name
                if logger_name not in self.stats['records_by_logger']:
                    self.stats['records_by_logger'][logger_name] = 0
                self.stats['records_by_logger'][logger_name] += 1

                # Track error rate
                if record.level >= logging.ERROR:
                    current_minute = int(record.timestamp / 60)
                    self.stats['errors_per_minute'].append(current_minute)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics."""
        with self._stats_lock:
            stats = self.stats.copy()

            # Calculate uptime
            uptime = (datetime.now() - stats['start_time']).total_seconds()
            stats['uptime_seconds'] = uptime

            # Calculate rates
            if uptime > 0:
                stats['records_per_second'] = stats['total_records'] / uptime

            # Calculate recent error rate
            if stats['errors_per_minute']:
                current_minute = int(datetime.now().timestamp() / 60)
                recent_errors = [
                    minute for minute in stats['errors_per_minute']
                    if current_minute - minute <= 5  # Last 5 minutes
                ]
                stats['recent_error_rate'] = len(recent_errors) / 5

            return stats


class AsyncRemoteHandler(AsyncLogHandler):
    """Handler that sends logs to a remote endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = None  # Will be initialized when needed

    def _flush_batch(self, batch: List[AsyncLogRecord]) -> None:
        """Send batch to remote endpoint."""
        try:
            # Convert records to JSON
            records_data = []
            for record in batch:
                record_dict = {
                    'timestamp': record.timestamp,
                    'level': record.level_name,
                    'logger': record.logger_name,
                    'message': record.message,
                    'module': record.module,
                    'function': record.function,
                    'line': record.line_number,
                }

                if record.exception_info:
                    record_dict['exception'] = record.exception_info

                if record.extra_data:
                    record_dict['extra'] = record.extra_data

                records_data.append(record_dict)

            # Send to remote endpoint (simplified - would use actual HTTP client)
            self._send_to_remote(records_data)

        except Exception as e:
            print(f"Error sending logs to remote endpoint: {e}")

    def _send_to_remote(self, records: List[Dict[str, Any]]) -> None:
        """Send records to remote endpoint."""
        # This is a placeholder - implement actual HTTP client
        # In practice, you'd use aiohttp or requests
        pass


def create_async_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    enable_stats: bool = True
) -> AsyncLogger:
    """Create a configured async logger.

    Args:
        name: Logger name
        log_file: Optional file for logging
        level: Logging level
        enable_stats: Whether to enable statistics collection

    Returns:
        Configured AsyncLogger instance
    """
    handlers = []

    # Add file handler if specified
    if log_file:
        file_handler = AsyncFileHandler(log_file)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Add stats handler if enabled
    if enable_stats:
        stats_handler = AsyncAggregateHandler()
        stats_handler.setLevel(logging.DEBUG)  # Capture all for stats
        handlers.append(stats_handler)

    return AsyncLogger(name, handlers, level)


async def test_async_logging():
    """Test function for async logging."""
    logger = create_async_logger("test", Path("test_async.log"))

    # Log some messages
    await logger.info("Starting async logging test")
    await logger.debug("Debug message with extra data", extra={"test_value": 42})
    await logger.warning("Warning message")
    await logger.error("Error message", exc_info=True)

    # Wait a bit for processing
    await asyncio.sleep(2)

    # Close logger
    await logger.close()

    print("Async logging test completed")


if __name__ == "__main__":
    asyncio.run(test_async_logging())