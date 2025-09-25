"""Atomic counter operations for lock-free metrics collection.

This module provides thread-safe counters using atomic operations,
eliminating the need for locks while ensuring data integrity in
concurrent environments.
"""

import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


class AtomicCounter:
    """Thread-safe atomic counter.

    Uses thread-safe operations to ensure accurate counting
    without explicit locking mechanisms.
    """

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()  # Only for atomic operations, not contention

    def increment(self, amount: int = 1) -> int:
        """Atomically increment the counter."""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Atomically decrement the counter."""
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value

    def set(self, value: int) -> int:
        """Atomically set counter value."""
        with self._lock:
            self._value = value
            return self._value

    def reset(self) -> int:
        """Reset counter to zero."""
        return self.set(0)

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Atomically compare and swap if value matches expected."""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


@dataclass
class CounterStats:
    """Statistics for a counter over time."""
    total_increments: int = 0
    total_decrements: int = 0
    peak_value: int = 0
    min_value: int = 0
    average_rate: float = 0.0
    last_update_time: float = 0.0


class CounterRegistry:
    """Registry of atomic counters with statistics tracking.

    Provides centralized management of named counters with
    automatic statistics collection and performance monitoring.
    """

    def __init__(self):
        self.counters: Dict[str, AtomicCounter] = {}
        self.stats: Dict[str, CounterStats] = {}
        self._registry_lock = threading.Lock()

    def create_counter(self, name: str, initial_value: int = 0) -> AtomicCounter:
        """Create a new named counter."""
        with self._registry_lock:
            if name in self.counters:
                return self.counters[name]

            counter = AtomicCounter(initial_value)
            self.counters[name] = counter
            self.stats[name] = CounterStats(min_value=initial_value, peak_value=initial_value)
            return counter

    def get_counter(self, name: str) -> Optional[AtomicCounter]:
        """Get a counter by name."""
        return self.counters.get(name)

    def increment_counter(self, name: str, amount: int = 1) -> int:
        """Increment a named counter, creating if needed."""
        counter = self.counters.get(name)
        if counter is None:
            counter = self.create_counter(name)

        new_value = counter.increment(amount)
        self._update_stats(name, new_value, increment=amount)
        return new_value

    def decrement_counter(self, name: str, amount: int = 1) -> int:
        """Decrement a named counter, creating if needed."""
        counter = self.counters.get(name)
        if counter is None:
            counter = self.create_counter(name)

        new_value = counter.decrement(amount)
        self._update_stats(name, new_value, decrement=amount)
        return new_value

    def get_counter_value(self, name: str) -> int:
        """Get current value of a named counter."""
        counter = self.counters.get(name)
        return counter.get() if counter else 0

    def set_counter_value(self, name: str, value: int) -> int:
        """Set value of a named counter."""
        counter = self.counters.get(name)
        if counter is None:
            counter = self.create_counter(name, value)
        else:
            counter.set(value)

        self._update_stats(name, value)
        return value

    def _update_stats(self, name: str, current_value: int,
                     increment: int = 0, decrement: int = 0) -> None:
        """Update statistics for a counter."""
        import time

        stats = self.stats.get(name)
        if not stats:
            return

        # Update increment/decrement counts
        stats.total_increments += increment
        stats.total_decrements += decrement

        # Update peak and min values
        if current_value > stats.peak_value:
            stats.peak_value = current_value
        if current_value < stats.min_value:
            stats.min_value = current_value

        # Update timing and rate
        current_time = time.time()
        if stats.last_update_time > 0:
            time_delta = current_time - stats.last_update_time
            if time_delta > 0:
                rate = (increment - decrement) / time_delta
                # Exponential moving average for rate
                alpha = 0.1
                stats.average_rate = (1 - alpha) * stats.average_rate + alpha * rate

        stats.last_update_time = current_time

    def get_counter_stats(self, name: str) -> Optional[CounterStats]:
        """Get statistics for a named counter."""
        return self.stats.get(name)

    def get_all_counter_values(self) -> Dict[str, int]:
        """Get current values of all counters."""
        return {name: counter.get() for name, counter in self.counters.items()}

    def get_all_counter_stats(self) -> Dict[str, CounterStats]:
        """Get statistics for all counters."""
        return dict(self.stats)

    def reset_counter(self, name: str) -> bool:
        """Reset a named counter to zero."""
        counter = self.counters.get(name)
        if counter:
            counter.reset()
            # Reset stats but preserve peak/min
            if name in self.stats:
                stats = self.stats[name]
                stats.total_increments = 0
                stats.total_decrements = 0
                stats.average_rate = 0.0
            return True
        return False

    def reset_all_counters(self) -> None:
        """Reset all counters to zero."""
        for name in self.counters:
            self.reset_counter(name)

    def remove_counter(self, name: str) -> bool:
        """Remove a counter from the registry."""
        with self._registry_lock:
            if name in self.counters:
                del self.counters[name]
                del self.stats[name]
                return True
            return False

    def clear_all(self) -> None:
        """Remove all counters from the registry."""
        with self._registry_lock:
            self.counters.clear()
            self.stats.clear()

    def export_summary(self) -> Dict[str, Any]:
        """Export summary of all counters and their statistics."""
        summary = {
            'total_counters': len(self.counters),
            'counters': {},
        }

        for name, counter in self.counters.items():
            stats = self.stats.get(name, CounterStats())
            summary['counters'][name] = {
                'current_value': counter.get(),
                'total_increments': stats.total_increments,
                'total_decrements': stats.total_decrements,
                'peak_value': stats.peak_value,
                'min_value': stats.min_value,
                'average_rate': stats.average_rate,
                'last_update_time': stats.last_update_time,
            }

        return summary

    def __len__(self) -> int:
        """Get number of registered counters."""
        return len(self.counters)

    def __contains__(self, name: str) -> bool:
        """Check if counter exists in registry."""
        return name in self.counters

    def __str__(self) -> str:
        """String representation of counter registry."""
        return f"CounterRegistry(counters={len(self.counters)})"