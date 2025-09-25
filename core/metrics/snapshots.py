"""Snapshot management for immutable state capture and analysis.

This module provides efficient management of simulation state snapshots
for analysis and visualization, using immutable data structures to
ensure data integrity and enable safe concurrent access.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, Tuple, Callable
import threading
import pickle
import gzip
from pathlib import Path

from ..data.state import StateSnapshot
from ..data.types import SimulationTime


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""
    timestep: SimulationTime
    timestamp: float
    size_bytes: int
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    checksum: Optional[str] = None


class SnapshotManager:
    """Manager for efficient snapshot storage and retrieval.

    Provides memory-efficient storage of simulation snapshots with
    optional compression, indexing, and persistence capabilities.
    Features circular buffering to limit memory usage.
    """

    def __init__(
        self,
        max_snapshots: int = 10000,
        enable_compression: bool = True,
        compression_level: int = 6,
        persistence_path: Optional[Path] = None
    ):
        self.max_snapshots = max_snapshots
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Snapshot storage
        self.snapshots: deque[StateSnapshot] = deque(maxlen=max_snapshots)
        self.metadata: deque[SnapshotMetadata] = deque(maxlen=max_snapshots)

        # Indexing for fast retrieval
        self.timestep_index: Dict[SimulationTime, int] = {}
        self.timestamp_index: List[Tuple[float, int]] = []

        # Statistics
        self.total_snapshots_added = 0
        self.total_bytes_stored = 0
        self.total_compressed_bytes = 0

        # Thread safety
        self._lock = threading.RLock()

        # Persistence settings
        self.auto_persist = False
        self.persist_interval = 100  # Persist every N snapshots

    async def add_snapshot(self, snapshot: StateSnapshot) -> None:
        """Add a new snapshot to the manager."""
        with self._lock:
            # Calculate snapshot size
            snapshot_bytes = self._estimate_size(snapshot)

            # Compress if enabled
            compressed_data = None
            compressed_size = snapshot_bytes
            compression_ratio = 1.0

            if self.enable_compression:
                try:
                    serialized = pickle.dumps(snapshot)
                    compressed_data = gzip.compress(serialized, self.compression_level)
                    compressed_size = len(compressed_data)
                    compression_ratio = compressed_size / len(serialized) if serialized else 1.0
                except Exception:
                    # Fall back to uncompressed
                    compressed_data = None
                    compressed_size = snapshot_bytes

            # Create metadata
            metadata = SnapshotMetadata(
                timestep=snapshot.timestep,
                timestamp=snapshot.timestamp,
                size_bytes=snapshot_bytes,
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio
            )

            # Add to storage
            self.snapshots.append(snapshot)
            self.metadata.append(metadata)

            # Update indexes
            self._update_indexes(snapshot, len(self.snapshots) - 1)

            # Update statistics
            self.total_snapshots_added += 1
            self.total_bytes_stored += snapshot_bytes
            self.total_compressed_bytes += compressed_size

            # Auto-persist if enabled
            if self.auto_persist and self.persistence_path:
                if self.total_snapshots_added % self.persist_interval == 0:
                    await self._persist_snapshots()

    def _update_indexes(self, snapshot: StateSnapshot, index: int) -> None:
        """Update internal indexes for fast retrieval."""
        # Update timestep index
        self.timestep_index[snapshot.timestep] = index

        # Update timestamp index (keep sorted)
        self.timestamp_index.append((snapshot.timestamp, index))
        self.timestamp_index.sort(key=lambda x: x[0])

        # Maintain index size limits
        if len(self.timestamp_index) > self.max_snapshots:
            self.timestamp_index = self.timestamp_index[-self.max_snapshots:]

    def _estimate_size(self, snapshot: StateSnapshot) -> int:
        """Estimate memory size of a snapshot."""
        # Rough estimation based on snapshot content
        base_size = 1000  # Base overhead

        # Population metrics
        base_size += snapshot.population_size * 10

        # Type and state counts
        base_size += len(snapshot.type_counts) * 50
        base_size += len(snapshot.state_counts) * 50

        # Custom metrics
        base_size += len(snapshot.custom_metrics) * 100

        return base_size

    def get_snapshot_by_timestep(self, timestep: SimulationTime) -> Optional[StateSnapshot]:
        """Get snapshot for a specific timestep."""
        with self._lock:
            index = self.timestep_index.get(timestep)
            if index is not None and index < len(self.snapshots):
                return self.snapshots[index]
            return None

    def get_snapshot_by_timestamp(self, timestamp: float, tolerance: float = 1.0) -> Optional[StateSnapshot]:
        """Get snapshot closest to a specific timestamp."""
        with self._lock:
            if not self.timestamp_index:
                return None

            # Binary search for closest timestamp
            best_diff = float('inf')
            best_index = None

            for snap_timestamp, index in self.timestamp_index:
                diff = abs(snap_timestamp - timestamp)
                if diff <= tolerance and diff < best_diff:
                    best_diff = diff
                    best_index = index

            if best_index is not None and best_index < len(self.snapshots):
                return self.snapshots[best_index]

            return None

    def get_snapshots_in_range(
        self,
        start_timestep: Optional[SimulationTime] = None,
        end_timestep: Optional[SimulationTime] = None,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        max_count: Optional[int] = None
    ) -> List[StateSnapshot]:
        """Get snapshots within specified ranges."""
        with self._lock:
            filtered_snapshots = []

            for snapshot in self.snapshots:
                # Filter by timestep range
                if start_timestep is not None and snapshot.timestep < start_timestep:
                    continue
                if end_timestep is not None and snapshot.timestep > end_timestep:
                    continue

                # Filter by timestamp range
                if start_timestamp is not None and snapshot.timestamp < start_timestamp:
                    continue
                if end_timestamp is not None and snapshot.timestamp > end_timestamp:
                    continue

                filtered_snapshots.append(snapshot)

                # Limit count if specified
                if max_count and len(filtered_snapshots) >= max_count:
                    break

            return filtered_snapshots

    def get_latest_snapshots(self, count: int = 100) -> List[StateSnapshot]:
        """Get the most recent snapshots."""
        with self._lock:
            return list(self.snapshots)[-count:]

    def get_snapshots_at_interval(self, interval: int = 10) -> List[StateSnapshot]:
        """Get snapshots at regular intervals."""
        with self._lock:
            result = []
            for i in range(0, len(self.snapshots), interval):
                result.append(self.snapshots[i])
            return result

    def analyze_metric_over_time(
        self,
        metric_extractor: Callable[[StateSnapshot], float],
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze how a metric changes over time."""
        with self._lock:
            snapshots = list(self.snapshots)
            if window_size:
                snapshots = snapshots[-window_size:]

            if not snapshots:
                return {'error': 'no_snapshots'}

            values = []
            timestamps = []
            timesteps = []

            for snapshot in snapshots:
                try:
                    value = metric_extractor(snapshot)
                    values.append(value)
                    timestamps.append(snapshot.timestamp)
                    timesteps.append(snapshot.timestep)
                except Exception:
                    continue

            if not values:
                return {'error': 'no_valid_values'}

            # Calculate statistics
            analysis = {
                'count': len(values),
                'min_value': min(values),
                'max_value': max(values),
                'mean_value': sum(values) / len(values),
                'start_time': min(timestamps),
                'end_time': max(timestamps),
                'start_timestep': min(timesteps),
                'end_timestep': max(timesteps),
                'values': values,
                'timestamps': timestamps,
                'timesteps': timesteps
            }

            # Calculate trend
            if len(values) > 1:
                # Simple linear trend
                n = len(values)
                sum_x = sum(range(n))
                sum_y = sum(values)
                sum_xy = sum(i * values[i] for i in range(n))
                sum_xx = sum(i * i for i in range(n))

                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                analysis['trend_slope'] = slope
                analysis['trend_direction'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

            return analysis

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get statistics about snapshot storage."""
        with self._lock:
            if not self.metadata:
                return {'status': 'empty'}

            total_size = sum(m.size_bytes for m in self.metadata)
            total_compressed = sum(m.compressed_size_bytes for m in self.metadata)

            stats = {
                'total_snapshots': len(self.snapshots),
                'max_snapshots': self.max_snapshots,
                'utilization': len(self.snapshots) / self.max_snapshots,
                'total_size_bytes': total_size,
                'total_compressed_bytes': total_compressed,
                'average_size_bytes': total_size / len(self.metadata) if self.metadata else 0,
                'compression_ratio': total_compressed / total_size if total_size > 0 else 1.0,
                'snapshots_added_total': self.total_snapshots_added,
                'compression_enabled': self.enable_compression,
            }

            # Time range
            if self.snapshots:
                stats['time_range'] = {
                    'start_timestep': self.snapshots[0].timestep,
                    'end_timestep': self.snapshots[-1].timestep,
                    'start_timestamp': self.snapshots[0].timestamp,
                    'end_timestamp': self.snapshots[-1].timestamp,
                    'duration_seconds': self.snapshots[-1].timestamp - self.snapshots[0].timestamp
                }

            return stats

    async def _persist_snapshots(self) -> None:
        """Persist snapshots to disk."""
        if not self.persistence_path:
            return

        try:
            # Create directory if needed
            self.persistence_path.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            filename = f"snapshots_{int(time.time())}.pkl.gz"
            filepath = self.persistence_path / filename

            # Serialize and compress
            with self._lock:
                data = {
                    'snapshots': list(self.snapshots),
                    'metadata': list(self.metadata),
                    'statistics': self.get_storage_statistics()
                }

            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized, self.compression_level)

            # Write to file
            with open(filepath, 'wb') as f:
                f.write(compressed)

        except Exception:
            # Ignore persistence errors to avoid disrupting simulation
            pass

    async def load_snapshots(self, filepath: Path) -> bool:
        """Load snapshots from disk."""
        try:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()

            serialized = gzip.decompress(compressed_data)
            data = pickle.loads(serialized)

            with self._lock:
                self.snapshots.clear()
                self.metadata.clear()
                self.timestep_index.clear()
                self.timestamp_index.clear()

                # Restore snapshots
                for snapshot in data['snapshots']:
                    self.snapshots.append(snapshot)

                # Restore metadata
                for metadata in data['metadata']:
                    self.metadata.append(metadata)

                # Rebuild indexes
                for i, snapshot in enumerate(self.snapshots):
                    self._update_indexes(snapshot, i)

            return True

        except Exception:
            return False

    def clear_snapshots(self) -> None:
        """Clear all stored snapshots."""
        with self._lock:
            self.snapshots.clear()
            self.metadata.clear()
            self.timestep_index.clear()
            self.timestamp_index.clear()
            self.total_snapshots_added = 0
            self.total_bytes_stored = 0
            self.total_compressed_bytes = 0

    async def cleanup(self) -> None:
        """Clean up resources and persist if enabled."""
        if self.auto_persist and self.persistence_path:
            await self._persist_snapshots()

        self.clear_snapshots()

    def __len__(self) -> int:
        """Get number of stored snapshots."""
        return len(self.snapshots)

    def __iter__(self) -> Iterator[StateSnapshot]:
        """Iterate over stored snapshots."""
        return iter(self.snapshots)

    def __str__(self) -> str:
        """String representation of snapshot manager."""
        return (
            f"SnapshotManager(snapshots={len(self.snapshots)}, "
            f"max={self.max_snapshots}, compression={self.enable_compression})"
        )