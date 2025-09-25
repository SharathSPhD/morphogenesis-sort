"""Data export utilities for metrics and analysis results.

This module provides various data export formats for simulation metrics,
enabling integration with external analysis tools and research workflows.
"""

import asyncio
import csv
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TextIO
import gzip

from ..data.state import StateSnapshot
from .collector import MetricsSnapshot


@dataclass
class ExportConfig:
    """Configuration for data export."""
    output_path: Path
    compress: bool = True
    include_metadata: bool = True
    flatten_nested: bool = True
    timestamp_format: str = "iso"  # iso, unix, or custom
    custom_timestamp_format: Optional[str] = None


class DataExporter(ABC):
    """Abstract base class for data exporters."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.exports_performed = 0
        self.last_export_time = 0.0

    @abstractmethod
    async def export_snapshots(self, snapshots: List[StateSnapshot]) -> bool:
        """Export state snapshots to the configured format."""
        pass

    @abstractmethod
    async def export_metrics(self, metrics: List[MetricsSnapshot]) -> bool:
        """Export metrics snapshots to the configured format."""
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this export format."""
        pass

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp according to configuration."""
        if self.config.timestamp_format == "unix":
            return str(timestamp)
        elif self.config.timestamp_format == "iso":
            import datetime
            return datetime.datetime.fromtimestamp(timestamp).isoformat()
        elif self.config.timestamp_format == "custom" and self.config.custom_timestamp_format:
            import datetime
            return datetime.datetime.fromtimestamp(timestamp).strftime(self.config.custom_timestamp_format)
        else:
            return str(timestamp)

    def _flatten_dict(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary structures."""
        if not self.config.flatten_nested:
            return data

        result = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten_dict(value, new_key))
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
                # Convert numeric lists to statistics
                result[f"{new_key}.count"] = len(value)
                result[f"{new_key}.sum"] = sum(value)
                result[f"{new_key}.mean"] = sum(value) / len(value)
                result[f"{new_key}.min"] = min(value)
                result[f"{new_key}.max"] = max(value)
            else:
                result[new_key] = value

        return result

    def _prepare_snapshot_data(self, snapshot: StateSnapshot) -> Dict[str, Any]:
        """Prepare state snapshot data for export."""
        data = asdict(snapshot)

        # Format timestamp
        data['formatted_timestamp'] = self._format_timestamp(snapshot.timestamp)

        # Flatten if configured
        if self.config.flatten_nested:
            data = self._flatten_dict(data)

        return data

    def _prepare_metrics_data(self, metrics: MetricsSnapshot) -> Dict[str, Any]:
        """Prepare metrics snapshot data for export."""
        data = asdict(metrics)

        # Format timestamp
        data['formatted_timestamp'] = self._format_timestamp(metrics.timestamp)

        # Flatten if configured
        if self.config.flatten_nested:
            data = self._flatten_dict(data)

        return data

    def _update_export_stats(self) -> None:
        """Update export performance statistics."""
        self.exports_performed += 1
        self.last_export_time = time.time()


class CSVExporter(DataExporter):
    """CSV format exporter for tabular data analysis."""

    def get_file_extension(self) -> str:
        return "csv.gz" if self.config.compress else "csv"

    async def export_snapshots(self, snapshots: List[StateSnapshot]) -> bool:
        """Export state snapshots to CSV format."""
        if not snapshots:
            return False

        try:
            # Prepare data
            rows = [self._prepare_snapshot_data(snapshot) for snapshot in snapshots]

            # Generate filename
            timestamp = int(time.time())
            filename = f"state_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Get all possible fieldnames
            all_fields = set()
            for row in rows:
                all_fields.update(row.keys())
            fieldnames = sorted(all_fields)

            # Write CSV
            if self.config.compress:
                with gzip.open(filepath, 'wt', newline='') as f:
                    await self._write_csv(f, rows, fieldnames)
            else:
                with open(filepath, 'w', newline='') as f:
                    await self._write_csv(f, rows, fieldnames)

            self._update_export_stats()
            return True

        except Exception:
            return False

    async def export_metrics(self, metrics: List[MetricsSnapshot]) -> bool:
        """Export metrics snapshots to CSV format."""
        if not metrics:
            return False

        try:
            # Prepare data
            rows = [self._prepare_metrics_data(metric) for metric in metrics]

            # Generate filename
            timestamp = int(time.time())
            filename = f"metrics_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Get all possible fieldnames
            all_fields = set()
            for row in rows:
                all_fields.update(row.keys())
            fieldnames = sorted(all_fields)

            # Write CSV
            if self.config.compress:
                with gzip.open(filepath, 'wt', newline='') as f:
                    await self._write_csv(f, rows, fieldnames)
            else:
                with open(filepath, 'w', newline='') as f:
                    await self._write_csv(f, rows, fieldnames)

            self._update_export_stats()
            return True

        except Exception:
            return False

    async def _write_csv(self, file: TextIO, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        """Write CSV data to file."""
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            # Fill missing fields with None
            complete_row = {field: row.get(field) for field in fieldnames}
            writer.writerow(complete_row)


class JSONExporter(DataExporter):
    """JSON format exporter for structured data."""

    def get_file_extension(self) -> str:
        return "json.gz" if self.config.compress else "json"

    async def export_snapshots(self, snapshots: List[StateSnapshot]) -> bool:
        """Export state snapshots to JSON format."""
        if not snapshots:
            return False

        try:
            # Prepare data
            data = {
                'export_metadata': {
                    'timestamp': time.time(),
                    'formatted_timestamp': self._format_timestamp(time.time()),
                    'snapshot_count': len(snapshots),
                    'exporter': 'JSONExporter',
                    'version': '1.0'
                } if self.config.include_metadata else {},
                'snapshots': [self._prepare_snapshot_data(snapshot) for snapshot in snapshots]
            }

            # Generate filename
            timestamp = int(time.time())
            filename = f"state_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON
            json_str = json.dumps(data, indent=2, default=str)

            if self.config.compress:
                compressed = gzip.compress(json_str.encode('utf-8'))
                with open(filepath, 'wb') as f:
                    f.write(compressed)
            else:
                with open(filepath, 'w') as f:
                    f.write(json_str)

            self._update_export_stats()
            return True

        except Exception:
            return False

    async def export_metrics(self, metrics: List[MetricsSnapshot]) -> bool:
        """Export metrics snapshots to JSON format."""
        if not metrics:
            return False

        try:
            # Prepare data
            data = {
                'export_metadata': {
                    'timestamp': time.time(),
                    'formatted_timestamp': self._format_timestamp(time.time()),
                    'metrics_count': len(metrics),
                    'exporter': 'JSONExporter',
                    'version': '1.0'
                } if self.config.include_metadata else {},
                'metrics': [self._prepare_metrics_data(metric) for metric in metrics]
            }

            # Generate filename
            timestamp = int(time.time())
            filename = f"metrics_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON
            json_str = json.dumps(data, indent=2, default=str)

            if self.config.compress:
                compressed = gzip.compress(json_str.encode('utf-8'))
                with open(filepath, 'wb') as f:
                    f.write(compressed)
            else:
                with open(filepath, 'w') as f:
                    f.write(json_str)

            self._update_export_stats()
            return True

        except Exception:
            return False


class ParquetExporter(DataExporter):
    """Parquet format exporter for high-performance columnar analysis."""

    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self._check_parquet_available()

    def _check_parquet_available(self) -> None:
        """Check if parquet dependencies are available."""
        try:
            import pandas as pd
            import pyarrow as pa
            self.pd = pd
            self.pa = pa
            self.parquet_available = True
        except ImportError:
            self.parquet_available = False

    def get_file_extension(self) -> str:
        return "parquet.gz" if self.config.compress else "parquet"

    async def export_snapshots(self, snapshots: List[StateSnapshot]) -> bool:
        """Export state snapshots to Parquet format."""
        if not self.parquet_available or not snapshots:
            return False

        try:
            # Prepare data
            rows = [self._prepare_snapshot_data(snapshot) for snapshot in snapshots]

            # Convert to DataFrame
            df = self.pd.DataFrame(rows)

            # Generate filename
            timestamp = int(time.time())
            filename = f"state_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write Parquet
            compression = 'gzip' if self.config.compress else None
            df.to_parquet(filepath, compression=compression, index=False)

            self._update_export_stats()
            return True

        except Exception:
            return False

    async def export_metrics(self, metrics: List[MetricsSnapshot]) -> bool:
        """Export metrics snapshots to Parquet format."""
        if not self.parquet_available or not metrics:
            return False

        try:
            # Prepare data
            rows = [self._prepare_metrics_data(metric) for metric in metrics]

            # Convert to DataFrame
            df = self.pd.DataFrame(rows)

            # Generate filename
            timestamp = int(time.time())
            filename = f"metrics_snapshots_{timestamp}.{self.get_file_extension()}"
            filepath = self.config.output_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write Parquet
            compression = 'gzip' if self.config.compress else None
            df.to_parquet(filepath, compression=compression, index=False)

            self._update_export_stats()
            return True

        except Exception:
            return False


class MultiFormatExporter:
    """Export data to multiple formats simultaneously."""

    def __init__(self, exporters: List[DataExporter]):
        self.exporters = exporters
        self.exports_performed = 0

    async def export_snapshots(self, snapshots: List[StateSnapshot]) -> Dict[str, bool]:
        """Export snapshots using all configured exporters."""
        results = {}

        # Run all exports concurrently
        tasks = [
            asyncio.create_task(exporter.export_snapshots(snapshots))
            for exporter in self.exporters
        ]

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for exporter, result in zip(self.exporters, completed_results):
            exporter_name = exporter.__class__.__name__
            if isinstance(result, Exception):
                results[exporter_name] = False
            else:
                results[exporter_name] = result

        if any(results.values()):
            self.exports_performed += 1

        return results

    async def export_metrics(self, metrics: List[MetricsSnapshot]) -> Dict[str, bool]:
        """Export metrics using all configured exporters."""
        results = {}

        # Run all exports concurrently
        tasks = [
            asyncio.create_task(exporter.export_metrics(metrics))
            for exporter in self.exporters
        ]

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for exporter, result in zip(self.exporters, completed_results):
            exporter_name = exporter.__class__.__name__
            if isinstance(result, Exception):
                results[exporter_name] = False
            else:
                results[exporter_name] = result

        if any(results.values()):
            self.exports_performed += 1

        return results

    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics for all exporters."""
        stats = {
            'total_exports': self.exports_performed,
            'exporters': {}
        }

        for exporter in self.exporters:
            exporter_name = exporter.__class__.__name__
            stats['exporters'][exporter_name] = {
                'exports_performed': exporter.exports_performed,
                'last_export_time': exporter.last_export_time,
                'file_extension': exporter.get_file_extension(),
            }

        return stats


# Factory functions for creating common exporter configurations
def create_csv_exporter(output_path: Path, compress: bool = True) -> CSVExporter:
    """Create a CSV exporter with standard configuration."""
    config = ExportConfig(
        output_path=output_path,
        compress=compress,
        flatten_nested=True,
        timestamp_format="iso"
    )
    return CSVExporter(config)


def create_json_exporter(output_path: Path, compress: bool = True) -> JSONExporter:
    """Create a JSON exporter with standard configuration."""
    config = ExportConfig(
        output_path=output_path,
        compress=compress,
        include_metadata=True,
        flatten_nested=False,
        timestamp_format="iso"
    )
    return JSONExporter(config)


def create_parquet_exporter(output_path: Path, compress: bool = True) -> ParquetExporter:
    """Create a Parquet exporter with standard configuration."""
    config = ExportConfig(
        output_path=output_path,
        compress=compress,
        flatten_nested=True,
        timestamp_format="unix"
    )
    return ParquetExporter(config)


def create_multi_format_exporter(output_path: Path) -> MultiFormatExporter:
    """Create a multi-format exporter with CSV, JSON, and Parquet."""
    exporters = [
        create_csv_exporter(output_path / "csv"),
        create_json_exporter(output_path / "json"),
        create_parquet_exporter(output_path / "parquet")
    ]
    return MultiFormatExporter(exporters)