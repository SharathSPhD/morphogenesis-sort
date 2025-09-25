"""Data serialization and persistence utilities.

This module provides comprehensive data serialization capabilities for the
morphogenesis simulation, including JSON, pickle, HDF5, and custom binary
formats for efficient storage and retrieval of simulation data.
"""

import json
import pickle
import gzip
import asyncio
import aiofiles
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Type, TypeVar
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
import logging

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from .types import (
    CellID, Position, CellType, CellState, CellParameters,
    WorldParameters, ExperimentMetadata, SimulationTime
)
from .state import CellData, SimulationState
from .actions import CellAction

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SerializationFormat:
    """Enumeration of supported serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_PICKLE = "compressed_pickle"
    HDF5 = "hdf5"
    CUSTOM_BINARY = "custom_binary"
    CSV = "csv"


@dataclass
class SerializationMetadata:
    """Metadata for serialized data files."""
    format: str
    version: str
    timestamp: datetime
    data_type: str
    compression: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: int = 0


class AsyncDataSerializer:
    """Async data serialization manager.

    Provides high-performance async serialization with support for
    multiple formats, compression, and streaming operations.
    """

    def __init__(self, base_path: Union[str, Path] = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Format-specific serializers
        self._serializers = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.COMPRESSED_PICKLE: self._serialize_compressed_pickle,
            SerializationFormat.HDF5: self._serialize_hdf5,
            SerializationFormat.CUSTOM_BINARY: self._serialize_custom_binary,
            SerializationFormat.CSV: self._serialize_csv,
        }

        # Format-specific deserializers
        self._deserializers = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.COMPRESSED_PICKLE: self._deserialize_compressed_pickle,
            SerializationFormat.HDF5: self._deserialize_hdf5,
            SerializationFormat.CUSTOM_BINARY: self._deserialize_custom_binary,
            SerializationFormat.CSV: self._deserialize_csv,
        }

    async def serialize_data(
        self,
        data: Any,
        filename: str,
        format: str = SerializationFormat.JSON,
        compress: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SerializationMetadata:
        """Serialize data to file with specified format.

        Args:
            data: Data to serialize
            filename: Target filename (without extension)
            format: Serialization format
            compress: Whether to compress the data
            metadata: Additional metadata to store

        Returns:
            SerializationMetadata with information about the serialized file
        """
        # Add format extension to filename
        file_extension = self._get_file_extension(format, compress)
        full_filename = f"{filename}{file_extension}"
        file_path = self.base_path / full_filename

        # Prepare data for serialization
        serializable_data = await self._prepare_data_for_serialization(data)

        # Serialize using appropriate method
        serializer = self._serializers.get(format)
        if not serializer:
            raise ValueError(f"Unsupported serialization format: {format}")

        start_time = datetime.now()
        await serializer(serializable_data, file_path, compress)

        # Create metadata
        file_size = file_path.stat().st_size if file_path.exists() else 0
        serialization_metadata = SerializationMetadata(
            format=format,
            version="1.0",
            timestamp=start_time,
            data_type=type(data).__name__,
            compression="gzip" if compress else None,
            size_bytes=file_size
        )

        # Save metadata
        metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
        await self._save_metadata(serialization_metadata, metadata_path, metadata)

        logger.info(f"Serialized {type(data).__name__} to {full_filename} ({file_size} bytes)")
        return serialization_metadata

    async def deserialize_data(
        self,
        filename: str,
        data_type: Type[T],
        format: Optional[str] = None
    ) -> T:
        """Deserialize data from file.

        Args:
            filename: Source filename (with or without extension)
            data_type: Expected type of the deserialized data
            format: Serialization format (auto-detected if None)

        Returns:
            Deserialized data object
        """
        # Find the file (try different extensions if needed)
        file_path = await self._find_data_file(filename)
        if not file_path:
            raise FileNotFoundError(f"Could not find data file: {filename}")

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(file_path)

        # Load metadata
        metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
        metadata = await self._load_metadata(metadata_path)

        # Deserialize using appropriate method
        deserializer = self._deserializers.get(format)
        if not deserializer:
            raise ValueError(f"Unsupported deserialization format: {format}")

        raw_data = await deserializer(file_path)

        # Convert back to proper data structures
        deserialized_data = await self._restore_data_from_serialization(raw_data, data_type)

        logger.info(f"Deserialized {data_type.__name__} from {file_path.name}")
        return deserialized_data

    async def serialize_simulation_state(
        self,
        state: SimulationState,
        filename: str,
        format: str = SerializationFormat.HDF5,
        incremental: bool = False
    ) -> SerializationMetadata:
        """Serialize complete simulation state.

        Args:
            state: Simulation state to serialize
            filename: Target filename
            format: Serialization format (HDF5 recommended for large states)
            incremental: Whether to do incremental save (append changes)

        Returns:
            SerializationMetadata
        """
        if incremental and format != SerializationFormat.HDF5:
            logger.warning("Incremental serialization only supported for HDF5 format")
            incremental = False

        # Prepare simulation state data
        state_data = {
            'timestep': state.timestep,
            'cells': {str(cid): asdict(cell_data) for cid, cell_data in state.cells.items()},
            'world_parameters': asdict(state.world_parameters),
            'global_metrics': state.global_metrics,
            'metadata': asdict(state.metadata) if state.metadata else None
        }

        metadata = {
            'cell_count': len(state.cells),
            'simulation_timestep': state.timestep,
            'incremental': incremental
        }

        return await self.serialize_data(state_data, filename, format, metadata=metadata)

    async def deserialize_simulation_state(
        self,
        filename: str,
        format: Optional[str] = None
    ) -> SimulationState:
        """Deserialize complete simulation state.

        Args:
            filename: Source filename
            format: Serialization format (auto-detected if None)

        Returns:
            SimulationState object
        """
        state_data = await self.deserialize_data(filename, dict, format)

        # Reconstruct simulation state
        cells = {}
        for cid_str, cell_dict in state_data['cells'].items():
            cell_id = CellID(int(cid_str))
            cell_data = CellData(
                cell_id=cell_id,
                position=Position(**cell_dict['position']),
                cell_type=CellType(cell_dict['cell_type']),
                cell_state=CellState(cell_dict['cell_state']),
                sort_value=cell_dict['sort_value'],
                age=cell_dict['age']
            )
            cells[cell_id] = cell_data

        # Reconstruct world parameters
        world_params = WorldParameters(**state_data['world_parameters'])

        # Reconstruct metadata if present
        metadata = None
        if state_data.get('metadata'):
            metadata = ExperimentMetadata(**state_data['metadata'])

        return SimulationState(
            timestep=SimulationTime(state_data['timestep']),
            cells=cells,
            world_parameters=world_params,
            global_metrics=state_data.get('global_metrics', {}),
            metadata=metadata
        )

    async def serialize_cell_trajectory(
        self,
        cell_id: CellID,
        trajectory: List[CellData],
        filename: str,
        format: str = SerializationFormat.HDF5
    ) -> SerializationMetadata:
        """Serialize cell trajectory data for analysis.

        Args:
            cell_id: ID of the cell
            trajectory: List of CellData over time
            filename: Target filename
            format: Serialization format

        Returns:
            SerializationMetadata
        """
        trajectory_data = {
            'cell_id': int(cell_id),
            'trajectory': [asdict(cell_data) for cell_data in trajectory],
            'timesteps': len(trajectory)
        }

        metadata = {
            'cell_id': int(cell_id),
            'trajectory_length': len(trajectory),
            'data_type': 'cell_trajectory'
        }

        return await self.serialize_data(trajectory_data, filename, format, metadata=metadata)

    async def serialize_experiment_batch(
        self,
        experiments: List[Dict[str, Any]],
        batch_filename: str,
        format: str = SerializationFormat.HDF5
    ) -> SerializationMetadata:
        """Serialize a batch of experiments for analysis.

        Args:
            experiments: List of experiment data dictionaries
            batch_filename: Target filename for the batch
            format: Serialization format

        Returns:
            SerializationMetadata
        """
        batch_data = {
            'experiment_count': len(experiments),
            'experiments': experiments,
            'batch_timestamp': datetime.now().isoformat()
        }

        metadata = {
            'experiment_count': len(experiments),
            'batch_type': 'experiment_batch'
        }

        return await self.serialize_data(batch_data, batch_filename, format, metadata=metadata)

    async def stream_simulation_data(
        self,
        data_source: AsyncGenerator[Dict[str, Any], None],
        filename: str,
        format: str = SerializationFormat.JSON,
        buffer_size: int = 1000
    ) -> SerializationMetadata:
        """Stream simulation data to file as it's generated.

        Args:
            data_source: Async generator of data records
            filename: Target filename
            format: Serialization format
            buffer_size: Number of records to buffer before writing

        Returns:
            SerializationMetadata
        """
        file_extension = self._get_file_extension(format, False)
        file_path = self.base_path / f"{filename}{file_extension}"

        record_count = 0
        buffer = []

        async with aiofiles.open(file_path, 'w') as f:
            # Write header for streaming format
            if format == SerializationFormat.JSON:
                await f.write('[\n')

            async for record in data_source:
                buffer.append(record)
                record_count += 1

                if len(buffer) >= buffer_size:
                    # Write buffer to file
                    await self._write_buffer_to_file(f, buffer, format, record_count > len(buffer))
                    buffer.clear()

            # Write remaining buffer
            if buffer:
                await self._write_buffer_to_file(f, buffer, format, record_count > len(buffer))

            # Write footer for streaming format
            if format == SerializationFormat.JSON:
                await f.write('\n]')

        file_size = file_path.stat().st_size
        metadata = SerializationMetadata(
            format=format,
            version="1.0",
            timestamp=datetime.now(),
            data_type="streaming_data",
            size_bytes=file_size
        )

        logger.info(f"Streamed {record_count} records to {filename} ({file_size} bytes)")
        return metadata

    # Private methods for format-specific serialization

    async def _serialize_json(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to JSON format."""
        json_data = json.dumps(data, indent=2, default=self._json_default)

        if compress:
            async with aiofiles.open(file_path, 'wb') as f:
                compressed_data = gzip.compress(json_data.encode('utf-8'))
                await f.write(compressed_data)
        else:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json_data)

    async def _deserialize_json(self, file_path: Path) -> Any:
        """Deserialize data from JSON format."""
        if file_path.suffix == '.gz':
            async with aiofiles.open(file_path, 'rb') as f:
                compressed_data = await f.read()
                json_data = gzip.decompress(compressed_data).decode('utf-8')
        else:
            async with aiofiles.open(file_path, 'r') as f:
                json_data = await f.read()

        return json.loads(json_data)

    async def _serialize_pickle(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to pickle format."""
        pickle_data = pickle.dumps(data)

        if compress:
            async with aiofiles.open(file_path, 'wb') as f:
                compressed_data = gzip.compress(pickle_data)
                await f.write(compressed_data)
        else:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pickle_data)

    async def _deserialize_pickle(self, file_path: Path) -> Any:
        """Deserialize data from pickle format."""
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()

        if file_path.suffix == '.gz':
            data = gzip.decompress(data)

        return pickle.loads(data)

    async def _serialize_compressed_pickle(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to compressed pickle format."""
        await self._serialize_pickle(data, file_path, True)

    async def _deserialize_compressed_pickle(self, file_path: Path) -> Any:
        """Deserialize data from compressed pickle format."""
        return await self._deserialize_pickle(file_path)

    async def _serialize_hdf5(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 serialization")

        # Use asyncio to run HDF5 operations in thread pool
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_hdf5, data, file_path, compress
        )

    def _write_hdf5(self, data: Any, file_path: Path, compress: bool) -> None:
        """Write data to HDF5 file (synchronous)."""
        compression = 'gzip' if compress else None

        with h5py.File(file_path, 'w') as f:
            self._write_dict_to_hdf5(f, data, compression)

    def _write_dict_to_hdf5(self, group: h5py.Group, data: Dict[str, Any], compression: Optional[str]) -> None:
        """Recursively write dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_hdf5(subgroup, value, compression)
            elif isinstance(value, (list, np.ndarray)):
                # Convert to numpy array
                array_data = np.array(value)
                group.create_dataset(key, data=array_data, compression=compression)
            else:
                # Store as scalar dataset
                group.create_dataset(key, data=value)

    async def _deserialize_hdf5(self, file_path: Path) -> Any:
        """Deserialize data from HDF5 format."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 deserialization")

        return await asyncio.get_event_loop().run_in_executor(
            None, self._read_hdf5, file_path
        )

    def _read_hdf5(self, file_path: Path) -> Dict[str, Any]:
        """Read data from HDF5 file (synchronous)."""
        with h5py.File(file_path, 'r') as f:
            return self._read_hdf5_group(f)

    def _read_hdf5_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Recursively read HDF5 group to dictionary."""
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = self._read_hdf5_group(item)
            else:
                # Dataset
                data = item[...]
                if data.shape == ():
                    # Scalar
                    result[key] = data.item()
                else:
                    # Array
                    result[key] = data.tolist()
        return result

    async def _serialize_custom_binary(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to custom binary format."""
        # This is a placeholder for a custom binary format
        # For now, use compressed pickle
        await self._serialize_pickle(data, file_path, True)

    async def _deserialize_custom_binary(self, file_path: Path) -> Any:
        """Deserialize data from custom binary format."""
        # This is a placeholder for a custom binary format
        # For now, use compressed pickle
        return await self._deserialize_pickle(file_path)

    async def _serialize_csv(self, data: Any, file_path: Path, compress: bool) -> None:
        """Serialize data to CSV format."""
        if not isinstance(data, (list, dict)):
            raise ValueError("CSV serialization requires list or dict data")

        # Convert to CSV-compatible format
        if isinstance(data, dict):
            # Assume dict contains tabular data
            if 'cells' in data:
                # Simulation state format
                csv_data = self._convert_simulation_to_csv(data)
            else:
                # Generic dict format
                csv_data = self._convert_dict_to_csv(data)
        else:
            # List format
            csv_data = self._convert_list_to_csv(data)

        if compress:
            async with aiofiles.open(file_path, 'wb') as f:
                compressed_data = gzip.compress(csv_data.encode('utf-8'))
                await f.write(compressed_data)
        else:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(csv_data)

    async def _deserialize_csv(self, file_path: Path) -> Any:
        """Deserialize data from CSV format."""
        if file_path.suffix == '.gz':
            async with aiofiles.open(file_path, 'rb') as f:
                compressed_data = await f.read()
                csv_data = gzip.decompress(compressed_data).decode('utf-8')
        else:
            async with aiofiles.open(file_path, 'r') as f:
                csv_data = await f.read()

        # Parse CSV data
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(csv_data))
        return [row for row in reader]

    # Utility methods

    def _get_file_extension(self, format: str, compress: bool) -> str:
        """Get file extension for format and compression."""
        extensions = {
            SerializationFormat.JSON: ".json",
            SerializationFormat.PICKLE: ".pkl",
            SerializationFormat.COMPRESSED_PICKLE: ".pkl.gz",
            SerializationFormat.HDF5: ".h5",
            SerializationFormat.CUSTOM_BINARY: ".bin",
            SerializationFormat.CSV: ".csv",
        }

        ext = extensions.get(format, ".dat")
        if compress and format not in [SerializationFormat.COMPRESSED_PICKLE]:
            ext += ".gz"

        return ext

    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect serialization format from file extension."""
        suffix = file_path.suffix.lower()

        if suffix == '.json':
            return SerializationFormat.JSON
        elif suffix in ['.pkl', '.pickle']:
            return SerializationFormat.PICKLE
        elif suffix == '.gz':
            # Check secondary extension
            secondary_suffix = file_path.with_suffix('').suffix.lower()
            if secondary_suffix in ['.pkl', '.pickle']:
                return SerializationFormat.COMPRESSED_PICKLE
            elif secondary_suffix == '.json':
                return SerializationFormat.JSON
            elif secondary_suffix == '.csv':
                return SerializationFormat.CSV
            else:
                return SerializationFormat.COMPRESSED_PICKLE
        elif suffix in ['.h5', '.hdf5']:
            return SerializationFormat.HDF5
        elif suffix == '.csv':
            return SerializationFormat.CSV
        elif suffix == '.bin':
            return SerializationFormat.CUSTOM_BINARY
        else:
            return SerializationFormat.JSON  # Default fallback

    async def _find_data_file(self, filename: str) -> Optional[Path]:
        """Find data file with various possible extensions."""
        base_name = Path(filename).stem

        # Try exact filename first
        exact_path = self.base_path / filename
        if exact_path.exists():
            return exact_path

        # Try with different extensions
        extensions = ['.json', '.pkl', '.h5', '.csv', '.bin', '.json.gz', '.pkl.gz', '.csv.gz']
        for ext in extensions:
            test_path = self.base_path / f"{base_name}{ext}"
            if test_path.exists():
                return test_path

        return None

    async def _prepare_data_for_serialization(self, data: Any) -> Any:
        """Prepare data for serialization by converting complex types."""
        if is_dataclass(data):
            return asdict(data)
        elif isinstance(data, dict):
            return {
                str(k) if not isinstance(k, str) else k: await self._prepare_data_for_serialization(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [await self._prepare_data_for_serialization(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return await self._prepare_data_for_serialization(data.__dict__)
        else:
            return data

    async def _restore_data_from_serialization(self, data: Any, target_type: Type[T]) -> T:
        """Restore data to proper types after deserialization."""
        if target_type == dict or not hasattr(target_type, '__annotations__'):
            return data

        # For dataclasses, reconstruct from dict
        if is_dataclass(target_type):
            if isinstance(data, dict):
                return target_type(**data)

        # For other types, attempt direct conversion
        try:
            return target_type(data)
        except (TypeError, ValueError):
            return data

    def _json_default(self, obj: Any) -> Any:
        """Default serialization for JSON encoder."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    async def _save_metadata(
        self,
        serialization_metadata: SerializationMetadata,
        metadata_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save serialization metadata to file."""
        metadata_dict = asdict(serialization_metadata)
        if additional_metadata:
            metadata_dict.update(additional_metadata)

        metadata_json = json.dumps(metadata_dict, indent=2, default=self._json_default)
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(metadata_json)

    async def _load_metadata(self, metadata_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata from file."""
        if not metadata_path.exists():
            return None

        try:
            async with aiofiles.open(metadata_path, 'r') as f:
                metadata_json = await f.read()
            return json.loads(metadata_json)
        except Exception as e:
            logger.warning(f"Could not load metadata from {metadata_path}: {e}")
            return None

    async def _write_buffer_to_file(
        self,
        file_handle: Any,
        buffer: List[Any],
        format: str,
        has_previous_records: bool
    ) -> None:
        """Write buffer to streaming file."""
        if format == SerializationFormat.JSON:
            for i, record in enumerate(buffer):
                if has_previous_records or i > 0:
                    await file_handle.write(',\n')
                json_record = json.dumps(record, default=self._json_default)
                await file_handle.write(f'  {json_record}')
        # Add other streaming formats as needed

    def _convert_simulation_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert simulation data to CSV format."""
        # Extract cell data for CSV
        csv_lines = []

        # Header
        headers = ['cell_id', 'position_x', 'position_y', 'cell_type', 'cell_state', 'sort_value', 'age', 'timestep']
        csv_lines.append(','.join(headers))

        # Data rows
        timestep = data.get('timestep', 0)
        for cell_id, cell_data in data.get('cells', {}).items():
            row = [
                cell_id,
                str(cell_data['position']['x']),
                str(cell_data['position']['y']),
                cell_data['cell_type'],
                cell_data['cell_state'],
                str(cell_data['sort_value']),
                str(cell_data['age']),
                str(timestep)
            ]
            csv_lines.append(','.join(row))

        return '\n'.join(csv_lines)

    def _convert_dict_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert generic dictionary to CSV format."""
        # Simple key-value CSV
        csv_lines = ['key,value']
        for key, value in data.items():
            csv_lines.append(f"{key},{value}")
        return '\n'.join(csv_lines)

    def _convert_list_to_csv(self, data: List[Any]) -> str:
        """Convert list to CSV format."""
        if not data:
            return ""

        # Assume list of dictionaries
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            csv_lines = [','.join(headers)]

            for row in data:
                values = [str(row.get(header, '')) for header in headers]
                csv_lines.append(','.join(values))

            return '\n'.join(csv_lines)
        else:
            # Simple list
            return '\n'.join(str(item) for item in data)