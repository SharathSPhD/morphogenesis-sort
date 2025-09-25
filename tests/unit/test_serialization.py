"""Unit tests for data serialization module."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path

from core.data.serialization import AsyncDataSerializer, SerializationFormat, SerializationMetadata
from core.data.state import SimulationState
from core.data.types import CellID


class TestAsyncDataSerializer:
    """Test AsyncDataSerializer functionality."""

    @pytest.fixture
    def serializer(self, temp_data_dir):
        """Create a serializer with temporary directory."""
        return AsyncDataSerializer(base_path=temp_data_dir)

    @pytest.mark.asyncio
    async def test_json_serialization(self, serializer):
        """Test JSON serialization and deserialization."""
        test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        filename = "test_json"

        # Serialize
        metadata = await serializer.serialize_data(
            test_data, filename, SerializationFormat.JSON
        )

        assert metadata.format == SerializationFormat.JSON
        assert metadata.data_type == "dict"

        # Deserialize
        loaded_data = await serializer.deserialize_data(
            filename, dict, SerializationFormat.JSON
        )

        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_pickle_serialization(self, serializer):
        """Test Pickle serialization and deserialization."""
        test_data = {"complex_key": set([1, 2, 3]), "tuple": (1, 2, 3)}
        filename = "test_pickle"

        # Serialize
        metadata = await serializer.serialize_data(
            test_data, filename, SerializationFormat.PICKLE
        )

        assert metadata.format == SerializationFormat.PICKLE

        # Deserialize
        loaded_data = await serializer.deserialize_data(
            filename, dict, SerializationFormat.PICKLE
        )

        # Convert set back for comparison (sets aren't guaranteed order)
        assert isinstance(loaded_data["complex_key"], set)
        assert loaded_data["complex_key"] == test_data["complex_key"]
        assert loaded_data["tuple"] == test_data["tuple"]

    @pytest.mark.asyncio
    async def test_compressed_serialization(self, serializer):
        """Test compressed serialization."""
        test_data = {"large_data": "x" * 1000}  # Some data that benefits from compression
        filename = "test_compressed"

        # Serialize with compression
        metadata = await serializer.serialize_data(
            test_data, filename, SerializationFormat.JSON, compress=True
        )

        assert metadata.compression == "gzip"

        # Deserialize
        loaded_data = await serializer.deserialize_data(
            filename, dict, SerializationFormat.JSON
        )

        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_simulation_state_serialization(self, serializer, sample_simulation_state):
        """Test simulation state serialization."""
        filename = "test_simulation_state"

        # Serialize
        metadata = await serializer.serialize_simulation_state(
            sample_simulation_state, filename, SerializationFormat.JSON
        )

        assert metadata.format == SerializationFormat.JSON
        assert isinstance(metadata.timestamp, type(metadata.timestamp))

        # Deserialize
        loaded_state = await serializer.deserialize_simulation_state(
            filename, SerializationFormat.JSON
        )

        assert loaded_state.timestep == sample_simulation_state.timestep
        assert len(loaded_state.cells) == len(sample_simulation_state.cells)
        assert loaded_state.world_parameters.width == sample_simulation_state.world_parameters.width

    @pytest.mark.asyncio
    async def test_cell_trajectory_serialization(self, serializer, sample_cells):
        """Test cell trajectory serialization."""
        cell_id = CellID(1)
        trajectory = list(sample_cells.values())[:3]  # Use first 3 cells as trajectory
        filename = "test_trajectory"

        # Serialize
        metadata = await serializer.serialize_cell_trajectory(
            cell_id, trajectory, filename
        )

        assert metadata.format == SerializationFormat.HDF5  # Default format

        # Check that file was created
        expected_path = serializer.base_path / "test_trajectory.h5"
        assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_format_detection(self, serializer):
        """Test automatic format detection."""
        # Create files with different extensions
        json_path = serializer.base_path / "test.json"
        pickle_path = serializer.base_path / "test.pkl"

        json_path.write_text('{"test": "data"}')
        pickle_path.write_bytes(b"pickle_data")

        # Test detection
        json_format = serializer._detect_format(json_path)
        pickle_format = serializer._detect_format(pickle_path)

        assert json_format == SerializationFormat.JSON
        assert pickle_format == SerializationFormat.PICKLE

    @pytest.mark.asyncio
    async def test_file_finding(self, serializer):
        """Test finding files with different extensions."""
        # Create a test file
        test_path = serializer.base_path / "test_file.json"
        test_path.write_text('{"test": "data"}')

        # Should find file with exact name
        found_path = await serializer._find_data_file("test_file.json")
        assert found_path == test_path

        # Should find file with base name
        found_path = await serializer._find_data_file("test_file")
        assert found_path == test_path

        # Should return None for non-existent file
        not_found = await serializer._find_data_file("non_existent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_csv_serialization(self, serializer):
        """Test CSV serialization."""
        # Test with simple list of dictionaries
        test_data = [
            {"id": 1, "name": "cell1", "value": 10.5},
            {"id": 2, "name": "cell2", "value": 20.3},
            {"id": 3, "name": "cell3", "value": 15.7}
        ]
        filename = "test_csv"

        # Serialize
        metadata = await serializer.serialize_data(
            test_data, filename, SerializationFormat.CSV
        )

        assert metadata.format == SerializationFormat.CSV

        # Check that file was created
        expected_path = serializer.base_path / "test_csv.csv"
        assert expected_path.exists()

        # Verify CSV content
        csv_content = expected_path.read_text()
        assert "id,name,value" in csv_content  # Header
        assert "1,cell1,10.5" in csv_content   # First row

    @pytest.mark.asyncio
    async def test_metadata_persistence(self, serializer):
        """Test that metadata is saved and loaded correctly."""
        test_data = {"test": "data"}
        filename = "test_metadata"
        additional_metadata = {"experiment_id": "exp_001", "researcher": "test_user"}

        # Serialize with additional metadata
        metadata = await serializer.serialize_data(
            test_data, filename, SerializationFormat.JSON,
            metadata=additional_metadata
        )

        # Check metadata file exists
        metadata_path = serializer.base_path / f"{filename}.json.meta"
        assert metadata_path.exists()

        # Load and verify metadata
        loaded_metadata = await serializer._load_metadata(metadata_path)
        assert loaded_metadata is not None
        assert loaded_metadata["experiment_id"] == "exp_001"
        assert loaded_metadata["researcher"] == "test_user"
        assert loaded_metadata["format"] == SerializationFormat.JSON

    @pytest.mark.asyncio
    async def test_data_preparation(self, serializer):
        """Test data preparation for serialization."""
        # Test with dataclass
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            name: str
            value: int

        test_obj = TestClass("test", 42)
        prepared = await serializer._prepare_data_for_serialization(test_obj)

        assert isinstance(prepared, dict)
        assert prepared["name"] == "test"
        assert prepared["value"] == 42

    @pytest.mark.asyncio
    async def test_error_handling(self, serializer):
        """Test error handling for various scenarios."""
        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported serialization format"):
            await serializer.serialize_data({}, "test", "unsupported_format")

        # Test file not found
        with pytest.raises(FileNotFoundError):
            await serializer.deserialize_data("non_existent_file", dict)

    @pytest.mark.asyncio
    async def test_streaming_serialization(self, serializer):
        """Test streaming data serialization."""
        async def data_generator():
            """Generate test data records."""
            for i in range(10):
                yield {"record_id": i, "value": i * 2, "timestamp": i * 100}

        filename = "test_streaming"

        # Serialize streaming data
        metadata = await serializer.stream_simulation_data(
            data_generator(), filename, SerializationFormat.JSON, buffer_size=3
        )

        assert metadata.format == SerializationFormat.JSON
        assert metadata.data_type == "streaming_data"

        # Verify file content
        expected_path = serializer.base_path / f"{filename}.json"
        assert expected_path.exists()

        content = expected_path.read_text()
        assert content.startswith('[')
        assert content.endswith(']')
        assert '"record_id": 0' in content
        assert '"record_id": 9' in content


class TestSerializationFormats:
    """Test different serialization formats."""

    def test_file_extensions(self):
        """Test file extension generation."""
        serializer = AsyncDataSerializer()

        # Test basic extensions
        assert serializer._get_file_extension(SerializationFormat.JSON, False) == ".json"
        assert serializer._get_file_extension(SerializationFormat.PICKLE, False) == ".pkl"
        assert serializer._get_file_extension(SerializationFormat.HDF5, False) == ".h5"
        assert serializer._get_file_extension(SerializationFormat.CSV, False) == ".csv"

        # Test compressed extensions
        assert serializer._get_file_extension(SerializationFormat.JSON, True) == ".json.gz"
        assert serializer._get_file_extension(SerializationFormat.CSV, True) == ".csv.gz"

        # Test already compressed format
        assert serializer._get_file_extension(SerializationFormat.COMPRESSED_PICKLE, False) == ".pkl.gz"

    def test_format_detection_edge_cases(self):
        """Test format detection for edge cases."""
        serializer = AsyncDataSerializer()

        # Test compressed JSON
        gz_json_path = Path("test.json.gz")
        assert serializer._detect_format(gz_json_path) == SerializationFormat.JSON

        # Test compressed pickle
        gz_pickle_path = Path("test.pkl.gz")
        assert serializer._detect_format(gz_pickle_path) == SerializationFormat.COMPRESSED_PICKLE

        # Test unknown extension
        unknown_path = Path("test.xyz")
        assert serializer._detect_format(unknown_path) == SerializationFormat.JSON  # Default fallback