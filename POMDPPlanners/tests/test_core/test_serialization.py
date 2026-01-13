"""Tests for centralized serialization utilities.

This module tests the serialization functionality in POMDPPlanners.core.serialization,
ensuring proper round-trip serialization for all supported types and edge cases.
"""

import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from POMDPPlanners.core.serialization import (
    deserialize_stateful_object,
    deserialize_value,
    extract_constructor_params,
    load_from_json,
    reconstruct_from_params,
    register_deserializer,
    register_serializer,
    save_to_json,
    serialize_stateful_object,
    serialize_value,
)


# Test helper classes at module level for importability
class ReconstructTestClass:
    """Test class for reconstruction tests."""

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ReconstructTestClassWithPath:
    """Test class with Path parameter for reconstruction tests."""

    def __init__(self, path: Path):
        self.path = path


class StatefulTestClass:
    """Test class with __getstate__/__setstate__ for serialization tests."""

    def __init__(self, value=None):
        self.value = value

    def __getstate__(self):
        return {"value": self.value}

    def __setstate__(self, state):
        self.value = state["value"]


class TestSerializeValue:
    """Tests for serialize_value function."""

    def test_serialize_none(self):
        """Test serialization of None value.

        Purpose: Validates that None is serialized correctly

        Given: None value
        When: serialize_value is called
        Then: None is returned

        Test type: unit
        """
        result = serialize_value(None)
        assert result is None

    def test_serialize_primitives(self):
        """Test serialization of primitive types.

        Purpose: Validates that primitive types are serialized as-is

        Given: String, int, float, and bool values
        When: serialize_value is called on each
        Then: Values are returned unchanged

        Test type: unit
        """
        assert serialize_value("test") == "test"
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value(True) is True
        assert serialize_value(False) is False

    def test_serialize_path(self):
        """Test serialization of Path objects.

        Purpose: Validates that Path objects are serialized to dict with marker

        Given: Path object
        When: serialize_value is called
        Then: Dictionary with __type__ marker and string value is returned

        Test type: unit
        """
        path = Path("/tmp/test.txt")
        result = serialize_value(path)

        assert isinstance(result, dict)
        assert result["__type__"] == "Path"
        assert result["value"] == "/tmp/test.txt"

    def test_serialize_numpy_array(self):
        """Test serialization of NumPy arrays.

        Purpose: Validates that NumPy arrays are serialized with type marker and dtype

        Given: NumPy array
        When: serialize_value is called
        Then: Dictionary with __type__, value list, and dtype is returned

        Test type: unit
        """
        arr = np.array([1, 2, 3])
        result = serialize_value(arr)

        assert isinstance(result, dict)
        assert result["__type__"] == "ndarray"
        assert result["value"] == [1, 2, 3]
        assert "dtype" in result

    def test_serialize_numpy_scalar(self):
        """Test serialization of NumPy scalar types.

        Purpose: Validates that NumPy scalars are converted to Python primitives

        Given: NumPy integer and floating types
        When: serialize_value is called
        Then: Python int or float is returned

        Test type: unit
        """
        np_int = np.int64(42)
        np_float = np.float64(3.14)

        assert serialize_value(np_int) == 42
        assert isinstance(serialize_value(np_int), int)
        assert serialize_value(np_float) == 3.14
        assert isinstance(serialize_value(np_float), float)

    def test_serialize_enum(self):
        """Test serialization of Enum types.

        Purpose: Validates that Enum values are serialized with class info

        Given: Enum value
        When: serialize_value is called
        Then: Dictionary with __type__, module, class, and value is returned

        Test type: unit
        """

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = serialize_value(Color.RED)

        assert isinstance(result, dict)
        assert result["__type__"] == "Enum"
        assert result["value"] == "red"
        assert "module" in result
        assert "class" in result

    def test_serialize_list(self):
        """Test serialization of lists.

        Purpose: Validates that lists are serialized recursively

        Given: List with mixed types
        When: serialize_value is called
        Then: List with serialized elements is returned

        Test type: unit
        """
        lst = [1, "test", Path("/tmp"), None]
        result = serialize_value(lst)

        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == "test"
        assert isinstance(result[2], dict)  # Path serialized to dict
        assert result[3] is None

    def test_serialize_tuple(self):
        """Test serialization of tuples.

        Purpose: Validates that tuples are serialized with type marker

        Given: Tuple with mixed types
        When: serialize_value is called
        Then: Dictionary with __type__ marker and values list is returned

        Test type: unit
        """
        tpl = (1, "test", 3.14)
        result = serialize_value(tpl)

        assert isinstance(result, dict)
        assert result["__type__"] == "tuple"
        assert result["values"] == [1, "test", 3.14]

    def test_serialize_set(self):
        """Test serialization of sets.

        Purpose: Validates that sets are serialized with type marker

        Given: Set with values
        When: serialize_value is called
        Then: Dictionary with __type__ marker and values list is returned

        Test type: unit
        """
        s = {1, 2, 3}
        result = serialize_value(s)

        assert isinstance(result, dict)
        assert result["__type__"] == "set"
        assert set(result["values"]) == {1, 2, 3}

    def test_serialize_dict(self):
        """Test serialization of dictionaries.

        Purpose: Validates that dicts are serialized recursively

        Given: Dictionary with mixed value types
        When: serialize_value is called
        Then: Dictionary with serialized values is returned

        Test type: unit
        """
        d = {"a": 1, "b": Path("/tmp"), "c": None}
        result = serialize_value(d)

        assert isinstance(result, dict)
        assert result["a"] == 1
        assert isinstance(result["b"], dict)  # Path serialized
        assert result["c"] is None

    def test_serialize_logger_skipped(self):
        """Test that loggers are skipped during serialization.

        Purpose: Validates that Logger objects return None

        Given: Logger instance
        When: serialize_value is called
        Then: None is returned

        Test type: unit
        """
        logger = logging.getLogger("test")
        result = serialize_value(logger)
        assert result is None

    def test_serialize_unknown_type_warns(self):
        """Test that unknown types produce warning and fallback.

        Purpose: Validates fallback behavior for unsupported types

        Given: Custom object with no serializer
        When: serialize_value is called
        Then: Warning is raised and str_fallback dict is returned

        Test type: unit
        """

        class CustomObject:
            def __str__(self):
                return "custom"

        obj = CustomObject()

        with pytest.warns(UserWarning, match="No serializer for type"):
            result = serialize_value(obj)

        assert isinstance(result, dict)
        assert result["__type__"] == "str_fallback"
        assert result["value"] == "custom"


class TestDeserializeValue:
    """Tests for deserialize_value function."""

    def test_deserialize_none(self):
        """Test deserialization of None value.

        Purpose: Validates that None is deserialized correctly

        Given: None value
        When: deserialize_value is called
        Then: None is returned

        Test type: unit
        """
        result = deserialize_value(None)
        assert result is None

    def test_deserialize_primitives(self):
        """Test deserialization of primitive types.

        Purpose: Validates that primitives are returned as-is

        Given: Primitive values
        When: deserialize_value is called
        Then: Values are returned unchanged

        Test type: unit
        """
        assert deserialize_value("test") == "test"
        assert deserialize_value(42) == 42
        assert deserialize_value(3.14) == 3.14
        assert deserialize_value(True) is True

    def test_deserialize_path(self):
        """Test deserialization of Path objects.

        Purpose: Validates that Path dict marker is converted to Path object

        Given: Dictionary with __type__ Path marker
        When: deserialize_value is called
        Then: Path object is returned

        Test type: unit
        """
        data = {"__type__": "Path", "value": "/tmp/test.txt"}
        result = deserialize_value(data)

        assert isinstance(result, Path)
        assert str(result) == "/tmp/test.txt"

    def test_deserialize_numpy_array(self):
        """Test deserialization of NumPy arrays.

        Purpose: Validates that ndarray dict marker is converted to NumPy array

        Given: Dictionary with __type__ ndarray marker
        When: deserialize_value is called
        Then: NumPy array is returned with correct dtype

        Test type: unit
        """
        data = {"__type__": "ndarray", "value": [1, 2, 3], "dtype": "int64"}
        result = deserialize_value(data)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))
        assert result.dtype == np.int64

    def test_deserialize_enum(self):
        """Test deserialization of Enum types.

        Purpose: Validates that Enum dict marker is converted to Enum value

        Given: Dictionary with __type__ Enum marker and class info
        When: deserialize_value is called
        Then: Enum value is returned

        Test type: unit
        """
        # Use SpaceType from core.environment which is a real importable Enum
        from POMDPPlanners.core.environment import SpaceType

        data = {
            "__type__": "Enum",
            "module": SpaceType.__module__,
            "class": "SpaceType",
            "value": "discrete",  # Use actual enum value (lowercase)
        }
        result = deserialize_value(data)

        assert result == SpaceType.DISCRETE
        assert isinstance(result, SpaceType)

    def test_deserialize_set(self):
        """Test deserialization of sets.

        Purpose: Validates that set dict marker is converted to set

        Given: Dictionary with __type__ set marker
        When: deserialize_value is called
        Then: Set is returned

        Test type: unit
        """
        data = {"__type__": "set", "values": [1, 2, 3]}
        result = deserialize_value(data)

        assert isinstance(result, set)
        assert result == {1, 2, 3}

    def test_deserialize_tuple(self):
        """Test deserialization of tuples.

        Purpose: Validates that tuple dict marker is converted to tuple

        Given: Dictionary with __type__ tuple marker
        When: deserialize_value is called
        Then: Tuple is returned

        Test type: unit
        """
        data = {"__type__": "tuple", "values": [1, "test", 3.14]}
        result = deserialize_value(data)

        assert isinstance(result, tuple)
        assert result == (1, "test", 3.14)

    def test_deserialize_with_target_type_path(self):
        """Test deserialization with Path target type hint.

        Purpose: Validates that string is converted to Path when target_type is Path

        Given: String value and target_type=Path
        When: deserialize_value is called
        Then: Path object is returned

        Test type: unit
        """
        result = deserialize_value("/tmp/test", target_type=Path)

        assert isinstance(result, Path)
        assert str(result) == "/tmp/test"

    def test_deserialize_with_optional_type(self):
        """Test deserialization with Optional type hint.

        Purpose: Validates that Optional[T] type hints are handled

        Given: Value and target_type=Optional[Path]
        When: deserialize_value is called
        Then: Inner type deserialization is applied

        Test type: unit
        """
        result = deserialize_value("/tmp/test", target_type=Optional[Path])

        assert isinstance(result, Path)
        assert str(result) == "/tmp/test"

    def test_deserialize_str_fallback(self):
        """Test deserialization of str_fallback marker.

        Purpose: Validates that str_fallback is converted to string

        Given: Dictionary with __type__ str_fallback marker
        When: deserialize_value is called
        Then: String value is returned

        Test type: unit
        """
        data = {"__type__": "str_fallback", "value": "custom_string"}
        result = deserialize_value(data)

        assert result == "custom_string"


class TestRoundTripSerialization:
    """Tests for round-trip serialization (serialize then deserialize)."""

    def test_roundtrip_path(self):
        """Test round-trip serialization of Path.

        Purpose: Validates that Path survives serialize→deserialize cycle

        Given: Path object
        When: Serialized then deserialized
        Then: Original Path is reconstructed

        Test type: unit
        """
        original = Path("/tmp/test/file.txt")
        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        assert deserialized == original
        assert isinstance(deserialized, Path)

    def test_roundtrip_numpy_array(self):
        """Test round-trip serialization of NumPy array.

        Purpose: Validates that NumPy array survives serialize→deserialize cycle

        Given: NumPy array
        When: Serialized then deserialized
        Then: Original array is reconstructed

        Test type: unit
        """
        original = np.array([[1, 2], [3, 4]], dtype=np.float32)
        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        assert np.array_equal(deserialized, original)
        assert deserialized.dtype == original.dtype

    def test_roundtrip_complex_nested_structure(self):
        """Test round-trip serialization of complex nested structure.

        Purpose: Validates complex nested data structures survive round-trip

        Given: Dictionary with mixed types including nested collections
        When: Serialized then deserialized
        Then: Original structure is reconstructed (with type markers for complex types)

        Test type: unit
        """
        original = {
            "path": Path("/tmp/test"),
            "array": np.array([1, 2, 3]),
            "tuple": (1, 2, "test"),
            "set": {4, 5, 6},
            "nested": {"a": 1, "b": [2, 3, 4]},
        }

        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        # For dict values, we get serialized format unless we have type hints
        # So we need to explicitly deserialize them
        assert deserialize_value(deserialized["path"]) == original["path"]
        assert np.array_equal(deserialize_value(deserialized["array"]), original["array"])
        assert deserialize_value(deserialized["tuple"]) == original["tuple"]
        assert deserialize_value(deserialized["set"]) == original["set"]
        assert deserialized["nested"] == original["nested"]


class TestCustomHandlers:
    """Tests for custom serialization handler registration."""

    def test_register_custom_serializer(self):
        """Test registering custom serializer.

        Purpose: Validates that custom serializers can be registered and used

        Given: Custom class and registered serializer
        When: serialize_value is called on custom object
        Then: Custom serializer is used

        Test type: unit
        """

        class CustomType:
            def __init__(self, value):
                self.value = value

        def custom_serializer(obj):
            return {"custom": obj.value}

        register_serializer(CustomType, custom_serializer)

        obj = CustomType("test")
        result = serialize_value(obj)

        assert result == {"custom": "test"}

    def test_register_custom_deserializer(self):
        """Test registering custom deserializer.

        Purpose: Validates that custom deserializers can be registered and used

        Given: Custom type and registered deserializer
        When: deserialize_value is called with target_type
        Then: Custom deserializer is used

        Test type: unit
        """

        class CustomType:
            def __init__(self, value):
                self.value = value

        def custom_deserializer(data):
            return CustomType(data["custom"])

        register_deserializer(CustomType, custom_deserializer)

        data = {"custom": "test"}
        result = deserialize_value(data, target_type=CustomType)

        assert isinstance(result, CustomType)
        assert result.value == "test"


class TestExtractConstructorParams:
    """Tests for extract_constructor_params function."""

    def test_extract_simple_params(self):
        """Test extracting parameters from simple class.

        Purpose: Validates parameter extraction from single-level class

        Given: Object with simple constructor parameters
        When: extract_constructor_params is called
        Then: All parameters (except excluded) are extracted

        Test type: unit
        """

        class SimpleClass:
            def __init__(self, a, b, c=10):
                self.a = a
                self.b = b
                self.c = c

        obj = SimpleClass(1, 2, 3)
        params = extract_constructor_params(obj)

        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert "self" not in params
        assert params["a"] == 1
        assert params["b"] == 2
        assert params["c"] == 3

    def test_extract_params_with_inheritance(self):
        """Test extracting parameters from class hierarchy.

        Purpose: Validates parameter extraction walks inheritance chain

        Given: Object with multi-level inheritance
        When: extract_constructor_params is called
        Then: Parameters from all levels are extracted

        Test type: unit
        """

        class BaseClass:
            def __init__(self, base_param):
                self.base_param = base_param

        class DerivedClass(BaseClass):
            def __init__(self, base_param, derived_param):
                super().__init__(base_param)
                self.derived_param = derived_param

        obj = DerivedClass("base", "derived")
        params = extract_constructor_params(obj)

        assert "base_param" in params
        assert "derived_param" in params
        assert params["base_param"] == "base"
        assert params["derived_param"] == "derived"

    def test_extract_params_with_exclude(self):
        """Test extracting parameters with exclusions.

        Purpose: Validates that excluded parameters are not extracted

        Given: Object and exclude list
        When: extract_constructor_params is called
        Then: Excluded parameters are not in result

        Test type: unit
        """

        class TestClass:
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

        obj = TestClass(1, 2, 3)
        params = extract_constructor_params(obj, exclude=("self", "b"))

        assert "a" in params
        assert "b" not in params
        assert "c" in params

    def test_extract_params_serializes_values(self):
        """Test that extracted parameters are serialized.

        Purpose: Validates that parameter values are serialized automatically

        Given: Object with Path parameter
        When: extract_constructor_params is called
        Then: Path is serialized to dict

        Test type: unit
        """

        class TestClass:
            def __init__(self, path):
                self.path = path

        obj = TestClass(Path("/tmp/test"))
        params = extract_constructor_params(obj)

        assert "path" in params
        assert isinstance(params["path"], dict)
        assert params["path"]["__type__"] == "Path"


class TestReconstructFromParams:
    """Tests for reconstruct_from_params function."""

    def test_reconstruct_simple_object(self):
        """Test reconstructing simple object.

        Purpose: Validates object reconstruction from class path and params

        Given: Class path and serialized parameters
        When: reconstruct_from_params is called
        Then: Object is reconstructed with correct parameters

        Test type: unit
        """
        class_path = f"{__name__}.ReconstructTestClass"
        params = {"a": 1, "b": 2}

        result = reconstruct_from_params(class_path, params)

        assert isinstance(result, ReconstructTestClass)
        assert result.a == 1
        assert result.b == 2

    def test_reconstruct_with_type_hints(self):
        """Test reconstruction with type annotations.

        Purpose: Validates that type hints guide deserialization

        Given: Class with Path type hint and serialized Path param
        When: reconstruct_from_params is called
        Then: String is deserialized to Path object

        Test type: unit
        """
        class_path = f"{__name__}.ReconstructTestClassWithPath"
        params = {"path": "/tmp/test"}

        result = reconstruct_from_params(class_path, params)

        assert isinstance(result, ReconstructTestClassWithPath)
        assert isinstance(result.path, Path)
        assert str(result.path) == "/tmp/test"


class TestStatefulObjectSerialization:
    """Tests for stateful object serialization using __getstate__/__setstate__."""

    def test_serialize_stateful_object(self):
        """Test serializing object with __getstate__.

        Purpose: Validates that objects with __getstate__ are serialized correctly

        Given: Object with __getstate__ method
        When: serialize_stateful_object is called
        Then: Dictionary with class info and state is returned

        Test type: unit
        """
        obj = StatefulTestClass(42)
        result = serialize_stateful_object(obj)

        assert "class" in result
        assert "module" in result
        assert "state" in result
        assert result["state"]["value"] == 42

    def test_deserialize_stateful_object(self):
        """Test deserializing object with __setstate__.

        Purpose: Validates that objects with __setstate__ are deserialized correctly

        Given: Serialized object data with state
        When: deserialize_stateful_object is called
        Then: Object is reconstructed using __setstate__

        Test type: unit
        """
        data = {
            "class": f"{__name__}.StatefulTestClass",
            "module": __name__,
            "state": {"value": 42},
        }

        result = deserialize_stateful_object(data)

        assert isinstance(result, StatefulTestClass)
        assert result.value == 42

    def test_serialize_stateful_object_no_getstate_raises(self):
        """Test error when object has no __getstate__.

        Purpose: Validates proper error handling for objects without __getstate__

        Given: Object without custom __getstate__ method
        When: serialize_stateful_object is called
        Then: ValueError is raised

        Test type: unit
        """

        class NonStatefulClass:
            pass

        obj = NonStatefulClass()

        with pytest.raises(ValueError, match="has no custom __getstate__ method"):
            serialize_stateful_object(obj)


class TestJsonFileOperations:
    """Tests for save_to_json and load_from_json functions."""

    def test_save_and_load_json(self):
        """Test saving and loading JSON file.

        Purpose: Validates round-trip JSON file operations

        Given: Data dictionary
        When: Saved to JSON then loaded
        Then: Original data is reconstructed

        Test type: unit
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            data = {"key": "value", "number": 42}
            save_to_json(temp_path, data)

            assert temp_path.exists()

            loaded = load_from_json(temp_path)
            assert loaded["key"] == "value"
            assert loaded["number"] == 42
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_json_with_metadata(self):
        """Test saving JSON with metadata.

        Purpose: Validates that metadata is included in saved file

        Given: Data and metadata dictionaries
        When: save_to_json is called with metadata
        Then: Both data and metadata are in saved file

        Test type: unit
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            data = {"key": "value"}
            metadata = {"version": "1.0", "timestamp": "2026-01-13"}

            save_to_json(temp_path, data, metadata=metadata)

            loaded = load_from_json(temp_path)

            assert "metadata" in loaded
            assert loaded["metadata"]["version"] == "1.0"
            assert loaded["key"] == "value"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_json_creates_directories(self):
        """Test that save_to_json creates parent directories.

        Purpose: Validates automatic directory creation

        Given: Path with non-existent parent directories
        When: save_to_json is called
        Then: Directories are created and file is saved

        Test type: unit
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "subdir1" / "subdir2" / "test.json"

            data = {"key": "value"}
            save_to_json(temp_path, data)

            assert temp_path.exists()
            assert temp_path.parent.exists()

    def test_load_json_nonexistent_file_raises(self):
        """Test error when loading non-existent file.

        Purpose: Validates proper error handling for missing files

        Given: Path to non-existent file
        When: load_from_json is called
        Then: FileNotFoundError is raised

        Test type: unit
        """
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_from_json("/nonexistent/path/file.json")

    def test_load_json_invalid_json_raises(self):
        """Test error when loading invalid JSON.

        Purpose: Validates proper error handling for corrupted JSON

        Given: File with invalid JSON content
        When: load_from_json is called
        Then: ValueError is raised

        Test type: unit
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
            f.write("{ invalid json content }")

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_from_json(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_json_with_complex_types(self):
        """Test saving JSON with complex Python types.

        Purpose: Validates that complex types are handled via _json_default

        Given: Data with Path, NumPy array, and other complex types
        When: save_to_json is called
        Then: File is saved without errors

        Test type: unit
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            data = {
                "path": Path("/tmp/test"),
                "array": np.array([1, 2, 3]),
                "tuple": (1, 2, 3),
            }

            save_to_json(temp_path, data)

            assert temp_path.exists()

            loaded = load_from_json(temp_path)
            assert "path" in loaded
            assert "array" in loaded
        finally:
            if temp_path.exists():
                temp_path.unlink()
