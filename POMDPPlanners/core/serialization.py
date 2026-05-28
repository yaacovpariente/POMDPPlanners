# SPDX-License-Identifier: MIT

"""Centralized serialization utilities for POMDPPlanners.

This module provides reusable serialization functions for converting Python
objects to/from JSON-compatible formats. Used by Environment, Policy, Belief,
and other components for configuration persistence.

Key Functions:
    serialize_value: Convert Python value to JSON-compatible format
    deserialize_value: Convert JSON format back to Python value
    extract_constructor_params: Extract constructor parameters from object
    reconstruct_from_params: Reconstruct object from class path and parameters
    save_to_json: Save data to JSON file with metadata
    load_from_json: Load data from JSON file
    serialize_stateful_object: Serialize using __getstate__ protocol
    deserialize_stateful_object: Deserialize using __setstate__ protocol

Example:
    >>> from pathlib import Path
    >>> import numpy as np
    >>>
    >>> # Serialize various types
    >>> serialize_value(Path("/tmp/test"))
    {'__type__': 'Path', 'value': '/tmp/test'}
    >>>
    >>> serialize_value(np.array([1, 2, 3]))
    {'__type__': 'ndarray', 'value': [1, 2, 3], 'dtype': 'int64'}
    >>>
    >>> # Deserialize back to original types
    >>> path_data = {'__type__': 'Path', 'value': '/tmp/test'}
    >>> deserialize_value(path_data)
    PosixPath('/tmp/test')
"""

import importlib
import inspect
import json
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np

# Type handler registry for extensibility
_SERIALIZE_HANDLERS: Dict[Type, Callable[[Any], Any]] = {}
_DESERIALIZE_HANDLERS: Dict[Type, Callable[[Any], Any]] = {}


def serialize_value(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    """Serialize Python value to JSON-compatible format.

    Handles common types used in POMDPPlanners:
    - Primitives: str, int, float, bool, None
    - Collections: list, tuple, set, dict
    - NumPy: ndarray, integer, floating
    - Path objects
    - Enums
    - Loggers (skipped)
    - Custom types with registered handlers

    Args:
        value: Value to serialize

    Returns:
        JSON-compatible representation of the value
    """
    # Check custom handlers first
    # pylint: disable=unidiomatic-typecheck
    # Need exact type match for handler registry, not isinstance
    if type(value) in _SERIALIZE_HANDLERS:
        return _SERIALIZE_HANDLERS[type(value)](value)

    # Built-in type handling
    if value is None:
        return None
    if isinstance(value, Path):
        return {"__type__": "Path", "value": str(value)}
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "value": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Enum):
        return {
            "__type__": "Enum",
            "module": value.__class__.__module__,
            "class": value.__class__.__name__,
            "value": value.value,
        }
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, set):
        return {"__type__": "set", "values": [serialize_value(v) for v in value]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "values": [serialize_value(v) for v in value]}
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, logging.Logger):
        return None  # Skip loggers

    # Fallback: string representation with warning
    warnings.warn(f"No serializer for type {type(value)}, using str()", UserWarning)
    return {"__type__": "str_fallback", "value": str(value)}


def deserialize_value(  # pylint: disable=too-many-return-statements,too-many-branches
    value: Any, target_type: Optional[Any] = None
) -> Any:
    """Deserialize JSON-compatible value to Python type.

    Args:
        value: Serialized value (from serialize_value)
        target_type: Optional type hint for deserialization

    Returns:
        Deserialized Python object
    """
    if value is None:
        return None

    # Handle type markers from serialize_value
    if isinstance(value, dict) and "__type__" in value:
        type_marker = value["__type__"]

        if type_marker == "Path":
            return Path(value["value"])
        if type_marker == "ndarray":
            return np.array(value["value"], dtype=value.get("dtype", "float64"))
        if type_marker == "Enum":
            module = importlib.import_module(value["module"])
            enum_class = getattr(module, value["class"])
            return enum_class(value["value"])
        if type_marker == "set":
            return set(deserialize_value(v) for v in value["values"])
        if type_marker == "tuple":
            return tuple(deserialize_value(v) for v in value["values"])
        if type_marker == "str_fallback":
            return value["value"]

    # Check custom handlers
    if target_type in _DESERIALIZE_HANDLERS:
        return _DESERIALIZE_HANDLERS[target_type](value)

    # Handle target_type hints
    if target_type is not None:
        if target_type == Path and isinstance(value, str):
            return Path(value)
        # Handle Optional[T] types
        if hasattr(target_type, "__origin__"):
            if target_type.__origin__ is Union:
                # Get non-None type from Optional
                # pylint: disable=unidiomatic-typecheck
                # type(None) is the idiomatic way to get NoneType
                args = [arg for arg in target_type.__args__ if arg is not type(None)]
                if args:
                    return deserialize_value(value, args[0])

    return value


def register_serializer(type_class: Type, serializer: Callable[[Any], Any]) -> None:
    """Register custom serialization handler for a type.

    Args:
        type_class: Type to register handler for
        serializer: Function that takes value and returns JSON-compatible data
    """
    _SERIALIZE_HANDLERS[type_class] = serializer


def register_deserializer(type_class: Type, deserializer: Callable[[Any], Any]) -> None:
    """Register custom deserialization handler for a type.

    Args:
        type_class: Type to register handler for
        deserializer: Function that takes serialized data and returns original type
    """
    _DESERIALIZE_HANDLERS[type_class] = deserializer


def extract_constructor_params(obj: Any, exclude: Tuple[str, ...] = ("self",)) -> Dict[str, Any]:
    """Extract constructor parameters from object instance.

    Uses inspect to discover constructor parameters and walks class hierarchy
    to capture all inherited parameters.

    Args:
        obj: Object instance to extract parameters from
        exclude: Parameter names to exclude

    Returns:
        Dictionary of parameter names to serialized values
    """
    params = {}

    # Walk through class hierarchy
    for cls in inspect.getmro(obj.__class__):
        if cls == object:
            break

        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            continue

        for param_name in sig.parameters:
            if param_name in exclude or param_name in params:
                continue

            if hasattr(obj, param_name):
                value = getattr(obj, param_name)
                params[param_name] = serialize_value(value)

    return params


def reconstruct_from_params(class_path: str, params: Dict[str, Any]) -> Any:
    """Reconstruct object from class path and parameters.

    Args:
        class_path: Full class path (e.g., "module.submodule.ClassName")
        params: Constructor parameters

    Returns:
        Reconstructed object instance

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
        TypeError: If parameters are invalid for constructor
    """
    module_name = ".".join(class_path.split(".")[:-1])
    class_name = class_path.split(".")[-1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Deserialize parameters based on constructor signature
    sig = inspect.signature(cls.__init__)
    deserialized_params = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name not in params:
            continue

        value = params[param_name]
        target_type = param.annotation if param.annotation != inspect.Parameter.empty else None
        deserialized_params[param_name] = deserialize_value(value, target_type)

    return cls(**deserialized_params)


def save_to_json(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save data to JSON file with optional metadata.

    Args:
        filepath: Path to save to
        data: Data dictionary to save
        metadata: Optional metadata (version, timestamp, etc.)

    Returns:
        Path where data was saved
    """
    save_data = {}

    if metadata:
        save_data["metadata"] = metadata

    save_data.update(data)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=_json_default)

    return filepath


def load_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        filepath: Path to load from

    Returns:
        Loaded data dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {str(e)}") from e


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for unhandled types.

    Args:
        obj: Object to serialize

    Returns:
        JSON-compatible representation
    """
    # Try serialize_value first
    result = serialize_value(obj)
    if result is not obj:  # If serialization succeeded
        return result

    # Fallback to string
    return str(obj)


def serialize_stateful_object(obj: Any) -> Dict[str, Any]:
    """Serialize object using __getstate__ protocol.

    Args:
        obj: Object with __getstate__ method

    Returns:
        Dictionary with class info and state

    Raises:
        ValueError: If object has no custom __getstate__ method
    """
    # Check if __getstate__ is explicitly defined (not just inherited from object)
    has_custom_getstate = False
    for cls in type(obj).__mro__:
        if cls is object:
            break
        if "__getstate__" in cls.__dict__:
            has_custom_getstate = True
            break

    if not has_custom_getstate:
        raise ValueError(
            f"Object {type(obj)} has no custom __getstate__ method. "
            "Only objects with explicitly defined __getstate__ can be serialized."
        )

    return {
        "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "module": obj.__class__.__module__,
        "state": obj.__getstate__(),
    }


def deserialize_stateful_object(data: Dict[str, Any]) -> Any:
    """Reconstruct object using __setstate__ protocol.

    Args:
        data: Serialized object data with keys: class, module, state

    Returns:
        Reconstructed object

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
        AttributeError: If object has no __setstate__ method
    """
    module_name = data["module"]
    class_name = data["class"].split(".")[-1]

    module = importlib.import_module(module_name)
    obj_class = getattr(module, class_name)

    obj = obj_class.__new__(obj_class)
    obj.__setstate__(data["state"])

    return obj
