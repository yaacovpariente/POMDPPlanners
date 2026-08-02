# SPDX-License-Identifier: MIT

"""Registered-field bookkeeping for the vectorized belief tree.

A *registered field* is an extra per-node column (belief-side or action-side)
that a planning algorithm attaches to the tree without modifying the core
storage schema. For example, a PORPP/VOPP-style planner registers a
``preferences`` belief field of shape ``(num_actions,)`` and a ``value``
belief field; a POMCP-style planner registers a ``q_value`` action field.

Each field is described by an immutable :class:`FieldSpec` and backed by a
preallocated tensor that a :class:`FieldRegistry` grows in lockstep with the
node capacity of the owning tree.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
from torch import Tensor

DefaultValue = Union[float, int, bool]


@dataclass(frozen=True)
class FieldSpec:
    """Immutable description of one registered per-node field.

    Attributes:
        name: Field name, unique within its node kind and distinct from the
            built-in column names.
        shape: Trailing shape of the field for a single node. ``()`` denotes a
            scalar field; ``(num_actions,)`` denotes a per-action vector, etc.
        dtype: The tensor dtype used to store the field.
        default: Value used to initialise newly allocated rows.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    default: DefaultValue


class FieldRegistry:
    """Owns the backing tensors for the registered fields of one node kind.

    The registry preallocates one tensor per field with the tree's current
    node capacity and keeps every field tensor on the tree's device. Growing
    the registry reallocates each field tensor geometrically while preserving
    existing rows.
    """

    def __init__(self, device: torch.device, capacity: int) -> None:
        self._device = device
        self._capacity = capacity
        self._specs: Dict[str, FieldSpec] = {}
        self._tensors: Dict[str, Tensor] = {}

    @property
    def device(self) -> torch.device:
        """Device shared by every field tensor in the registry."""
        return self._device

    def names(self) -> Tuple[str, ...]:
        """Return the registered field names in insertion order."""
        return tuple(self._specs)

    def specs(self) -> Tuple[FieldSpec, ...]:
        """Return the registered :class:`FieldSpec` objects in insertion order."""
        return tuple(self._specs.values())

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def register(self, spec: FieldSpec, reserved_names: frozenset) -> None:
        """Register a new field, allocating its backing tensor.

        Args:
            spec: The field description.
            reserved_names: Built-in column names the field may not shadow.

        Raises:
            ValueError: If the name collides with a reserved name or an
                already-registered field.
        """
        if spec.name in reserved_names:
            raise ValueError(f"field name '{spec.name}' collides with a built-in column")
        if spec.name in self._specs:
            raise ValueError(f"field '{spec.name}' is already registered")
        self._specs[spec.name] = spec
        self._tensors[spec.name] = self._allocate(spec, self._capacity)

    def tensor(self, name: str) -> Tensor:
        """Return the full backing tensor for a registered field.

        Raises:
            KeyError: If no field with ``name`` is registered.
        """
        if name not in self._tensors:
            raise KeyError(f"no registered field named '{name}'")
        return self._tensors[name]

    def grow(self, new_capacity: int) -> None:
        """Reallocate every field tensor to ``new_capacity`` rows, preserving data."""
        for name, spec in self._specs.items():
            old = self._tensors[name]
            grown = self._allocate(spec, new_capacity)
            grown[: old.shape[0]] = old
            self._tensors[name] = grown
        self._capacity = new_capacity

    def reset_rows(self, start: int, end: int) -> None:
        """Reset rows ``[start, end)`` of every field tensor to its default."""
        for spec in self._specs.values():
            self._tensors[spec.name][start:end] = spec.default

    def to(self, device: torch.device) -> None:
        """Move every field tensor to ``device`` in place."""
        self._device = device
        for name, tensor in self._tensors.items():
            self._tensors[name] = tensor.to(device)

    def _allocate(self, spec: FieldSpec, capacity: int) -> Tensor:
        return torch.full(
            (capacity, *spec.shape),
            spec.default,
            dtype=spec.dtype,
            device=self._device,
        )
