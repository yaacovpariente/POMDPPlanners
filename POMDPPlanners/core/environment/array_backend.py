# SPDX-License-Identifier: MIT

"""Letting one model serve both a scalar planner and a GPU-vectorized one.

A vectorized planner such as VOPP drives its whole forward search in torch on
one device, and its contract is that nothing crosses back to the host mid-search
-- a single ``.numpy()`` per step costs more than the kernels it surrounds. A
model written in numpy therefore cannot serve it, however fast the numpy is.

The alternative to writing each model twice is to write the arithmetic once and
let it follow its input: numpy in, numpy out; a tensor in, a tensor out on that
tensor's device and dtype. The formulas are identical either way -- ``@``,
``sqrt``, ``sum`` mean the same thing in both libraries -- so what actually has
to be arranged is the model's *stored parameters*, which are numpy and must
appear as tensors on the right device without being converted on every call.

That is what :class:`BackendParameters` is: the numpy master copy, plus a
converted copy per ``(device, dtype)`` built once and reused. Converting per
call would move the cost from the host boundary into the kernel and give back
the whole point.

Classes:
    BackendParameters: Stored arrays, served in the backend of a reference array.

Functions:
    is_tensor: Whether a value is a torch tensor.
    as_backend: Convert a value into the backend of a reference array.
    as_rows: View a vector or a batch of vectors as 2-D, reporting which it was.
    standard_normal: Standard-normal draw in the backend of a reference array.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch


def is_tensor(value: Any) -> bool:
    """Whether ``value`` is a torch tensor.

    Args:
        value: Any array-like.

    Returns:
        True for a :class:`torch.Tensor`, False otherwise.
    """
    return isinstance(value, torch.Tensor)


class BackendParameters:
    """Fixed numpy parameters, served in whichever backend the caller is using.

    Args:
        arrays: Named parameter arrays. Stored as float64 numpy masters.

    Example:
        >>> import numpy as np
        >>> params = BackendParameters(weight=np.eye(2))
        >>> params.matching(np.zeros(2))["weight"].shape
        (2, 2)
        >>> import torch
        >>> params.matching(torch.zeros(2))["weight"].dtype
        torch.float32
    """

    def __init__(self, **arrays: Any) -> None:
        self._numpy: Dict[str, np.ndarray] = {
            name: np.asarray(value, dtype=float) for name, value in arrays.items()
        }
        self._converted: Dict[Tuple[Any, Any], Dict[str, torch.Tensor]] = {}

    @property
    def numpy(self) -> Mapping[str, np.ndarray]:
        """The numpy master copies."""
        return self._numpy

    def matching(self, reference: Any) -> Mapping[str, Any]:
        """The parameters in the backend, device and dtype of ``reference``.

        Args:
            reference: An array whose backend the parameters should match.

        Returns:
            The parameter mapping. For a numpy reference this is the master copy
            itself; for a tensor it is a per-``(device, dtype)`` copy, built on
            first use and cached thereafter.
        """
        if not is_tensor(reference):
            return self._numpy
        key = (reference.device, reference.dtype)
        if key not in self._converted:
            self._converted[key] = {
                name: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
                for name, value in self._numpy.items()
            }
        return self._converted[key]


def as_backend(value: Any, reference: Any) -> Any:
    """Convert ``value`` into the backend, device and dtype of ``reference``.

    The *state* is what decides a model's backend, and everything else it is
    handed follows -- an action arriving as a Python list must not silently drag
    a tensor computation back to the host.

    Args:
        value: Any array-like.
        reference: The array whose backend to adopt.

    Returns:
        ``value`` as a numpy array or as a tensor on the reference's device.
    """
    if is_tensor(reference):
        if is_tensor(value):
            return value.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(
            np.asarray(value, dtype=float), dtype=reference.dtype, device=reference.device
        )
    if is_tensor(value):
        return value.detach().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def as_rows(value: Any, width: int) -> Tuple[Any, bool]:
    """View ``value`` as a ``(N, width)`` block and report whether it arrived batched.

    A single vector and a one-row batch are the same numbers in different shapes,
    and the caller has to give back the shape it was handed -- returning a
    ``(1, width)`` array to someone who passed a ``(width,)`` one breaks every
    existing call site silently rather than loudly.

    Args:
        value: A ``(width,)`` vector or a ``(N, width)`` batch.
        width: Expected trailing width.

    Returns:
        ``(rows, was_batched)`` where ``rows`` is 2-D.

    Raises:
        ValueError: If the trailing dimension is not ``width``, or the input has
            more than two dimensions.
    """
    array = value if is_tensor(value) else np.asarray(value, dtype=float)
    if array.ndim > 2:
        raise ValueError(f"expected a vector or a batch of vectors, got shape {tuple(array.shape)}")
    if int(array.shape[-1]) != width:
        raise ValueError(
            f"expected trailing dimension {width}, got shape {tuple(array.shape)}"
        )
    if array.ndim == 1:
        return array.reshape(1, width), False
    return array, True


def standard_normal(
    shape: Tuple[int, ...],
    reference: Any,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Draw standard-normal noise in the backend of ``reference``.

    Args:
        shape: Shape of the draw.
        reference: An array whose backend, device and dtype to match.
        rng: Generator for the numpy path. ``None`` uses the global numpy state,
            which is what the scalar models have always used.

    Returns:
        The noise, in the reference's backend.
    """
    if is_tensor(reference):
        return torch.randn(shape, dtype=reference.dtype, device=reference.device)
    if rng is not None:
        return rng.standard_normal(shape)
    return np.random.standard_normal(shape)
