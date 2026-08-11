# SPDX-License-Identifier: MIT

"""Small numeric helpers the IsaacLab stack shares.

Both functions here were copied between modules before they lived in one place, which is the usual
way two copies drift apart. :func:`first_row` had a copy in the Franka reach setup and another in
the study's task definitions; :func:`spatial_hash_primes` had three, one per vectorized model. The
prime table in particular must not drift: it fixes the observation keys a planner's tree is built
on, so two models disagreeing about it would quietly bucket the same observation differently.

Neither function imports anything from the rest of the package, so any module in the IsaacLab stack
can use it without a circular import.

Functions:
    first_row: Detach a batched torch tensor down to its first environment's row.
    spatial_hash_primes: Large primes used as per-dimension spatial-hash weights.
"""

from typing import Any

import numpy as np

#: Where the prime search starts. Large enough that the products it weights do not collide for
#: nearby observations, and odd, so the search can step by two.
_HASH_PRIME_SEED = 73856093


def first_row(value: Any) -> np.ndarray:
    """Detach a torch tensor (or array-like) down to the first environment's row.

    IsaacLab keeps every quantity batched over parallel environments even when only one is running,
    and on the GPU. A caller that wants the single environment's reading has to come back to a flat
    numpy vector on the host.

    Args:
        value: A torch tensor or anything :func:`numpy.asarray` accepts, batched over environments
            along its leading axis.

    Returns:
        The first environment's row, flattened to one dimension.

    Example:
        Reading one environment's row out of a batch::

            >>> import numpy as np
            >>> first_row(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist()
            [1.0, 2.0]
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)[0].reshape(-1)


def spatial_hash_primes(count: int) -> np.ndarray:
    """Return ``count`` large primes used as spatial-hash weights.

    A vectorized model turns a continuous observation into an integer tree key by quantizing it and
    weighting each dimension by a distinct large prime. The weights have to be the same everywhere
    for two models to agree on which observations share a node.

    Args:
        count: How many primes to return, one per observation dimension.

    Returns:
        A ``(count,)`` int64 array of distinct primes.

    Example:
        The table starts at a fixed seed, so it is the same on every call::

            >>> spatial_hash_primes(2).tolist()
            [73856093, 73856099]
    """
    primes = []
    candidate = _HASH_PRIME_SEED
    while len(primes) < count:
        if all(candidate % divisor != 0 for divisor in range(2, int(candidate**0.5) + 1)):
            primes.append(candidate)
        candidate += 2
    return np.asarray(primes, dtype=np.int64)
