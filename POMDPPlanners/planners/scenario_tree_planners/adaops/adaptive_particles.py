# SPDX-License-Identifier: MIT

"""Small, independently testable particle operations used by AdaOPS."""

from statistics import NormalDist
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Return finite non-negative weights summing to one."""
    array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return array / total


def effective_sample_size(weights: Sequence[float]) -> float:
    """Equation (1) of Wu et al. (2021), in particles."""
    normalized = normalize_weights(weights)
    return float(1.0 / np.dot(normalized, normalized))


def design_effect(weights: Sequence[float]) -> float:
    """Particle count divided by effective sample size."""
    normalized = normalize_weights(weights)
    return float(normalized.size / effective_sample_size(normalized))


def kld_sample_size(occupied_bins: int, zeta: float, confidence: float = 0.95) -> int:
    """Fox's normal-quantile approximation used by AdaOPS (paper equation 2)."""
    if occupied_bins <= 1:
        return 1
    if zeta <= 0.0:
        raise ValueError("zeta must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    degrees = occupied_bins - 1
    quantile = NormalDist().inv_cdf(confidence)
    correction = 1.0 - 2.0 / (9.0 * degrees) + quantile * np.sqrt(2.0 / (9.0 * degrees))
    return max(1, int(np.ceil(degrees * correction**3 / (2.0 * zeta))))


def bounded_kld_sample_size(
    occupied_bins: int,
    zeta: float,
    minimum: int,
    maximum: int,
    confidence: float = 0.95,
) -> int:
    """KLD size clamped to the configured particle limits."""
    if minimum <= 0 or maximum < minimum:
        raise ValueError("particle limits must satisfy 0 < minimum <= maximum")
    return min(maximum, max(minimum, kld_sample_size(occupied_bins, zeta, confidence)))


def l1_weight_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """L1 distance for sibling beliefs over one shared particle vector."""
    left_array = normalize_weights(left)
    right_array = normalize_weights(right)
    if left_array.shape != right_array.shape:
        raise ValueError("packed beliefs must share the same particle vector")
    return float(np.abs(left_array - right_array).sum())


def systematic_resample(
    particles: Sequence[Any], weights: Sequence[float], count: int, rng: np.random.Generator
) -> Tuple[List[Any], np.ndarray]:
    """Low-variance resampling with uniform output weights."""
    if count <= 0:
        raise ValueError("count must be positive")
    normalized = normalize_weights(weights)
    if len(particles) != normalized.size:
        raise ValueError("particles and weights must have the same length")
    positions = (float(rng.random()) + np.arange(count, dtype=np.float64)) / count
    indices = np.searchsorted(np.cumsum(normalized), positions, side="right")
    indices = np.clip(indices, 0, normalized.size - 1)
    return [particles[int(index)] for index in indices], np.full(count, 1.0 / count)


def occupied_bin_count(particles: Iterable[Any], state_binner: Callable[[Any], Any]) -> int:
    """Count explicit, hashable state bins; never infer a grid."""
    bins = set()
    for state in particles:
        key = state_binner(state)
        try:
            hash(key)
        except TypeError as error:
            raise TypeError("state_binner must return hashable bin keys") from error
        bins.add(key)
    return len(bins)


def adaptive_resample(
    particles: Sequence[Any],
    weights: Sequence[float],
    rng: np.random.Generator,
    minimum: int,
    maximum: int,
    zeta: float,
    state_binner: Optional[Callable[[Any], Any]],
    confidence: float = 0.95,
) -> Tuple[List[Any], np.ndarray, int]:
    """Resample to a KLD size, or to ``maximum`` when adaptation is disabled."""
    occupied = occupied_bin_count(particles, state_binner) if state_binner is not None else 0
    count = (
        bounded_kld_sample_size(occupied, zeta, minimum, maximum, confidence)
        if state_binner is not None
        else maximum
    )
    sampled, sampled_weights = systematic_resample(particles, weights, count, rng)
    return sampled, sampled_weights, occupied
