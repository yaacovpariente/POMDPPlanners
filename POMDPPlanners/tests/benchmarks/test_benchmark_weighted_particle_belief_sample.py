"""Benchmarks for ``WeightedParticleBelief.sample``.

Compares the **lazy-CDF** implementation (``bisect_left`` on a cached
cumulative-weight list, O(log K) per call after the first) against an
inlined **Python-reference** that mirrors the pre-optimization path
(``np.random.choice`` with a probability vector, O(K) per call). The
reference lives in this test file so it keeps running the baseline code
path even after the shipped ``sample`` has been replaced.

Cases:

| case                              | impl     | N              | metric |
|-----------------------------------|----------|----------------|--------|
| sample-cached-cdf                 | new      | 200/500/1k/2k  | ops/s  |
| sample-np-choice-ref              | baseline | 200/500/1k/2k  | ops/s  |
| sample-first-call-cdf             | new      | 200/500/1k/2k  | ops/s  |

``sample-cached-cdf`` benchmarks steady-state sampling on a belief whose
CDF has already been built (the common case inside MCTS rollouts). The
``sample-first-call-cdf`` case rebuilds a fresh belief per round to
capture the one-time CDF construction cost.

Run::

    pytest POMDPPlanners/tests/benchmarks/test_benchmark_weighted_particle_belief_sample.py \\
        --benchmark-only -v
"""

from __future__ import annotations

import random
from typing import Any, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief

N_VALUES = [200, 500, 1000, 2000]


def _generate_particles(n: int, dim: int = 2, seed: int = 123) -> Tuple[List[Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    particles: List[Any] = [rng.standard_normal(dim) for _ in range(n)]
    log_weights = rng.uniform(-2.0, 2.0, size=n)
    return particles, log_weights


class _PyRefBelief:
    """Python-reference mirroring the pre-optimization ``sample`` path.

    Inlined so the benchmark keeps exercising the ``np.random.choice``
    baseline even after the shipped ``WeightedParticleBelief.sample`` has
    been migrated to the lazy-CDF + ``bisect_left`` implementation.
    """

    def __init__(self, particles: List[Any], log_weights: np.ndarray) -> None:
        self.particles = particles
        weights = np.exp(log_weights - np.max(log_weights))
        self.normalized_weights = weights / np.sum(weights)

    def sample(self) -> Any:
        idx = np.random.choice(len(self.particles), p=self.normalized_weights)
        return self.particles[idx]


# ---------------------------------------------------------------------------
# steady-state sampling (CDF cached)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="sample-cached-cdf")
@pytest.mark.parametrize("n", N_VALUES)
def test_benchmark_sample_cached_cdf(benchmark, n: int) -> None:
    """New sample(): cached-CDF draw on a pre-built belief."""
    particles, log_weights = _generate_particles(n)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    belief.sample()  # prime the CDF so the benchmark measures steady state
    random.seed(42)
    benchmark(belief.sample)


@pytest.mark.benchmark(group="sample-np-choice-ref")
@pytest.mark.parametrize("n", N_VALUES)
def test_benchmark_sample_np_choice_ref(benchmark, n: int) -> None:
    """Baseline sample(): ``np.random.choice`` O(K) draw."""
    particles, log_weights = _generate_particles(n)
    belief = _PyRefBelief(particles, log_weights)
    np.random.seed(42)
    benchmark(belief.sample)


# ---------------------------------------------------------------------------
# first-call sampling (fresh belief each round, measures CDF build + draw)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="sample-first-call-cdf")
@pytest.mark.parametrize("n", N_VALUES)
def test_benchmark_sample_first_call_cdf(benchmark, n: int) -> None:
    """New sample(): fresh belief per round -- one-time CDF build + draw."""
    particles, log_weights = _generate_particles(n)

    def _run() -> None:
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        belief.sample()

    random.seed(42)
    benchmark(_run)
