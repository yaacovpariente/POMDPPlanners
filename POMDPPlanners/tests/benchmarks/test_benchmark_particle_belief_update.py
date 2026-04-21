"""Benchmarks for WeightedParticleBelief / VectorizedWeightedParticleBelief
update paths across native (MountainCar) and non-native (CartPole) envs.

Measures the four cases laid out in Layer 2's plan:

    | Case                  | Env        | Belief class                        | Path            |
    |-----------------------|------------|-------------------------------------|-----------------|
    | MC-generic-cpp        | MountainCar| WeightedParticleBelief.update       | C++ batch       |
    | MC-vectorized-cpp     | MountainCar| VectorizedWeightedParticleBelief    | C++ batch       |
    | CP-generic-python     | CartPole   | WeightedParticleBelief.update       | Python fallback |
    | CP-vectorized-numpy   | CartPole   | VectorizedWeightedParticleBelief    | numpy batch     |

Same N=100 particles, same action, same observation across all four.

Use ``pytest-benchmark compare`` across the four cases to report:
    1. MC-generic-cpp vs MC-vectorized-cpp   -- auto-dispatch parity with
        the explicit vectorized path on a native env.
    2. CP-generic-python vs CP-vectorized-numpy -- the non-native baseline
        gap (the prize a future CartPole port to pomdp_native would close).
    3. MC-generic-cpp vs CP-generic-python   -- headline speedup a native
        port buys for callers who use plain WeightedParticleBelief.

Run::

    pytest POMDPPlanners/tests/benchmarks/test_benchmark_particle_belief_update.py \\
        -m benchmark --benchmark-save=layer2_batch_dispatch -v
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_beliefs import (
    CartPoleVectorizedUpdater,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
    MountainCarVectorizedUpdater,
)

pytestmark = [pytest.mark.slow]

_N_PARTICLES = 100


# ---------------------------------------------------------------------------
# MountainCar (native) cases
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="belief-update-mc-generic-cpp")
def test_bench_mc_generic_belief_update(benchmark):
    """Benchmark WeightedParticleBelief.update on MountainCar (auto-dispatch).

    Purpose: Measures the generic per-particle-looking belief update path
    on an env whose transition/observation models expose native batch
    entry points. WeightedParticleBelief._update_weights sniffs the batch
    interface and dispatches to C++ batch_sample / batch_log_likelihood in
    a single round-trip per update.

    Given: MountainCarPOMDP + WeightedParticleBelief with N=100 particles.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    particles = list(env.initial_state_dist().sample(n_samples=_N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    observation = np.array([-0.5, 0.0])

    def run():
        return belief.update(action=1, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-mc-vectorized-cpp")
def test_bench_mc_vectorized_belief_update(benchmark):
    """Benchmark VectorizedWeightedParticleBelief.update on MountainCar.

    Purpose: Measures the explicit vectorized belief path on MountainCar.
    Its updater (MountainCarVectorizedUpdater) delegates batch_transition
    and batch_observation_log_likelihood directly to the native C++ batch
    methods.

    Given: MountainCarPOMDP + VectorizedWeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    updater = MountainCarVectorizedUpdater.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=_N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
    )
    observation = np.array([-0.5, 0.0])

    def run():
        return belief.update(action=1, observation=observation, pomdp=env)

    benchmark(run)


# ---------------------------------------------------------------------------
# CartPole (non-native) cases
# ---------------------------------------------------------------------------


def _cartpole_initial_particles(env: CartPolePOMDP, n: int) -> Any:
    """Draw n initial CartPole state particles (ndarray or list format)."""
    return env.initial_state_dist().sample(n_samples=n)


@pytest.mark.benchmark(group="belief-update-cp-generic-python")
def test_bench_cp_generic_belief_update(benchmark):
    """Benchmark WeightedParticleBelief.update on CartPole (Python fallback).

    Purpose: Measures the generic per-particle Python loop on a non-native
    env. CartPole has no ``_cpp`` extension; the auto-dispatch hasattr check
    fails, and _update_weights falls back to its pre-Layer-2 per-particle
    ``state_transition_model().sample()`` loop. This is the baseline a
    future CartPole port to ``pomdp_native`` would target.

    Given: CartPolePOMDP + WeightedParticleBelief with N=100 particles.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    particles = list(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-cp-vectorized-numpy")
def test_bench_cp_vectorized_belief_update(benchmark):
    """Benchmark VectorizedWeightedParticleBelief.update on CartPole (numpy).

    Purpose: Measures the explicit vectorized belief path on CartPole. The
    updater (CartPoleVectorizedUpdater) implements batch_transition and
    batch_observation_log_likelihood in numpy -- no C++ involved.

    Given: CartPolePOMDP + VectorizedWeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    updater = CartPoleVectorizedUpdater.from_environment(env)
    particles = np.array(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
    )
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)
