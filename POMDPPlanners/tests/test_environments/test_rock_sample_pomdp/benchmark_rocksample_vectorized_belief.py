# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Performance benchmark: RockSample vectorized vs non-vectorized belief.

Run manually to measure speedup:
    python -m POMDPPlanners.tests.test_environments.test_rock_sample_pomdp.benchmark_rocksample_vectorized_belief
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    _native,
    create_random_rock_sample,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_belief_factory import (
    create_rocksample_belief,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (
    OBS_GOOD,
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.utils.belief_factory import BeliefType


def _time_fn(fn: Callable[[], Any], n_runs: int = 100) -> Tuple[float, float]:
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000  # ms
    return float(arr.mean()), float(arr.std())


def _create_env() -> RockSamplePOMDP:
    return create_random_rock_sample(map_size=7, num_rocks=8, seed=42)


def _benchmark_single_ops(env: RockSamplePOMDP) -> None:
    print("\n--- Baseline: Single-state operations ---")
    state = env.initial_state_dist().sample()[0]
    action = 2  # East

    mean_t, std_t = _time_fn(lambda: env.sample_next_state(state=state, action=action), n_runs=500)
    print(f"  env.sample_next_state()              : {mean_t:.4f} +/- {std_t:.4f} ms")

    next_state = env.sample_next_state(state=state, action=action)
    obs = env.sample_observation(next_state=next_state, action=action)

    mean_t, std_t = _time_fn(
        lambda: env.observation_log_probability(next_state, action, [obs]),
        n_runs=500,
    )
    print(f"  env.observation_log_probability()    : {mean_t:.4f} +/- {std_t:.4f} ms")

    check_action = 5
    next_state_check = env.sample_next_state(state=state, action=check_action)
    obs_check = env.sample_observation(next_state=next_state_check, action=check_action)

    mean_t, std_t = _time_fn(
        lambda: env.observation_log_probability(next_state_check, check_action, [obs_check]),
        n_runs=500,
    )
    print(f"  env.observation_log_probability(chk) : {mean_t:.4f} +/- {std_t:.4f} ms")


def _benchmark_sample(env: RockSamplePOMDP, particle_counts: List[int]) -> None:
    print("\n--- belief.sample() ---")
    print(f"{'N':>6} | {'Baseline (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    for n in particle_counts:
        np.random.seed(42)
        base_belief = get_initial_belief(env, n)
        mean_base, _ = _time_fn(lambda b=base_belief: b.sample(), n_runs=500)

        np.random.seed(42)
        vec_belief = create_rocksample_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=n
        )
        mean_vec, _ = _time_fn(lambda b=vec_belief: b.sample(), n_runs=500)

        speedup = mean_base / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_base:>14.4f} | {mean_vec:>16.4f} | {speedup:>7.1f}x")


def _benchmark_batch_transition(env: RockSamplePOMDP, particle_counts: List[int]) -> None:
    print("\n--- batch_transition vs loop of env.sample_next_state() ---")
    print(f"{'N':>6} | {'Loop (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    updater = RockSampleVectorizedUpdater.from_environment(env)
    action = 5  # check action

    for n in particle_counts:
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=n)
        particles = np.stack(states)

        mean_loop, _ = _time_fn(
            lambda s=states: [env.sample_next_state(state=x, action=action) for x in s],
            n_runs=100,
        )

        mean_vec, _ = _time_fn(
            lambda p=particles: updater.batch_transition(p, np.array(action)),
            n_runs=100,
        )

        speedup = mean_loop / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_loop:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def _benchmark_batch_log_likelihood(env: RockSamplePOMDP, particle_counts: List[int]) -> None:
    print("\n--- batch_observation_log_likelihood vs loop of env.observation_log_probability() ---")
    print(f"{'N':>6} | {'Loop (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    updater = RockSampleVectorizedUpdater.from_environment(env)
    action = 5
    obs_str = "good"

    for n in particle_counts:
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=n)
        particles = np.stack(states)

        mean_loop, _ = _time_fn(
            lambda s=states: [env.observation_log_probability(x, action, [obs_str])[0] for x in s],
            n_runs=100,
        )

        mean_vec, _ = _time_fn(
            lambda p=particles: updater.batch_observation_log_likelihood(
                p, np.array(action), np.array(OBS_GOOD)
            ),
            n_runs=100,
        )

        speedup = mean_loop / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_loop:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def _benchmark_belief_update(env: RockSamplePOMDP, particle_counts: List[int]) -> None:
    print("\n--- belief.update() ---")
    print(f"{'N':>6} | {'Baseline (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    action = 5
    state = env.initial_state_dist().sample()[0]
    next_state = env.sample_next_state(state=state, action=action)
    obs = env.sample_observation(next_state=next_state, action=action)

    for n in particle_counts:
        np.random.seed(42)
        base_belief = get_initial_belief(env, n)
        mean_base, _ = _time_fn(
            lambda b=base_belief: b.update(action=action, observation=obs, pomdp=env),
            n_runs=20,
        )

        np.random.seed(42)
        vec_belief = create_rocksample_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=n
        )
        mean_vec, _ = _time_fn(
            lambda b=vec_belief: b.update(action=action, observation=obs, pomdp=env),
            n_runs=20,
        )

        speedup = mean_base / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_base:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def main() -> None:
    print("=" * 80)
    print("RockSample Vectorized Belief Benchmark")
    print("=" * 80)

    # Fixed seeds for reproducible numbers across runs. The native C++
    # extension has its own module-level RNG that numpy's seed does not
    # reach; seed both so sample() paths are reproducible.
    np.random.seed(0)
    _native.set_seed(0)

    env = _create_env()
    particle_counts = [200, 500, 1000]

    _benchmark_single_ops(env)
    _benchmark_sample(env, particle_counts)
    _benchmark_batch_transition(env, particle_counts)
    _benchmark_batch_log_likelihood(env, particle_counts)
    _benchmark_belief_update(env, particle_counts)

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
