# SPDX-License-Identifier: MIT

"""Performance benchmark: PacMan vectorized vs non-vectorized belief.

Run manually to measure speedup:
    python -m POMDPPlanners.tests.test_environments.test_pacman_pomdp_beliefs.benchmark_pacman_vectorized_belief
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
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


def _create_env() -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        ghost_coordination="independent",
        ghost_aggressiveness=2.0,
        discount_factor=0.95,
    )


def _benchmark_single_ops(env: PacManPOMDP) -> None:
    print("\n--- Baseline: Single-state operations ---")
    state = env.initial_state_dist().sample()[0]
    action = 2

    mean_t, std_t = _time_fn(lambda: env.sample_next_state(state=state, action=action), n_runs=500)
    print(f"  env.sample_next_state                : {mean_t:.4f} +/- {std_t:.4f} ms")

    next_state = env.sample_next_state(state=state, action=action)
    obs = env.sample_observation(next_state=next_state, action=action)

    mean_t, std_t = _time_fn(
        lambda: np.exp(env.observation_log_probability(next_state, action, [obs])),
        n_runs=500,
    )
    print(f"  env.observation_log_probability      : {mean_t:.4f} +/- {std_t:.4f} ms")


def _benchmark_belief_update(env: PacManPOMDP, particle_counts: List[int]) -> None:
    print("\n--- Belief Update ---")
    print(f"{'N':>6} | {'Baseline (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    action = 2
    state = env.initial_state_dist().sample()[0]
    next_state = env.sample_next_state(state=state, action=action)
    obs = env.sample_observation(next_state=next_state, action=action)
    obs_arr = env.observation_to_array(obs)

    for n in particle_counts:
        # Baseline: WeightedParticleBelief
        np.random.seed(42)
        base_belief = get_initial_belief(env, n)
        mean_base, _ = _time_fn(
            lambda: base_belief.update(action=action, observation=obs, pomdp=env),
            n_runs=20,
        )

        # Vectorized
        np.random.seed(42)
        vec_belief = create_pacman_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=n
        )
        mean_vec, _ = _time_fn(
            lambda: vec_belief.update(action=action, observation=obs_arr, pomdp=env),
            n_runs=20,
        )

        speedup = mean_base / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_base:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def _benchmark_reward_batch(env: PacManPOMDP, particle_counts: List[int]) -> None:
    print("\n--- reward_batch ---")
    print(f"{'N':>6} | {'Loop (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    action = 2

    for n in particle_counts:
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=n)
        arr = np.stack(states)

        # Baseline: loop
        mean_base, _ = _time_fn(
            lambda: np.array([env.reward(states[i], action) for i in range(n)]),
            n_runs=20,
        )

        # Vectorized
        mean_vec, _ = _time_fn(lambda: env.reward_batch(arr, action), n_runs=100)

        speedup = mean_base / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_base:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def _benchmark_belief_expectation_reward(env: PacManPOMDP, particle_counts: List[int]) -> None:
    print("\n--- belief_expectation_reward ---")
    print(f"{'N':>6} | {'Baseline (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)

    action = 2

    for n in particle_counts:
        # Baseline
        np.random.seed(42)
        base_belief = get_initial_belief(env, n)
        mean_base, _ = _time_fn(
            lambda: belief_expectation_reward(belief=base_belief, action=action, env=env),
            n_runs=50,
        )

        # Vectorized
        np.random.seed(42)
        vec_belief = create_pacman_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=n
        )
        mean_vec, _ = _time_fn(
            lambda vb=vec_belief: belief_expectation_reward(belief=vb, action=action, env=env),
            n_runs=50,
        )

        speedup = mean_base / mean_vec if mean_vec > 0 else float("inf")
        print(f"{n:>6} | {mean_base:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


def _benchmark_batch_transition(env: PacManPOMDP, particle_counts: List[int]) -> None:
    print("\n--- batch_transition ---")
    updater = PacManVectorizedUpdater.from_environment(env)
    print(f"{'N':>6} | {'Vectorized (ms)':>16}")
    print("-" * 26)

    action = 2
    for n in particle_counts:
        np.random.seed(42)
        states = env.initial_state_dist().sample(n_samples=n)
        arr = np.stack(states)
        mean_t, _ = _time_fn(
            lambda a=arr: updater.batch_transition(a, np.array(action)),  # type: ignore[arg-type]
            n_runs=100,
        )
        print(f"{n:>6} | {mean_t:>16.3f}")


def main() -> None:
    print("=" * 55)
    print("PacMan Vectorized Belief Benchmark")
    print("=" * 55)

    env = _create_env()
    particle_counts = [50, 200, 500, 1000]

    _benchmark_single_ops(env)
    _benchmark_batch_transition(env, particle_counts)
    _benchmark_reward_batch(env, particle_counts)
    _benchmark_belief_expectation_reward(env, particle_counts)
    _benchmark_belief_update(env, particle_counts)

    print("\n" + "=" * 55)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
