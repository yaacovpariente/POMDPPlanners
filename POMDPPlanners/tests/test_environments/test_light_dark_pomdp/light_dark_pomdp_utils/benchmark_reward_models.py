"""Performance benchmark: light_dark env-level ``reward`` / ``reward_batch``.

Times the public Environment-API reward methods
(:meth:`ContinuousLightDarkPOMDP.reward` and
:meth:`ContinuousLightDarkPOMDP.reward_batch`) end-to-end so the numbers
include the thin Python wrapper at ``continuous_light_dark_pomdp.py:637``
on top of the reward model's ``compute_reward`` / ``compute_reward_batch``.
Covers all three reward-model variants (Standard, ZERO_MEAN_HAZARD_SHOCK,
DISTANCE_DECAYED_HAZARD_PENALTY) on a fixed workload. Used to compare BEFORE vs
AFTER the dangerous-area generic-kernel refactor.

Note: the C++ ``_native.simulate_rollout`` path is a separate hot path
with its own embedded reward logic and is *not* exercised here — to
benchmark that, time ``env._rollout`` or a full MCTS rollout instead.

Run manually:
    python -m POMDPPlanners.tests.test_environments.test_light_dark_pomdp.light_dark_pomdp_utils.benchmark_reward_models
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    RewardModelType,
)

# Fixed workload: 4 obstacles on a 10x10 grid, goal at (8, 8).
GOAL_STATE = np.array([8.0, 8.0])
OBSTACLES_LIST: List[Tuple[float, float]] = [(2.0, 2.0), (4.0, 6.0), (6.0, 3.0), (5.0, 7.0)]
GOAL_RADIUS = 0.5
OBSTACLE_RADIUS = 0.8
GRID_SIZE = 10
OBSTACLE_HIT_PROB = 0.5
OBSTACLE_REWARD = -10.0
GOAL_REWARD = 100.0
FUEL_COST = 1.0
PENALTY_DECAY = 1.5

BATCH_SIZES: List[int] = [100, 1000, 10000]
SCALAR_REPS = 5000
N_RUNS_SCALAR = 5
N_RUNS_BATCH = 20


def _time_fn(fn: Callable[[], Any], n_runs: int) -> Tuple[float, float]:
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000.0  # ms
    return float(arr.mean()), float(arr.std())


def _build_envs() -> List[Tuple[str, ContinuousLightDarkPOMDP]]:
    common = {
        "discount_factor": 0.95,
        "goal_state": GOAL_STATE,
        "obstacles": OBSTACLES_LIST,
        "goal_state_radius": GOAL_RADIUS,
        "obstacle_radius": OBSTACLE_RADIUS,
        "grid_size": GRID_SIZE,
        "obstacle_hit_probability": OBSTACLE_HIT_PROB,
        "obstacle_reward": OBSTACLE_REWARD,
        "goal_reward": GOAL_REWARD,
        "fuel_cost": FUEL_COST,
        "penalty_decay": PENALTY_DECAY,
    }
    return [
        (
            "ConstantHazardPenalty",
            ContinuousLightDarkPOMDP(
                reward_model_type=RewardModelType.CONSTANT_HAZARD_PENALTY, **common
            ),
        ),
        (
            "ZeroMeanHazardShock",
            ContinuousLightDarkPOMDP(
                reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK, **common
            ),
        ),
        (
            "DistanceDecayedHazardPenalty",
            ContinuousLightDarkPOMDP(
                reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, **common
            ),
        ),
    ]


def _scalar_workload(env: ContinuousLightDarkPOMDP, states: np.ndarray, action: np.ndarray) -> None:
    for i in range(states.shape[0]):
        env.reward(states[i], action, next_state=states[i])


def _benchmark_scalar(envs: List[Tuple[str, ContinuousLightDarkPOMDP]]) -> None:
    action = np.array([0.5, 0.5])
    np.random.seed(42)
    states = np.random.rand(SCALAR_REPS, 2) * GRID_SIZE

    print(f"\n--- env.reward (reps={SCALAR_REPS}) ---")
    print(f"{'model':>14} | {'mean (ms)':>12} | {'std (ms)':>10} | {'per-call (us)':>14}")
    print("-" * 60)
    for name, env in envs:
        mean, std = _time_fn(
            lambda e=env: _scalar_workload(e, states, action),
            n_runs=N_RUNS_SCALAR,
        )
        per_call_us = (mean / SCALAR_REPS) * 1000.0
        print(f"{name:>14} | {mean:>12.3f} | {std:>10.3f} | {per_call_us:>14.3f}")


def _benchmark_batch(envs: List[Tuple[str, ContinuousLightDarkPOMDP]]) -> None:
    action = np.array([0.5, 0.5])

    for n in BATCH_SIZES:
        np.random.seed(42)
        states = np.random.rand(n, 2) * GRID_SIZE

        print(f"\n--- env.reward_batch (N={n}) ---")
        print(f"{'model':>14} | {'mean (ms)':>12} | {'std (ms)':>10}")
        print("-" * 45)
        for name, env in envs:
            mean, std = _time_fn(
                lambda e=env, s=states: e.reward_batch(s, action, next_states=s),
                n_runs=N_RUNS_BATCH,
            )
            print(f"{name:>14} | {mean:>12.3f} | {std:>10.3f}")


def main() -> None:
    envs = _build_envs()
    # Warm up the numba kernels so the first call's JIT cost is not charged
    # to the first model in each table.
    warmup_states = np.random.rand(8, 2) * GRID_SIZE
    warmup_action = np.array([0.5, 0.5])
    for _, env in envs:
        env.reward(warmup_states[0], warmup_action, next_state=warmup_states[0])
        env.reward_batch(warmup_states, warmup_action, next_states=warmup_states)

    _benchmark_scalar(envs)
    _benchmark_batch(envs)


if __name__ == "__main__":
    main()
