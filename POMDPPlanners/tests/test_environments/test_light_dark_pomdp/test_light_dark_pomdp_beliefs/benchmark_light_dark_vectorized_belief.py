"""Performance benchmark: LightDark vectorized vs non-vectorized belief.

Benchmarks all 6 observation-model-aware vectorized updaters (3 continuous,
3 discrete) against the standard per-particle loop.

Run manually:
    python -m POMDPPlanners.tests.test_environments.test_light_dark_pomdp.test_light_dark_pomdp_beliefs.benchmark_light_dark_vectorized_belief
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ObservationModelType as ContinuousObsType,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
    ObservationModelType as DiscreteObsType,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    ContinuousLightDarkDistanceBasedVectorizedUpdater,
    ContinuousLightDarkNoObsInDarkVectorizedUpdater,
    ContinuousLightDarkVectorizedUpdater,
    DiscreteLightDarkDistanceBasedVectorizedUpdater,
    DiscreteLightDarkNoObsInDarkVectorizedUpdater,
    DiscreteLightDarkVectorizedUpdater,
)

PARTICLE_COUNTS = [200]
N_RUNS_FAST = 5
N_RUNS_UPDATE = 5


def _time_fn(fn: Callable[[], Any], n_runs: int = 100) -> Tuple[float, float]:
    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000  # ms
    return float(arr.mean()), float(arr.std())


def _print_table_header(title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"{'N':>6} | {'Loop (ms)':>14} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print("-" * 55)


def _print_row(n: int, mean_loop: float, mean_vec: float) -> None:
    speedup = mean_loop / mean_vec if mean_vec > 0 else float("inf")
    print(f"{n:>6} | {mean_loop:>14.3f} | {mean_vec:>16.3f} | {speedup:>7.1f}x")


# ======================================================================
# Continuous LightDark benchmarks
# ======================================================================


def _benchmark_continuous_transition(env: ContinuousLightDarkPOMDP, updater) -> None:
    _print_table_header("batch_transition vs per-particle loop")
    action = np.array([1.0, 0.0])

    for n in PARTICLE_COUNTS:
        np.random.seed(42)
        particles = np.random.rand(n, 2) * 10
        states = [particles[i].copy() for i in range(n)]

        mean_loop, _ = _time_fn(
            lambda: [env.sample_next_state(s, action) for s in states],
            n_runs=N_RUNS_FAST,
        )
        mean_vec, _ = _time_fn(
            lambda: updater.batch_transition(particles, action),
            n_runs=N_RUNS_FAST,
        )
        _print_row(n, mean_loop, mean_vec)


def _benchmark_continuous_log_lik(env: ContinuousLightDarkPOMDP, updater, observation: Any) -> None:
    obs_label = repr(observation) if isinstance(observation, str) else "array"
    _print_table_header(f"batch_observation_log_likelihood vs loop (obs={obs_label})")
    action = np.array([1.0, 0.0])

    for n in PARTICLE_COUNTS:
        np.random.seed(42)
        particles = np.random.rand(n, 2) * 10

        if isinstance(observation, str):
            mean_loop, _ = _time_fn(
                lambda: np.array(
                    [
                        env.observation_log_probability(particles[i], action, [observation])[0]
                        for i in range(n)
                    ]
                ),
                n_runs=N_RUNS_FAST,
            )
        else:

            def _per_particle_log_lik():
                result = np.empty(n)
                for i in range(n):
                    obs_model = env.observation_model(particles[i], action)
                    dist = getattr(obs_model, "_active_dist")
                    result[i] = dist.log_pdf(np.array([observation]), particles[i])[0]
                return result

            mean_loop, _ = _time_fn(_per_particle_log_lik, n_runs=N_RUNS_FAST)

        mean_vec, _ = _time_fn(
            lambda: updater.batch_observation_log_likelihood(particles, action, observation),
            n_runs=N_RUNS_FAST,
        )
        _print_row(n, mean_loop, mean_vec)


def _benchmark_continuous_update(env: ContinuousLightDarkPOMDP, updater, observation: Any) -> None:
    obs_label = repr(observation) if isinstance(observation, str) else "array"
    _print_table_header(f"belief.update() (obs={obs_label})")
    action = np.array([1.0, 0.0])

    for n in PARTICLE_COUNTS:
        np.random.seed(42)
        particles_array = np.random.rand(n, 2) * 10
        log_w = np.full(n, -np.log(n))

        # Per-particle baseline
        mean_base, _ = _time_fn(
            lambda: get_initial_belief(env, n).update(
                action=action, observation=observation, pomdp=env
            ),
            n_runs=N_RUNS_UPDATE,
        )

        # Vectorized
        def vec_update():
            belief = VectorizedWeightedParticleBelief(
                particles_array.copy(), log_w.copy(), updater, resampling=False
            )
            return belief.update(action=action, observation=observation)

        # Only test array observations (VectorizedWeightedParticleBelief
        # doesn't yet support string observations in update())
        if isinstance(observation, str):
            print(f"{n:>6} | {'(skipped — string obs not supported in update())':>46}")
            continue

        mean_vec, _ = _time_fn(vec_update, n_runs=N_RUNS_UPDATE)
        _print_row(n, mean_base, mean_vec)


def _run_continuous_benchmark(
    name: str, obs_type: ContinuousObsType, updater_cls, observations: List[Any]
) -> None:
    print(f"\n{'=' * 70}")
    print(f"Continuous LightDark — {name}")
    print(f"{'=' * 70}")

    env = ContinuousLightDarkPOMDP(discount_factor=0.95, observation_model_type=obs_type)
    updater = updater_cls.from_environment(env)

    _benchmark_continuous_transition(env, updater)
    for obs in observations:
        _benchmark_continuous_log_lik(env, updater, obs)
    for obs in observations:
        _benchmark_continuous_update(env, updater, obs)


# ======================================================================
# Discrete LightDark benchmarks
# ======================================================================


def _benchmark_discrete_transition(env: DiscreteLightDarkPOMDP, updater) -> None:
    _print_table_header("batch_transition vs per-particle loop")
    action = "right"

    for n in PARTICLE_COUNTS:
        np.random.seed(42)
        particles = np.random.randint(0, 10, size=(n, 2)).astype(float)
        states = [particles[i].copy() for i in range(n)]

        mean_loop, _ = _time_fn(
            lambda: [env.sample_next_state(s, action) for s in states],
            n_runs=N_RUNS_FAST,
        )
        mean_vec, _ = _time_fn(
            lambda: updater.batch_transition(particles, action),
            n_runs=N_RUNS_FAST,
        )
        _print_row(n, mean_loop, mean_vec)


def _benchmark_discrete_log_lik(env: DiscreteLightDarkPOMDP, updater, observation: Any) -> None:
    obs_label = repr(observation) if isinstance(observation, str) else "array"
    _print_table_header(f"batch_observation_log_likelihood vs loop (obs={obs_label})")
    action = "right"

    for n in PARTICLE_COUNTS:
        np.random.seed(42)
        particles = np.random.randint(0, 10, size=(n, 2)).astype(float)

        mean_loop, _ = _time_fn(
            lambda: np.array(
                [
                    env.observation_log_probability(particles[i], action, [observation])[0]
                    for i in range(n)
                ]
            ),
            n_runs=N_RUNS_FAST,
        )
        mean_vec, _ = _time_fn(
            lambda: updater.batch_observation_log_likelihood(particles, action, observation),
            n_runs=N_RUNS_FAST,
        )
        _print_row(n, mean_loop, mean_vec)


def _benchmark_discrete_update(env: DiscreteLightDarkPOMDP, updater, observation: Any) -> None:
    obs_label = repr(observation) if isinstance(observation, str) else "array"
    _print_table_header(f"belief.update() (obs={obs_label})")
    action = "right"

    for n in PARTICLE_COUNTS:
        np.random.seed(42)

        # Per-particle baseline
        mean_base, _ = _time_fn(
            lambda: get_initial_belief(env, n).update(
                action=action, observation=observation, pomdp=env
            ),
            n_runs=N_RUNS_UPDATE,
        )

        # Vectorized — only for non-string observations
        if isinstance(observation, str):
            print(f"{n:>6} | {'(skipped — string obs not supported in update())':>46}")
            continue

        particles_array = np.random.randint(0, 10, size=(n, 2)).astype(float)
        log_w = np.full(n, -np.log(n))

        def vec_update():
            belief = VectorizedWeightedParticleBelief(
                particles_array.copy(), log_w.copy(), updater, resampling=False
            )
            return belief.update(action=action, observation=observation)

        mean_vec, _ = _time_fn(vec_update, n_runs=N_RUNS_UPDATE)
        _print_row(n, mean_base, mean_vec)


def _run_discrete_benchmark(
    name: str, obs_type: DiscreteObsType, updater_cls, observations: List[Any]
) -> None:
    print(f"\n{'=' * 70}")
    print(f"Discrete LightDark — {name}")
    print(f"{'=' * 70}")

    env = DiscreteLightDarkPOMDP(discount_factor=0.95, observation_model_type=obs_type)
    updater = updater_cls.from_environment(env)

    _benchmark_discrete_transition(env, updater)
    for obs in observations:
        _benchmark_discrete_log_lik(env, updater, obs)
    for obs in observations:
        _benchmark_discrete_update(env, updater, obs)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    print("=" * 70)
    print("LightDark Vectorized Belief Benchmark")
    print(f"Particle counts: {PARTICLE_COUNTS}")
    print("=" * 70)

    # --- Continuous ---
    _run_continuous_benchmark(
        "NORMAL_NOISE",
        ContinuousObsType.NORMAL_NOISE,
        ContinuousLightDarkVectorizedUpdater,
        [np.array([5.0, 5.0])],
    )
    _run_continuous_benchmark(
        "NORMAL_NOISE_NO_OBS_IN_DARK",
        ContinuousObsType.NORMAL_NOISE_NO_OBS_IN_DARK,
        ContinuousLightDarkNoObsInDarkVectorizedUpdater,
        [np.array([5.0, 5.0]), "None"],
    )
    _run_continuous_benchmark(
        "DISTANCE_BASED",
        ContinuousObsType.DISTANCE_BASED,
        ContinuousLightDarkDistanceBasedVectorizedUpdater,
        [np.array([5.0, 5.0]), "None"],
    )

    # --- Discrete ---
    _run_discrete_benchmark(
        "NORMAL",
        DiscreteObsType.NORMAL,
        DiscreteLightDarkVectorizedUpdater,
        [np.array([5, 5])],
    )
    _run_discrete_benchmark(
        "NO_OBS_IN_DARK",
        DiscreteObsType.NO_OBS_IN_DARK,
        DiscreteLightDarkNoObsInDarkVectorizedUpdater,
        [np.array([5, 5]), "None"],
    )
    _run_discrete_benchmark(
        "DISTANCE_BASED",
        DiscreteObsType.DISTANCE_BASED,
        DiscreteLightDarkDistanceBasedVectorizedUpdater,
        [np.array([5, 5]), "None"],
    )

    print(f"\n{'=' * 70}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
