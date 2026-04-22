"""Runtime benchmark for the PacMan state-as-ndarray refactor.

Runs five fixed cases and prints a Markdown table suitable for pasting into
the PR body. Numbers are medians across ``measure`` iterations after a
``warmup`` phase; numpy RNG is seeded to 0 for every case.

Run from the repo root with the venv activated:

    python bench_pacman_ndarray_state.py

The same file is run twice: once on ``origin/develop`` for the baseline,
once on the tip of ``refactor/pacman-ndarray-state`` for the after numbers.
The script auto-detects whether ``make_state`` is available and skips the
``make_state`` row on the baseline.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


WARMUP = 100
MEASURE = 1000
ROLLOUT_EPISODES = 1000
ROLLOUT_CAP = 50
BELIEF_PARTICLES = 200
BELIEF_UPDATES = 50


def _build_env() -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        discount_factor=0.95,
    )


def _time_per_call(fn: Callable[[], object], warmup: int, measure: int) -> float:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(measure):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _has_make_state(env: PacManPOMDP) -> bool:
    return callable(getattr(env, "make_state", None))


def _initial_state(env: PacManPOMDP) -> object:
    return env.initial_state_dist().sample()[0]


def bench_reward(env: PacManPOMDP) -> float:
    np.random.seed(0)
    state = _initial_state(env)
    return _time_per_call(lambda: env.reward(state, 1), WARMUP, MEASURE)


def bench_sample(env: PacManPOMDP) -> float:
    np.random.seed(0)
    state = _initial_state(env)

    def once() -> object:
        return env.state_transition_model(state, 1).sample()[0]

    return _time_per_call(once, WARMUP, MEASURE)


def bench_rollout(env: PacManPOMDP) -> Tuple[float, float]:
    np.random.seed(0)
    actions = env.get_actions()
    durations: List[float] = []
    for _ in range(ROLLOUT_EPISODES):
        state = _initial_state(env)
        t0 = time.perf_counter()
        for _ in range(ROLLOUT_CAP):
            if env.is_terminal(state):
                break
            action = actions[np.random.randint(len(actions))]
            state = env.state_transition_model(state, action).sample()[0]
        durations.append(time.perf_counter() - t0)
    return statistics.median(durations), sum(durations)


def bench_vectorized_belief(env: PacManPOMDP) -> float:
    np.random.seed(0)
    belief = create_pacman_belief(
        env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=BELIEF_PARTICLES
    )
    assert isinstance(belief, VectorizedWeightedParticleBelief)
    samples: List[float] = []
    action = 2
    obs = np.array([4.0, 4.0, 4.0, 4.0])
    for _ in range(BELIEF_UPDATES):
        t0 = time.perf_counter()
        belief = belief.update(action=action, observation=obs, pomdp=env)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_make_state(env: PacManPOMDP) -> float:
    if not _has_make_state(env):
        return float("nan")
    np.random.seed(0)

    def once() -> object:
        return env.make_state(  # type: ignore[attr-defined]
            pacman_pos=(3, 3),
            ghost_positions=((0, 0), (6, 6)),
            pellets=((1, 1), (1, 5), (5, 1), (5, 5)),
            score=0.0,
            terminal=False,
        )

    return _time_per_call(once, WARMUP, MEASURE)


def _fmt_us(seconds: float) -> str:
    return "n/a" if np.isnan(seconds) else f"{seconds * 1e6:.2f} µs"


def _fmt_ms(seconds: float) -> str:
    return "n/a" if np.isnan(seconds) else f"{seconds * 1e3:.3f} ms"


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f} s"


def main() -> None:
    env = _build_env()
    label = "after" if _has_make_state(env) else "before (develop)"
    print(f"# pacman ndarray-state benchmark ({label})")
    print(
        f"numpy seed 0, warmup={WARMUP}, measure={MEASURE}, "
        f"rollout episodes={ROLLOUT_EPISODES} cap={ROLLOUT_CAP}, "
        f"belief particles={BELIEF_PARTICLES} updates={BELIEF_UPDATES}"
    )
    print()

    reward_median = bench_reward(env)
    sample_median = bench_sample(env)
    rollout_ep_median, rollout_total = bench_rollout(env)
    belief_median = bench_vectorized_belief(env)
    make_state_median = bench_make_state(env)

    print("| case | value |")
    print("|---|---|")
    print(f"| `reward` per-call | {_fmt_us(reward_median)} |")
    print(f"| `state_transition_model.sample()` per-call | {_fmt_us(sample_median)} |")
    print(
        f"| random-policy rollout median / total ({ROLLOUT_EPISODES} eps) "
        f"| {_fmt_ms(rollout_ep_median)} / {_fmt_s(rollout_total)} |"
    )
    print(f"| `VectorizedWeightedParticleBelief.update` per-call | {_fmt_ms(belief_median)} |")
    print(f"| `make_state` per-call | {_fmt_us(make_state_median)} |")


if __name__ == "__main__":
    main()
