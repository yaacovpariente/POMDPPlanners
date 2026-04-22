"""Runtime benchmark for the PacMan C++ native port.

Runs the original PR #87 cases plus three new ones (observation sample,
POMCPOW sims/sec, and a mixed-strategy config). Numbers are medians across
``MEASURE`` iterations after ``WARMUP`` warmups; numpy seed 0 and (if
available) the module-local native RNG seed 0 are set per case. Prints a
Markdown table suitable for pasting into the PR body.

Run from the repo root with the venv activated:

    python bench_pacman_ndarray_state.py

The script auto-detects:
  - ``PacManPOMDP.make_state`` (indicates PR #87 ndarray state is in place)
  - ``POMDPPlanners.environments.pacman_pomdp._native`` (indicates the C++
    port is installed)

Rows that depend on an unavailable surface are skipped.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.tree_statistics import TreeMetrics

try:  # pylint: disable=ungrouped-imports
    from POMDPPlanners.environments.pacman_pomdp import (
        _native as _pacman_native,  # pylint: disable=no-name-in-module
    )
except ImportError:  # pragma: no cover - fallback for pre-port builds
    _pacman_native = None  # type: ignore[assignment]


WARMUP = 100
MEASURE = 1000
ROLLOUT_EPISODES = 1000
ROLLOUT_CAP = 50
BELIEF_PARTICLES = 200
BELIEF_UPDATES = 50
POMCPOW_BUDGET_SECONDS = 30


def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    if _pacman_native is not None:
        _pacman_native.set_seed(seed)


def _build_env() -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        ghost_aggressiveness=2.0,
        ghost_coordination="independent",
        discount_factor=0.95,
    )


def _build_mixed_env() -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        ghost_aggressiveness=2.0,
        ghost_coordination="mixed",
        ghost_strategies=["aggressive", "patrol"],
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


def _initial_state(env: PacManPOMDP) -> Any:
    return env.initial_state_dist().sample()[0]


def bench_reward(env: PacManPOMDP) -> float:
    _seed_all()
    state = _initial_state(env)
    return _time_per_call(lambda: env.reward(state, 1), WARMUP, MEASURE)


def bench_sample(env: PacManPOMDP) -> float:
    _seed_all()
    state = _initial_state(env)

    def once() -> object:
        return env.state_transition_model(state, 1).sample()[0]

    return _time_per_call(once, WARMUP, MEASURE)


def bench_observation_sample(env: PacManPOMDP) -> float:
    _seed_all()
    state = _initial_state(env)
    # Move once so the observation isn't trivial.
    next_state = env.state_transition_model(state, 1).sample()[0]

    def once() -> object:
        return env.observation_model(next_state, 1).sample()[0]

    return _time_per_call(once, WARMUP, MEASURE)


def bench_rollout(env: PacManPOMDP) -> Tuple[float, float]:
    _seed_all()
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
    _seed_all()
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
    _seed_all()

    def once() -> object:
        return env.make_state(  # type: ignore[attr-defined]
            pacman_pos=(3, 3),
            ghost_positions=((0, 0), (6, 6)),
            pellets=((1, 1), (1, 5), (5, 1), (5, 5)),
            score=0.0,
            terminal=False,
        )

    return _time_per_call(once, WARMUP, MEASURE)


def bench_pomcpow_sims_per_sec(env: PacManPOMDP) -> float:
    """Measure POMCPOW simulations per second under a wall-clock budget.

    Mirrors ``bench_pomcpow_sims_per_sec.py`` at the repo root: reads
    ``root_visit_count`` (== number of MCTS simulations) from the
    ``PolicyRunData`` returned by the planner. Divides by elapsed wall time.
    """
    _seed_all()
    action_sampler = DiscreteActionSampler(actions=env.get_actions())

    def _make_planner(budget: int) -> POMCPOW:
        return POMCPOW(
            environment=env,
            discount_factor=0.95,
            depth=20,
            exploration_constant=1.0,
            k_a=10.0,
            alpha_a=0.5,
            k_o=10.0,
            alpha_o=0.5,
            name=f"POMCPOW_bench_pacman_{budget}s",
            action_sampler=action_sampler,
            time_out_in_seconds=budget,
        )

    # Warm-up with a 1-second budget; discard.
    warmup_planner = _make_planner(1)
    warmup_belief = get_initial_belief(env, BELIEF_PARTICLES)
    warmup_planner.action(warmup_belief)

    _seed_all()
    planner = _make_planner(POMCPOW_BUDGET_SECONDS)
    belief = get_initial_belief(env, BELIEF_PARTICLES)
    t0 = time.perf_counter()
    _, run_data = planner.action(belief)
    elapsed = time.perf_counter() - t0
    root_visit_count = _extract_root_visit_count(run_data)
    return root_visit_count / elapsed if elapsed > 0 else float("nan")


def _extract_root_visit_count(run_data: Any) -> int:
    for info in run_data.info_variables:
        if info.name == TreeMetrics.ROOT_VISIT_COUNT.value:
            return int(info.value)
    raise RuntimeError("root_visit_count not found in PolicyRunData.info_variables")


def _fmt_us(seconds: float) -> str:
    return "n/a" if np.isnan(seconds) else f"{seconds * 1e6:.2f} µs"


def _fmt_ms(seconds: float) -> str:
    return "n/a" if np.isnan(seconds) else f"{seconds * 1e3:.3f} ms"


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f} s"


def _fmt_sims(sims_per_sec: float) -> str:
    return "n/a" if np.isnan(sims_per_sec) else f"{sims_per_sec:,.0f} sims/s"


def main() -> None:
    env = _build_env()
    mixed_env = _build_mixed_env()
    label = "native" if _pacman_native is not None else "python"
    print(f"# pacman ndarray-state benchmark ({label})")
    print(
        f"numpy seed 0{' + _native.set_seed(0)' if _pacman_native is not None else ''}, "
        f"warmup={WARMUP}, measure={MEASURE}, "
        f"rollout episodes={ROLLOUT_EPISODES} cap={ROLLOUT_CAP}, "
        f"belief particles={BELIEF_PARTICLES} updates={BELIEF_UPDATES}, "
        f"POMCPOW budget={POMCPOW_BUDGET_SECONDS}s"
    )
    print()

    reward_median = bench_reward(env)
    sample_median = bench_sample(env)
    obs_median = bench_observation_sample(env)
    rollout_ep_median, rollout_total = bench_rollout(env)
    belief_median = bench_vectorized_belief(env)
    make_state_median = bench_make_state(env)
    pomcpow_sims_per_sec = bench_pomcpow_sims_per_sec(env)

    mixed_sample_median = bench_sample(mixed_env)
    mixed_belief_median = bench_vectorized_belief(mixed_env)

    print("| case | value |")
    print("|---|---|")
    print(f"| `reward` per-call | {_fmt_us(reward_median)} |")
    print(f"| `state_transition_model.sample()` per-call | {_fmt_us(sample_median)} |")
    print(f"| `observation_model.sample()` per-call | {_fmt_us(obs_median)} |")
    print(
        f"| random-policy rollout median / total ({ROLLOUT_EPISODES} eps) "
        f"| {_fmt_ms(rollout_ep_median)} / {_fmt_s(rollout_total)} |"
    )
    print(f"| `VectorizedWeightedParticleBelief.update` per-call | {_fmt_ms(belief_median)} |")
    print(f"| `make_state` per-call | {_fmt_us(make_state_median)} |")
    print(
        f"| POMCPOW sims/sec ({POMCPOW_BUDGET_SECONDS}s budget, {BELIEF_PARTICLES} particles) "
        f"| {_fmt_sims(pomcpow_sims_per_sec)} |"
    )
    print(f"| mixed-strategy `sample()` per-call | {_fmt_us(mixed_sample_median)} |")
    print(
        f"| mixed-strategy `VectorizedWeightedParticleBelief.update` per-call "
        f"| {_fmt_ms(mixed_belief_median)} |"
    )


if __name__ == "__main__":
    main()
