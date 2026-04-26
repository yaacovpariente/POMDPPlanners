"""Microbench for the asarray-round-trip removal in WeightedParticleBelief._update_weights.

Reports sims/sec for ``planner.action(belief)`` on the 6 cells called out in the
PR plan: ``POMCPOW`` and ``PFT_DPW`` x ``ContinuousLightDark``, ``LaserTag``,
``Push``. 200 particles, 1.0s budget, seed=42, 5 trials per cell, median of trials.

Run:
    python bench_drop_asarray.py
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


PLANNER_PARAMS = {
    "discount_factor": 0.95,
    "depth": 20,
    "exploration_constant": 1.0,
    "k_o": 5.0,
    "k_a": 5.0,
    "alpha_o": 0.5,
    "alpha_a": 0.5,
    "min_visit_count_per_action": 1,
}

N_PARTICLES = 200
TIME_BUDGET_S = 1.0
REPEATS = 5
SEED = 42


def _build_continuous_light_dark() -> Any:
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (  # pylint: disable=import-outside-toplevel
        ContinuousLightDarkPOMDPDiscreteActions,
    )

    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


def _build_laser_tag() -> Any:
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (  # pylint: disable=import-outside-toplevel
        LaserTagPOMDP,
    )

    return LaserTagPOMDP(discount_factor=0.95)


def _build_push() -> Any:
    from POMDPPlanners.environments.push_pomdp.push_pomdp import (  # pylint: disable=import-outside-toplevel
        PushPOMDP,
    )

    return PushPOMDP(discount_factor=0.95)


def _make_pomcpow(env: Any) -> POMCPOW:
    return POMCPOW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pomcpow_bench",
        n_simulations=1,
        **PLANNER_PARAMS,
    )


def _make_pft_dpw(env: Any) -> PFT_DPW:
    return PFT_DPW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pft_dpw_bench",
        n_simulations=1,
        **PLANNER_PARAMS,
    )


def _time_simulate_path_arena(
    planner: Any, initial_belief: Any, time_budget_s: float
) -> Tuple[int, float]:
    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + time_budget_s
    while time.perf_counter() < deadline:
        # pylint: disable-next=protected-access
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)
        n += 1
    return n, time.perf_counter() - t0


def _bench_one_cell(
    planner_name: str,
    planner_factory: Callable[[Any], Any],
    env_name: str,
    env: Any,
) -> Tuple[float, float]:
    np.random.seed(SEED)
    initial_belief = get_initial_belief(env, n_particles=N_PARTICLES)
    planner = planner_factory(env)

    runs: List[float] = []
    for trial in range(REPEATS):
        np.random.seed(SEED + trial)
        n, t = _time_simulate_path_arena(planner, initial_belief, TIME_BUDGET_S)
        runs.append(n / t)
    median = statistics.median(runs)
    stdev = statistics.stdev(runs) if len(runs) > 1 else 0.0
    print(
        f"  {env_name:<25} {planner_name:<10} "
        f"median={median:>10,.0f} sims/s  stdev={stdev:>8,.0f}"
    )
    return median, stdev


def main() -> None:
    print("=" * 80)
    print(
        f" sims/sec bench: 200 particles, {TIME_BUDGET_S}s budget, "
        f"{REPEATS} trials, seed={SEED}"
    )
    print("=" * 80)

    envs = [
        ("ContinuousLightDark", _build_continuous_light_dark()),
        ("LaserTag", _build_laser_tag()),
        ("Push", _build_push()),
    ]
    planners = [
        ("POMCPOW", _make_pomcpow),
        ("PFT_DPW", _make_pft_dpw),
    ]

    for planner_name, factory in planners:
        for env_name, env in envs:
            _bench_one_cell(planner_name, factory, env_name, env)


if __name__ == "__main__":
    main()
