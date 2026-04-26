"""Sims/sec bench for ContinuousLightDarkPOMDP with the action-sampler /
action-hashing PR.

Drives ``planner._simulate_path`` directly under a 1.0 s wall-clock budget per
trial, after a 0.1 s warm-up. Reports the median across 3 trials for both
POMCPOW and PFT-DPW.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler


PLANNER_PARAMS = dict(
    discount_factor=0.95,
    depth=20,
    exploration_constant=1.0,
    k_o=5.0,
    k_a=5.0,
    alpha_o=0.5,
    alpha_a=0.5,
    min_visit_count_per_action=1,
)
TIME_BUDGET_S = 1.0
WARMUP_S = 0.1
N_PARTICLES = 100
N_REPEATS = 3


def make_pomcpow(env: Any) -> POMCPOW:
    return POMCPOW(
        environment=env,
        action_sampler=UnitCircleActionSampler(max_action_magnitude=1.0),
        name="pomcpow_bench",
        n_simulations=1,
        **PLANNER_PARAMS,
    )


def make_pft_dpw(env: Any) -> PFT_DPW:
    return PFT_DPW(
        environment=env,
        action_sampler=UnitCircleActionSampler(max_action_magnitude=1.0),
        name="pft_dpw_bench",
        n_simulations=1,
        **PLANNER_PARAMS,
    )


def run_for_budget(planner: Any, belief: Any, budget_s: float) -> Tuple[int, float]:
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    while time.perf_counter() < deadline:
        planner._simulate_path(
            tree=tree, belief_id=root_id, depth=0
        )  # pylint: disable=protected-access
        n += 1
    return n, time.perf_counter() - t0


def bench_pair(name: str, factory: Callable[[Any], Any], env: Any) -> None:
    rates: List[float] = []
    for _ in range(N_REPEATS):
        np.random.seed(0)
        belief_i = get_initial_belief(env, n_particles=N_PARTICLES)
        planner_i = factory(env)
        run_for_budget(planner_i, belief_i, WARMUP_S)
        n, t = run_for_budget(planner_i, belief_i, TIME_BUDGET_S)
        rates.append(n / t)
    median = statistics.median(rates)
    stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
    print(
        f"  {name:<8} x ContinuousLightDark   median={median:>8,.0f} sims/s  "
        f"stdev={stdev:>7,.0f}  runs={[f'{r:.0f}' for r in rates]}"
    )


def main() -> None:
    env = ContinuousLightDarkPOMDP(discount_factor=0.95)
    print()
    print("=" * 80)
    print(
        f"  CLD bench (action-hashing PR) — {TIME_BUDGET_S:.1f}s × {N_REPEATS}, {N_PARTICLES} particles"
    )
    print("=" * 80)
    bench_pair("POMCPOW", make_pomcpow, env)
    bench_pair("PFT_DPW", make_pft_dpw, env)


if __name__ == "__main__":
    main()
