"""Lean sims/sec bench for ICVaR_PFT_DPW + ICVaR_POMCPOW on PacMan, ContinuousLaserTag, RockSample.

For each (planner, env) pair, drives ``planner._simulate_path`` directly for a
1.0 s wall-clock budget and reports the median sims/s across ``n_repeats``
trials (default 3). A 0.1 s warm-up populates per-action C++ kernel caches.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import ICVaR_PFT_DPW
from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import ICVaR_POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


DISCOUNT_FACTOR = 0.95
DEPTH = 20
EXPLORATION_CONSTANT = 1.0
K_O = 5.0
K_A = 5.0
ALPHA_O = 0.5
ALPHA_A = 0.5
MIN_VISIT_COUNT_PER_ACTION = 1
DELTA = 0.1
ALPHA = 0.1
MIN_IMM = -100.0
MAX_IMM = 100.0

TIME_BUDGET_S = 1.0
WARMUP_S = 0.1
N_PARTICLES = 100
N_REPEATS = 3


def build_envs() -> List[Tuple[str, Any]]:
    return [
        ("PacMan", PacManPOMDP()),
        ("ContinuousLaserTag", ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95)),
        ("RockSample", RockSamplePOMDP()),
    ]


def make_pft_dpw(env: Any) -> ICVaR_PFT_DPW:
    return ICVaR_PFT_DPW(
        environment=env,
        name="icvar_pft_dpw",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        time_out_in_seconds=int(TIME_BUDGET_S),
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        k_o=K_O,
        k_a=K_A,
        alpha_o=ALPHA_O,
        alpha_a=ALPHA_A,
        min_visit_count_per_action=MIN_VISIT_COUNT_PER_ACTION,
        delta=DELTA,
        alpha=ALPHA,
        min_immediate_cost=MIN_IMM,
        max_immediate_cost=MAX_IMM,
    )


def make_pomcpow(env: Any) -> ICVaR_POMCPOW:
    return ICVaR_POMCPOW(
        environment=env,
        name="icvar_pomcpow",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        time_out_in_seconds=int(TIME_BUDGET_S),
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        k_o=K_O,
        k_a=K_A,
        alpha_o=ALPHA_O,
        alpha_a=ALPHA_A,
        min_visit_count_per_action=MIN_VISIT_COUNT_PER_ACTION,
        delta=DELTA,
        alpha=ALPHA,
        min_immediate_cost=MIN_IMM,
        max_immediate_cost=MAX_IMM,
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


def bench_pair(
    planner_name: str,
    env_name: str,
    factory: Callable[[Any], Any],
    env: Any,
) -> Tuple[float, float]:
    np.random.seed(0)
    belief = get_initial_belief(env, n_particles=N_PARTICLES)
    planner = factory(env)
    run_for_budget(planner, belief, WARMUP_S)
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
        f"  {planner_name:<14} x {env_name:<22} "
        f"median={median:>8,.0f} sims/s   stdev={stdev:>7,.0f}   runs={[f'{r:.0f}' for r in rates]}"
    )
    return median, stdev


def main() -> None:
    print()
    print("=" * 88)
    print(
        f"  ICVaR sims/sec — {TIME_BUDGET_S:.1f}s budget × {N_REPEATS} runs, "
        f"{N_PARTICLES} particles, depth {DEPTH}"
    )
    print("=" * 88)
    pairs = [
        ("ICVaR_PFT_DPW", make_pft_dpw),
        ("ICVaR_POMCPOW", make_pomcpow),
    ]
    envs = build_envs()
    for planner_name, factory in pairs:
        for env_name, env in envs:
            bench_pair(planner_name, env_name, factory, env)
    print()


if __name__ == "__main__":
    main()
