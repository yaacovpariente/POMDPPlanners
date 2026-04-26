"""Sims/sec bench for the 5 envs targeted by the kernel-cache extension PR.

Drives ``planner._simulate_path`` directly under a 1.0 s wall-clock budget per
trial and reports the median across 3 trials. Each (planner, env) pair gets a
0.1 s warm-up to populate per-action C++ kernel caches.

Envs covered:
- CartPole
- MountainCar
- LaserTag (discrete grid)
- DiscreteLightDark
- Push (discrete)

Planners: POMCPOW (single-state path -- exercises ``sample_next_state`` /
``sample_observation`` / ``observation_log_probability``) and PFT-DPW
(belief-update path -- exercises ``sample_next_state_batch`` /
``observation_log_probability_per_state`` / ``reward_batch``).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


DISCOUNT_FACTOR = 0.95
DEPTH = 20
EXPLORATION_CONSTANT = 1.0
K_O = 5.0
K_A = 5.0
ALPHA_O = 0.5
ALPHA_A = 0.5
MIN_VISIT_COUNT_PER_ACTION = 1

TIME_BUDGET_S = 1.0
WARMUP_S = 0.1
N_PARTICLES = 100
N_REPEATS = 3


def build_envs() -> List[Tuple[str, Any]]:
    cartpole_noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    return [
        ("CartPole", CartPolePOMDP(discount_factor=DISCOUNT_FACTOR, noise_cov=cartpole_noise_cov)),
        ("MountainCar", MountainCarPOMDP(discount_factor=DISCOUNT_FACTOR)),
        ("LaserTagDiscrete", LaserTagPOMDP(discount_factor=DISCOUNT_FACTOR)),
        ("DiscreteLightDark", DiscreteLightDarkPOMDP(discount_factor=DISCOUNT_FACTOR)),
        ("PushDiscrete", PushPOMDP(discount_factor=DISCOUNT_FACTOR)),
    ]


def make_pomcpow(env: Any) -> POMCPOW:
    return POMCPOW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pomcpow_bench",
        n_simulations=1,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        k_o=K_O,
        k_a=K_A,
        alpha_o=ALPHA_O,
        alpha_a=ALPHA_A,
        min_visit_count_per_action=MIN_VISIT_COUNT_PER_ACTION,
    )


def make_pft_dpw(env: Any) -> PFT_DPW:
    return PFT_DPW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pft_dpw_bench",
        n_simulations=1,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        k_o=K_O,
        k_a=K_A,
        alpha_o=ALPHA_O,
        alpha_a=ALPHA_A,
        min_visit_count_per_action=MIN_VISIT_COUNT_PER_ACTION,
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
        f"  {planner_name:<8} x {env_name:<22} "
        f"median={median:>8,.0f} sims/s   stdev={stdev:>7,.0f}   runs={[f'{r:.0f}' for r in rates]}"
    )
    return median, stdev


def main() -> None:
    print()
    print("=" * 88)
    print(
        f"  Sims/sec bench (extension PR) — {TIME_BUDGET_S:.1f}s × {N_REPEATS} runs, "
        f"{N_PARTICLES} particles, depth {DEPTH}"
    )
    print("=" * 88)
    pairs: List[Tuple[str, Callable[[Any], Any]]] = [
        ("POMCPOW", make_pomcpow),
        ("PFT_DPW", make_pft_dpw),
    ]
    envs = build_envs()
    for planner_name, factory in pairs:
        for env_name, env in envs:
            try:
                bench_pair(planner_name, env_name, factory, env)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                import traceback  # pylint: disable=import-outside-toplevel

                print(f"  [skip] {planner_name} x {env_name}: {exc}")
                traceback.print_exc()
    print()


if __name__ == "__main__":
    main()
