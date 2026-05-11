"""Headline sims/sec bench for PFT-DPW native-rollout migration.

Times how many calls to ``planner.action(belief)`` finish in a 1.0 s budget
on each (planner, env) combination. Reports total root-visit count divided
by total wall time, averaged across N=5 calls per cell.

Cases:
  planners: POMCPOW, PFT_DPW
  envs:     ContinuousLightDark (DiscreteActions variant), LaserTag, Push
  N=5 calls, 1.0 s budget per call, 200 particles, seed=42

Usage:
    python bench_pft_dpw_native_rollout.py > bench_before.txt
    # (apply the patch)
    python bench_pft_dpw_native_rollout.py > bench_after.txt

The bench measures wall time only (no cProfile) so headline numbers are
not distorted by tracing overhead.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


SEED = 42
N_PARTICLES = 200
N_CALLS = 5
TIME_BUDGET_S = 1.0

PLANNER_PARAMS: Dict[str, Any] = {
    "discount_factor": 0.95,
    "depth": 20,
    "exploration_constant": 1.0,
    "k_o": 5.0,
    "k_a": 5.0,
    "alpha_o": 0.5,
    "alpha_a": 0.5,
    "min_visit_count_per_action": 1,
}


def _build_continuous_light_dark() -> Any:
    # Imported lazily so that pyright treats the env as Any (its concrete
    # subclass is structurally complete but the abstract-class checker still
    # flags direct construction at import-time).
    # pylint: disable=import-outside-toplevel
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDPDiscreteActions,
    )

    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


def _build_laser_tag() -> Any:
    # pylint: disable=import-outside-toplevel
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP

    return LaserTagPOMDP(discount_factor=0.95)


def _build_push() -> Any:
    # pylint: disable=import-outside-toplevel
    from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP

    return PushPOMDP(discount_factor=0.95)


def _build_envs() -> List[Tuple[str, Any]]:
    return [
        ("ContinuousLightDark", _build_continuous_light_dark()),
        ("LaserTag", _build_laser_tag()),
        ("Push", _build_push()),
    ]


def _make_pomcpow(env: Any) -> POMCPOW:
    return POMCPOW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pomcpow_bench",
        time_out_in_seconds=int(TIME_BUDGET_S),
        n_simulations=None,
        **PLANNER_PARAMS,
    )


def _make_pft_dpw(env: Any) -> PFT_DPW:
    return PFT_DPW(
        environment=env,
        action_sampler=DiscreteActionSampler(env.get_actions()),
        name="pft_dpw_bench",
        time_out_in_seconds=int(TIME_BUDGET_S),
        n_simulations=None,
        **PLANNER_PARAMS,
    )


def _root_visit_count(run_data: Any) -> int:
    for var in run_data.info_variables:
        if var.name == "root_visit_count":
            return int(var.value)
    return 0


def _run_calls(planner: Any, belief: Any, n_calls: int) -> Tuple[int, float]:
    """Run ``planner.action(belief)`` ``n_calls`` times. Return (total_root_visits, total_wall_s)."""
    total_visits = 0
    total_wall = 0.0
    for _ in range(n_calls):
        t0 = time.perf_counter()
        _, run_data = planner.action(belief)
        elapsed = time.perf_counter() - t0
        total_wall += elapsed
        total_visits += _root_visit_count(run_data)
    return total_visits, total_wall


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _bench_cell(
    planner_name: str,
    planner_factory: Callable[[Any], Any],
    env_name: str,
    env: Any,
) -> float:
    _seed_all(SEED)
    belief = get_initial_belief(env, n_particles=N_PARTICLES)
    planner = planner_factory(env)

    # Warmup: one call to amortise import / JIT effects.
    planner.action(belief)

    visits, wall = _run_calls(planner, belief, N_CALLS)
    sims_per_sec = visits / wall if wall > 0 else 0.0
    print(
        f"  {planner_name:<10} {env_name:<22} "
        f"visits={visits:>10,d}  wall={wall:>6.3f}s  "
        f"sims/sec={sims_per_sec:>10,.0f}"
    )
    return sims_per_sec


def main() -> None:
    print("=" * 90)
    print(" PFT-DPW native-rollout bench")
    print(
        f" seed={SEED}  N_calls={N_CALLS}  budget={TIME_BUDGET_S}s/call  "
        f"particles={N_PARTICLES}"
    )
    print("=" * 90)

    envs = _build_envs()
    planners: List[Tuple[str, Callable[[Any], Any]]] = [
        ("POMCPOW", _make_pomcpow),
        ("PFT_DPW", _make_pft_dpw),
    ]

    results: List[Tuple[str, str, float]] = []
    for planner_name, factory in planners:
        for env_name, env in envs:
            sps = _bench_cell(planner_name, factory, env_name, env)
            results.append((planner_name, env_name, sps))

    print()
    print("Markdown:")
    print("| planner | env | sims/sec |")
    print("|---|---|---|")
    for planner_name, env_name, sps in results:
        print(f"| {planner_name} | {env_name} | {sps:,.0f} |")


if __name__ == "__main__":
    main()
