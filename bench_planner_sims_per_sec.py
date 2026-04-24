"""Generic sims/sec benchmark for any MCTS planner.

Times how many calls to ``planner.action(belief)`` finish in a given
budget on a given env. Used to capture before/after numbers when
migrating planners to the arena tree backend.

Run:
    python bench_planner_sims_per_sec.py
"""

from __future__ import annotations

import statistics
import time
import traceback
from typing import Any, Callable, Dict, List, Tuple

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


PLANNER_PARAMS: Dict[str, Any] = dict(
    discount_factor=0.95,
    depth=20,
    exploration_constant=1.0,
    k_o=5.0,
    k_a=5.0,
    alpha_o=0.5,
    alpha_a=0.5,
    min_visit_count_per_action=1,
)


def _try_build_envs() -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for name, factory in (
        ("Tiger", _build_tiger),
        ("ContinuousLightDarkDiscreteActions", _build_continuous_light_dark_discrete),
        ("LaserTag", _build_laser_tag),
        ("Push", _build_push),
    ):
        try:
            env = factory()
            out.append((name, env))
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"[skip] {name}: failed to construct env")
            traceback.print_exc()
    return out


def _build_tiger():
    from POMDPPlanners.environments.tiger_pomdp import (
        TigerPOMDP,
    )  # pylint: disable=import-outside-toplevel

    return TigerPOMDP(discount_factor=0.95)


def _build_continuous_light_dark_discrete():
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (  # pylint: disable=import-outside-toplevel
        ContinuousLightDarkPOMDPDiscreteActions,
    )

    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


def _build_laser_tag():
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (  # pylint: disable=import-outside-toplevel
        LaserTagPOMDP,
    )

    return LaserTagPOMDP(discount_factor=0.95)


def _build_push():
    from POMDPPlanners.environments.push_pomdp.push_pomdp import (
        PushPOMDP,
    )  # pylint: disable=import-outside-toplevel

    return PushPOMDP(discount_factor=0.95)


def _time_simulate_path_legacy(
    planner: Any, initial_belief: Any, time_budget_s: float
) -> Tuple[int, float]:
    """Drive planner._simulate_path(belief_node, depth=0) tightly. For legacy node-based planners."""
    from POMDPPlanners.core.tree import BeliefNode  # pylint: disable=import-outside-toplevel

    root = BeliefNode(belief=initial_belief)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + time_budget_s
    while time.perf_counter() < deadline:
        planner._simulate_path(belief_node=root, depth=0)  # pylint: disable=protected-access
        n += 1
    return n, time.perf_counter() - t0


def _time_simulate_path_arena(
    planner: Any, initial_belief: Any, time_budget_s: float
) -> Tuple[int, float]:
    """Drive planner._simulate_path(tree, belief_id, depth=0) tightly. For arena-tree planners."""
    from POMDPPlanners.core.tree.arena import Tree  # pylint: disable=import-outside-toplevel

    tree = Tree()
    root_id = tree.add_belief_node(initial_belief)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + time_budget_s
    while time.perf_counter() < deadline:
        planner._simulate_path(
            tree=tree, belief_id=root_id, depth=0
        )  # pylint: disable=protected-access
        n += 1
    return n, time.perf_counter() - t0


def _median(xs: List[float]) -> float:
    return statistics.median(xs)


def _bench_planner_on_env(
    planner_name: str,
    planner_factory: Callable[[Any], Any],
    env_name: str,
    env: Any,
    backend_kind: str,  # "legacy" or "arena"
    time_budget_s: float,
    repeats: int,
) -> Tuple[float, float]:
    """Run the planner on ``env`` ``repeats`` times for ``time_budget_s`` each. Return (median, stdev)."""
    initial_belief = get_initial_belief(env, n_particles=100)
    planner = planner_factory(env)

    runs: List[float] = []
    for _ in range(repeats):
        if backend_kind == "legacy":
            n, t = _time_simulate_path_legacy(planner, initial_belief, time_budget_s)
        elif backend_kind == "arena":
            n, t = _time_simulate_path_arena(planner, initial_belief, time_budget_s)
        else:
            raise ValueError(f"unknown backend_kind: {backend_kind}")
        runs.append(n / t)
    median = _median(runs)
    stdev = statistics.stdev(runs) if len(runs) > 1 else 0.0
    print(
        f"  {env_name:<35} {planner_name:<15} {backend_kind:<7} "
        f"median={median:>10,.0f} sims/s  stdev={stdev:>8,.0f}"
    )
    return median, stdev


def bench_pomcpow(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """POMCPOW (current state in the codebase) on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.pomcpow import (
        POMCPOW,
    )  # pylint: disable=import-outside-toplevel

    # Detect backend by inspecting the planner's MRO
    from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (  # pylint: disable=import-outside-toplevel
        ArenaPathSimulationPolicy,
    )

    is_arena = issubclass(POMCPOW, ArenaPathSimulationPolicy)
    backend_kind = "arena" if is_arena else "legacy"

    def make(env: Any) -> POMCPOW:
        return POMCPOW(
            environment=env,
            action_sampler=DiscreteActionSampler(env.get_actions()),
            name="pomcpow_bench",
            n_simulations=1,
            **PLANNER_PARAMS,
        )

    print()
    print("=" * 100)
    print(f" POMCPOW sims/sec — backend: {backend_kind}")
    print(f" budget {time_budget_s}s × {repeats} trials per env, params={PLANNER_PARAMS}")
    print("=" * 100)
    envs = _try_build_envs()
    for env_name, env in envs:
        try:
            _bench_planner_on_env(
                planner_name="POMCPOW",
                planner_factory=make,
                env_name=env_name,
                env=env,
                backend_kind=backend_kind,
                time_budget_s=time_budget_s,
                repeats=repeats,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"  [skip] {env_name}: bench failed")
            traceback.print_exc()


def main() -> None:
    bench_pomcpow(time_budget_s=1.0, repeats=3)


if __name__ == "__main__":
    main()
