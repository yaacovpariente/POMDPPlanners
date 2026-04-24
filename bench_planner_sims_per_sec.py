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
    from POMDPPlanners.environments.tiger_pomdp import (  # pylint: disable=import-outside-toplevel
        TigerPOMDP,
    )

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
    from POMDPPlanners.environments.push_pomdp.push_pomdp import (  # pylint: disable=import-outside-toplevel
        PushPOMDP,
    )

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
        # pylint: disable-next=protected-access
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)
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


def _detect_backend(planner_class: Any) -> str:
    """Return 'arena' if the planner inherits from ArenaPathSimulationPolicy, else 'legacy'."""
    from POMDPPlanners.planners.planners_utils.path_simulations_policy_arena import (  # pylint: disable=import-outside-toplevel
        ArenaPathSimulationPolicy,
    )

    return "arena" if issubclass(planner_class, ArenaPathSimulationPolicy) else "legacy"


def bench_pomcpow(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """POMCPOW on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.pomcpow import (  # pylint: disable=import-outside-toplevel
        POMCPOW,
    )

    backend_kind = _detect_backend(POMCPOW)

    def make(env: Any) -> POMCPOW:
        return POMCPOW(
            environment=env,
            action_sampler=DiscreteActionSampler(env.get_actions()),
            name="pomcpow_bench",
            n_simulations=1,
            **PLANNER_PARAMS,
        )

    _bench_planner_across_envs("POMCPOW", make, backend_kind, time_budget_s, repeats)


def bench_pomcp_dpw(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """POMCP_DPW on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.pomcp_dpw import (  # pylint: disable=import-outside-toplevel
        POMCP_DPW,
    )

    backend_kind = _detect_backend(POMCP_DPW)

    def make(env: Any) -> POMCP_DPW:
        return POMCP_DPW(
            environment=env,
            action_sampler=DiscreteActionSampler(env.get_actions()),
            name="pomcp_dpw_bench",
            n_simulations=1,
            **PLANNER_PARAMS,
        )

    _bench_planner_across_envs("POMCP_DPW", make, backend_kind, time_budget_s, repeats)


def bench_pft_dpw(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """PFT-DPW on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.pft_dpw import (  # pylint: disable=import-outside-toplevel
        PFT_DPW,
    )

    backend_kind = _detect_backend(PFT_DPW)

    def make(env: Any) -> PFT_DPW:
        return PFT_DPW(
            environment=env,
            action_sampler=DiscreteActionSampler(env.get_actions()),
            name="pft_dpw_bench",
            n_simulations=1,
            **PLANNER_PARAMS,
        )

    _bench_planner_across_envs("PFT_DPW", make, backend_kind, time_budget_s, repeats)


def bench_sparse_pft(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """SparsePFT on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.sparse_pft import (  # pylint: disable=import-outside-toplevel
        SparsePFT,
    )

    backend_kind = _detect_backend(SparsePFT)

    def make(env: Any) -> SparsePFT:
        return SparsePFT(
            environment=env,
            discount_factor=PLANNER_PARAMS["discount_factor"],
            gamma=PLANNER_PARAMS["discount_factor"],
            depth=PLANNER_PARAMS["depth"],
            c_ucb=1.0,
            beta_ucb=2.0,
            belief_child_num=5,
            name="sparse_pft_bench",
            n_simulations=1,
        )

    _bench_planner_across_envs("SparsePFT", make, backend_kind, time_budget_s, repeats)


def bench_pomcp(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """POMCP on multiple envs (no progressive widening — uses the simpler config subset)."""
    from POMDPPlanners.planners.mcts_planners.pomcp import (  # pylint: disable=import-outside-toplevel
        POMCP,
    )

    backend_kind = _detect_backend(POMCP)

    def make(env: Any) -> POMCP:
        return POMCP(
            environment=env,
            discount_factor=PLANNER_PARAMS["discount_factor"],
            depth=PLANNER_PARAMS["depth"],
            exploration_constant=PLANNER_PARAMS["exploration_constant"],
            name="pomcp_bench",
            n_simulations=1,
        )

    _bench_planner_across_envs("POMCP", make, backend_kind, time_budget_s, repeats)


def bench_beta_zero(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """BetaZero on multiple envs. Uses an untrained network."""
    from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import (  # pylint: disable=import-outside-toplevel
        BetaZero,
    )

    backend_kind = _detect_backend(BetaZero)

    def make(env: Any) -> BetaZero:
        return BetaZero(
            environment=env,
            discount_factor=PLANNER_PARAMS["discount_factor"],
            depth=PLANNER_PARAMS["depth"],
            name="beta_zero_bench",
            action_sampler=DiscreteActionSampler(env.get_actions()),
            k_a=PLANNER_PARAMS["k_a"],
            alpha_a=PLANNER_PARAMS["alpha_a"],
            k_o=PLANNER_PARAMS["k_o"],
            alpha_o=PLANNER_PARAMS["alpha_o"],
            exploration_constant=PLANNER_PARAMS["exploration_constant"],
            n_simulations=1,
            state_dim=1,
        )

    _bench_planner_across_envs("BetaZero", make, backend_kind, time_budget_s, repeats)


def bench_icvar_pft_dpw(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """ICVaR_PFT_DPW on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import (  # pylint: disable=import-outside-toplevel
        ICVaR_PFT_DPW,
    )

    backend_kind = _detect_backend(ICVaR_PFT_DPW)

    def make(env: Any) -> ICVaR_PFT_DPW:
        return ICVaR_PFT_DPW(
            environment=env,
            discount_factor=PLANNER_PARAMS["discount_factor"],
            depth=PLANNER_PARAMS["depth"],
            exploration_constant=PLANNER_PARAMS["exploration_constant"],
            k_o=PLANNER_PARAMS["k_o"],
            k_a=PLANNER_PARAMS["k_a"],
            alpha_o=PLANNER_PARAMS["alpha_o"],
            alpha_a=PLANNER_PARAMS["alpha_a"],
            min_immediate_cost=-100.0,
            max_immediate_cost=100.0,
            min_visit_count_per_action=PLANNER_PARAMS["min_visit_count_per_action"],
            delta=0.1,
            name="icvar_pft_dpw_bench",
            action_sampler=DiscreteActionSampler(env.get_actions()),
            n_simulations=1,
            alpha=0.1,
        )

    _bench_planner_across_envs("ICVaR_PFT_DPW", make, backend_kind, time_budget_s, repeats)


def bench_icvar_pomcpow(time_budget_s: float = 1.0, repeats: int = 3) -> None:
    """ICVaR_POMCPOW on multiple envs."""
    from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import (  # pylint: disable=import-outside-toplevel
        ICVaR_POMCPOW,
    )

    backend_kind = _detect_backend(ICVaR_POMCPOW)

    def make(env: Any) -> ICVaR_POMCPOW:
        return ICVaR_POMCPOW(
            environment=env,
            discount_factor=PLANNER_PARAMS["discount_factor"],
            depth=PLANNER_PARAMS["depth"],
            exploration_constant=PLANNER_PARAMS["exploration_constant"],
            k_o=PLANNER_PARAMS["k_o"],
            k_a=PLANNER_PARAMS["k_a"],
            alpha_o=PLANNER_PARAMS["alpha_o"],
            alpha_a=PLANNER_PARAMS["alpha_a"],
            min_immediate_cost=-100.0,
            max_immediate_cost=100.0,
            min_visit_count_per_action=PLANNER_PARAMS["min_visit_count_per_action"],
            delta=0.1,
            name="icvar_pomcpow_bench",
            action_sampler=DiscreteActionSampler(env.get_actions()),
            n_simulations=1,
            alpha=0.1,
        )

    _bench_planner_across_envs("ICVaR_POMCPOW", make, backend_kind, time_budget_s, repeats)


def _bench_planner_across_envs(
    planner_name: str,
    factory: Callable[[Any], Any],
    backend_kind: str,
    time_budget_s: float,
    repeats: int,
) -> None:
    print()
    print("=" * 100)
    print(f" {planner_name} sims/sec — backend: {backend_kind}")
    print(f" budget {time_budget_s}s × {repeats} trials per env")
    print("=" * 100)
    envs = _try_build_envs()
    for env_name, env in envs:
        try:
            _bench_planner_on_env(
                planner_name=planner_name,
                planner_factory=factory,
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
    bench_pft_dpw(time_budget_s=1.0, repeats=3)
    bench_pomcp_dpw(time_budget_s=1.0, repeats=3)
    bench_sparse_pft(time_budget_s=1.0, repeats=3)
    bench_pomcp(time_budget_s=1.0, repeats=3)
    bench_icvar_pomcpow(time_budget_s=1.0, repeats=3)
    bench_icvar_pft_dpw(time_budget_s=1.0, repeats=3)


if __name__ == "__main__":
    main()
