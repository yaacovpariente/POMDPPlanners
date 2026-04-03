"""Layer 3: Combined planner+environment benchmarks.

Measures realistic end-to-end planning performance. Compare with Layer 1
(environment-only) and Layer 2 (planner-only) to attribute regressions.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

DISCOUNT = 0.95
N_SIMS = 500
DEPTH = 5
N_PARTICLES = 100
SEED = 42


def _run_planner(env, planner):
    np.random.seed(SEED)
    belief = get_initial_belief(pomdp=env, n_particles=N_PARTICLES)
    return planner.action(belief=belief)


# ---------------------------------------------------------------------------
# POMCP combinations
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="combined-pomcp")
def test_bench_pomcp_tiger(benchmark):
    """Benchmark POMCP on TigerPOMDP.

    Purpose: Measure end-to-end POMCP planning on the simplest discrete environment.

    Given: A POMCP planner with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = POMCP(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        exploration_constant=1.0,
        n_simulations=N_SIMS,
        name="bench",
    )
    benchmark(_run_planner, env, planner)


@pytest.mark.benchmark(group="combined-pomcp")
def test_bench_pomcp_rocksample(benchmark):
    """Benchmark POMCP on RockSamplePOMDP.

    Purpose: Measure end-to-end POMCP planning on a larger discrete environment.

    Given: A POMCP planner with 500 simulations on RockSamplePOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = RockSamplePOMDP(discount_factor=DISCOUNT)
    planner = POMCP(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        exploration_constant=1.0,
        n_simulations=N_SIMS,
        name="bench",
    )
    benchmark(_run_planner, env, planner)


@pytest.mark.benchmark(group="combined-pomcp")
def test_bench_pomcp_discrete_ld(benchmark):
    """Benchmark POMCP on DiscreteLightDarkPOMDP.

    Purpose: Measure end-to-end POMCP planning on a discrete-action environment.

    Given: A POMCP planner with 500 simulations on DiscreteLightDarkPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = DiscreteLightDarkPOMDP(discount_factor=DISCOUNT)
    planner = POMCP(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        exploration_constant=1.0,
        n_simulations=N_SIMS,
        name="bench",
    )
    benchmark(_run_planner, env, planner)


# ---------------------------------------------------------------------------
# SparsePFT combinations
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="combined-sparse_pft")
def test_bench_sparse_pft_tiger(benchmark):
    """Benchmark SparsePFT on TigerPOMDP.

    Purpose: Measure end-to-end SparsePFT planning on a simple discrete environment.

    Given: A SparsePFT planner with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparsePFT(
        environment=env,
        discount_factor=DISCOUNT,
        gamma=DISCOUNT,
        depth=DEPTH,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=5,
        n_simulations=N_SIMS,
        name="bench",
    )
    benchmark(_run_planner, env, planner)


@pytest.mark.benchmark(group="combined-sparse_pft")
def test_bench_sparse_pft_discrete_ld(benchmark):
    """Benchmark SparsePFT on DiscreteLightDarkPOMDP.

    Purpose: Measure end-to-end SparsePFT planning on a mixed observation space.

    Given: A SparsePFT planner with 500 simulations on DiscreteLightDarkPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = DiscreteLightDarkPOMDP(discount_factor=DISCOUNT)
    planner = SparsePFT(
        environment=env,
        discount_factor=DISCOUNT,
        gamma=DISCOUNT,
        depth=DEPTH,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=5,
        n_simulations=N_SIMS,
        name="bench",
    )
    benchmark(_run_planner, env, planner)


# ---------------------------------------------------------------------------
# DPW planner combinations (on DiscreteLightDarkPOMDP)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="combined-dpw")
def test_bench_pft_dpw_discrete_ld(benchmark):
    """Benchmark PFT_DPW on DiscreteLightDarkPOMDP.

    Purpose: Measure end-to-end PFT_DPW planning with progressive widening.

    Given: A PFT_DPW planner with 500 simulations on DiscreteLightDarkPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = DiscreteLightDarkPOMDP(discount_factor=DISCOUNT)
    actions = env.get_actions()
    planner = PFT_DPW(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        name="bench",
        action_sampler=DiscreteActionSampler(actions=actions),
        k_a=3.0,
        alpha_a=0.5,
        k_o=3.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=N_SIMS,
    )
    benchmark(_run_planner, env, planner)


@pytest.mark.benchmark(group="combined-dpw")
def test_bench_pomcpow_discrete_ld(benchmark):
    """Benchmark POMCPOW on DiscreteLightDarkPOMDP.

    Purpose: Measure end-to-end POMCPOW planning with weighted observations.

    Given: A POMCPOW planner with 500 simulations on DiscreteLightDarkPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = DiscreteLightDarkPOMDP(discount_factor=DISCOUNT)
    actions = env.get_actions()
    planner = POMCPOW(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="bench",
        action_sampler=DiscreteActionSampler(actions=actions),
        n_simulations=N_SIMS,
    )
    benchmark(_run_planner, env, planner)


@pytest.mark.benchmark(group="combined-dpw")
def test_bench_pomcp_dpw_discrete_ld(benchmark):
    """Benchmark POMCP_DPW on DiscreteLightDarkPOMDP.

    Purpose: Measure end-to-end POMCP_DPW planning with double progressive widening.

    Given: A POMCP_DPW planner with 500 simulations on DiscreteLightDarkPOMDP.
    When: planner.action is called with a fresh belief.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env = DiscreteLightDarkPOMDP(discount_factor=DISCOUNT)
    actions = env.get_actions()
    planner = POMCP_DPW(
        environment=env,
        discount_factor=DISCOUNT,
        depth=DEPTH,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="bench",
        action_sampler=DiscreteActionSampler(actions=actions),
        n_simulations=N_SIMS,
    )
    benchmark(_run_planner, env, planner)
