"""Layer 2: Planner-only benchmarks.

Measures planner logic using TigerPOMDP (the cheapest environment) so that
timing changes are attributable to planner code, not environment code.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

from POMDPPlanners.core.belief import get_initial_belief

SEED = 42
N_PARTICLES = 100


def _fresh_belief(env):
    np.random.seed(SEED)
    return get_initial_belief(pomdp=env, n_particles=N_PARTICLES)


@pytest.mark.benchmark(group="planner-action")
def test_bench_pomcp_action(benchmark, pomcp_planner, tiger_env):
    """Benchmark POMCP.action on TigerPOMDP.

    Purpose: Measure POMCP planning time in isolation from environment cost.

    Given: A POMCP planner configured with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief each round.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """

    def run():
        belief = _fresh_belief(tiger_env)
        return pomcp_planner.action(belief=belief)

    benchmark(run)


@pytest.mark.benchmark(group="planner-action")
def test_bench_sparse_pft_action(benchmark, sparse_pft_planner, tiger_env):
    """Benchmark SparsePFT.action on TigerPOMDP.

    Purpose: Measure SparsePFT planning time in isolation from environment cost.

    Given: A SparsePFT planner configured with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief each round.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """

    def run():
        belief = _fresh_belief(tiger_env)
        return sparse_pft_planner.action(belief=belief)

    benchmark(run)


@pytest.mark.benchmark(group="planner-action")
def test_bench_pft_dpw_action(benchmark, pft_dpw_planner, tiger_env):
    """Benchmark PFT_DPW.action on TigerPOMDP.

    Purpose: Measure PFT_DPW planning time in isolation from environment cost.

    Given: A PFT_DPW planner configured with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief each round.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """

    def run():
        belief = _fresh_belief(tiger_env)
        return pft_dpw_planner.action(belief=belief)

    benchmark(run)


@pytest.mark.benchmark(group="planner-action")
def test_bench_pomcpow_action(benchmark, pomcpow_planner, tiger_env):
    """Benchmark POMCPOW.action on TigerPOMDP.

    Purpose: Measure POMCPOW planning time in isolation from environment cost.

    Given: A POMCPOW planner configured with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief each round.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """

    def run():
        belief = _fresh_belief(tiger_env)
        return pomcpow_planner.action(belief=belief)

    benchmark(run)


@pytest.mark.benchmark(group="planner-action")
def test_bench_pomcp_dpw_action(benchmark, pomcp_dpw_planner, tiger_env):
    """Benchmark POMCP_DPW.action on TigerPOMDP.

    Purpose: Measure POMCP_DPW planning time in isolation from environment cost.

    Given: A POMCP_DPW planner configured with 500 simulations on TigerPOMDP.
    When: planner.action is called with a fresh belief each round.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """

    def run():
        belief = _fresh_belief(tiger_env)
        return pomcp_dpw_planner.action(belief=belief)

    benchmark(run)
