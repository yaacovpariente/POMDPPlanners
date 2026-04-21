"""Benchmark fixtures for performance testing.

Provides pre-built environments, planners, beliefs, and initial states
for the three benchmark layers: environment-only, planner-only, and combined.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

DISCOUNT_FACTOR = 0.95
N_SIMULATIONS = 500
DEPTH = 5
N_PARTICLES = 100
SEED = 42


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiger_env():
    """TigerPOMDP environment for benchmarking."""
    return TigerPOMDP(discount_factor=DISCOUNT_FACTOR)


@pytest.fixture
def discrete_ld_env():
    """DiscreteLightDarkPOMDP environment for benchmarking."""
    return DiscreteLightDarkPOMDP(discount_factor=DISCOUNT_FACTOR)


@pytest.fixture
def continuous_ld_env():
    """ContinuousLightDarkPOMDPDiscreteActions environment for benchmarking."""
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=DISCOUNT_FACTOR)


@pytest.fixture
def rock_sample_env():
    """RockSamplePOMDP environment for benchmarking."""
    return RockSamplePOMDP(discount_factor=DISCOUNT_FACTOR)


@pytest.fixture
def laser_tag_env():
    """LaserTagPOMDP environment for benchmarking."""
    return LaserTagPOMDP(discount_factor=DISCOUNT_FACTOR)


@pytest.fixture
def mountain_car_env():
    """MountainCarPOMDP environment for benchmarking."""
    return MountainCarPOMDP(discount_factor=DISCOUNT_FACTOR)


# ---------------------------------------------------------------------------
# State and action fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiger_state_action(tiger_env):
    """Initial state and action for TigerPOMDP."""
    np.random.seed(SEED)
    state = tiger_env.initial_state_dist().sample()[0]
    action = tiger_env.get_actions()[0]
    return tiger_env, state, action


@pytest.fixture
def discrete_ld_state_action(discrete_ld_env):
    """Initial state and action for DiscreteLightDarkPOMDP."""
    np.random.seed(SEED)
    state = discrete_ld_env.initial_state_dist().sample()[0]
    action = discrete_ld_env.get_actions()[0]
    return discrete_ld_env, state, action


@pytest.fixture
def continuous_ld_state_action(continuous_ld_env):
    """Initial state and action for ContinuousLightDarkPOMDPDiscreteActions."""
    np.random.seed(SEED)
    state = continuous_ld_env.initial_state_dist().sample()[0]
    action = continuous_ld_env.get_actions()[0]
    return continuous_ld_env, state, action


@pytest.fixture
def rock_sample_state_action(rock_sample_env):
    """Initial state and action for RockSamplePOMDP."""
    np.random.seed(SEED)
    state = rock_sample_env.initial_state_dist().sample()[0]
    action = rock_sample_env.get_actions()[0]
    return rock_sample_env, state, action


@pytest.fixture
def laser_tag_state_action(laser_tag_env):
    """Initial state and action for LaserTagPOMDP."""
    np.random.seed(SEED)
    state = laser_tag_env.initial_state_dist().sample()[0]
    action = laser_tag_env.get_actions()[0]
    return laser_tag_env, state, action


@pytest.fixture
def mountain_car_state_action(mountain_car_env):
    """Initial state and action for MountainCarPOMDP."""
    np.random.seed(SEED)
    state = mountain_car_env.initial_state_dist().sample()[0]
    action = mountain_car_env.get_actions()[0]
    return mountain_car_env, state, action


# ---------------------------------------------------------------------------
# Belief fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiger_belief(tiger_env):
    """Initial belief for TigerPOMDP."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=tiger_env, n_particles=N_PARTICLES)


@pytest.fixture
def discrete_ld_belief(discrete_ld_env):
    """Initial belief for DiscreteLightDarkPOMDP."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=discrete_ld_env, n_particles=N_PARTICLES)


@pytest.fixture
def continuous_ld_belief(continuous_ld_env):
    """Initial belief for ContinuousLightDarkPOMDPDiscreteActions."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=continuous_ld_env, n_particles=N_PARTICLES)


@pytest.fixture
def rock_sample_belief(rock_sample_env):
    """Initial belief for RockSamplePOMDP."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=rock_sample_env, n_particles=N_PARTICLES)


@pytest.fixture
def laser_tag_belief(laser_tag_env):
    """Initial belief for LaserTagPOMDP."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=laser_tag_env, n_particles=N_PARTICLES)


@pytest.fixture
def mountain_car_belief(mountain_car_env):
    """Initial belief for MountainCarPOMDP."""
    np.random.seed(SEED)
    return get_initial_belief(pomdp=mountain_car_env, n_particles=N_PARTICLES)


# ---------------------------------------------------------------------------
# Planner fixtures (all on TigerPOMDP for Layer 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def pomcp_planner(tiger_env):
    """POMCP planner on TigerPOMDP."""
    return POMCP(
        environment=tiger_env,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=1.0,
        n_simulations=N_SIMULATIONS,
        name="bench_pomcp",
    )


@pytest.fixture
def sparse_pft_planner(tiger_env):
    """SparsePFT planner on TigerPOMDP."""
    return SparsePFT(
        environment=tiger_env,
        discount_factor=DISCOUNT_FACTOR,
        gamma=DISCOUNT_FACTOR,
        depth=DEPTH,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=5,
        n_simulations=N_SIMULATIONS,
        name="bench_sparse_pft",
    )


@pytest.fixture
def pft_dpw_planner(tiger_env):
    """PFT_DPW planner on TigerPOMDP."""
    actions = tiger_env.get_actions()
    return PFT_DPW(
        environment=tiger_env,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        name="bench_pft_dpw",
        action_sampler=DiscreteActionSampler(actions=actions),
        k_a=3.0,
        alpha_a=0.5,
        k_o=3.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=N_SIMULATIONS,
    )


@pytest.fixture
def pomcpow_planner(tiger_env):
    """POMCPOW planner on TigerPOMDP."""
    actions = tiger_env.get_actions()
    return POMCPOW(
        environment=tiger_env,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="bench_pomcpow",
        action_sampler=DiscreteActionSampler(actions=actions),
        n_simulations=N_SIMULATIONS,
    )


@pytest.fixture
def pomcp_dpw_planner(tiger_env):
    """POMCP_DPW planner on TigerPOMDP."""
    actions = tiger_env.get_actions()
    return POMCP_DPW(
        environment=tiger_env,
        discount_factor=DISCOUNT_FACTOR,
        depth=DEPTH,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="bench_pomcp_dpw",
        action_sampler=DiscreteActionSampler(actions=actions),
        n_simulations=N_SIMULATIONS,
    )
