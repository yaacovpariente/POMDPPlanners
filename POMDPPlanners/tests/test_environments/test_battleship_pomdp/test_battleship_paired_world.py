# SPDX-License-Identifier: MIT

"""Tests for the pinned-board world used by the paired Battleship QA pass.

The QA comparison is only paired if both arms start from the *same* hidden
fleet, and nothing in the simulator enforces that: it seeds each episode from
``md5(env_name, policy_name, episode_id)``, so a world that draws its start from
the RNG hands each policy a different board and the comparison silently becomes
unpaired. These tests pin the three properties the pairing actually rests on --
the start is deterministic, it is the layout the board id names, and each board
gets its own cache identity -- plus the one property that makes the whole thing
honest: the planner's prior is never narrowed to the pinned board.
"""

import random

import numpy as np
import pytest

from POMDPPlanners.environments.battleship_pomdp import BattleshipBelief, BattleshipPOMDP
from POMDPPlanners.tests.test_utils.battleship_qa_config import (
    BATTLESHIP_QA_BOARD_IDS,
    BattleshipFixedBoardWorld,
    battleship_qa_layout_index,
)


@pytest.fixture(name="world")
def _world() -> BattleshipFixedBoardWorld:
    return BattleshipFixedBoardWorld(board_id=0, discount_factor=0.99)


def test_initial_state_is_deterministic_under_any_global_seed(world):
    """The pinned start must not move when the episode seed moves.

    This is the property the pairing is built on: the two arms run under
    different ``md5``-derived seeds, so a start that consults the global RNG
    would differ between them.
    """
    draws = []
    for seed in (0, 1, 7, 2**31 - 1):
        random.seed(seed)
        np.random.seed(seed)
        draws.append(world.initial_state_dist().sample()[0])
    for other in draws[1:]:
        np.testing.assert_array_equal(draws[0], other)


def test_initial_state_is_the_named_layout_with_nothing_probed(world):
    """The start is exactly the layout the board id selects, unprobed."""
    state = world.initial_state_dist().sample()[0]
    np.testing.assert_array_equal(
        state[: world.num_cells], world.hidden_occupancy.astype(np.float64)
    )
    assert not state[world.num_cells :].any()
    assert world.layout_index == battleship_qa_layout_index(0, world.layouts.num_layouts)
    assert int(world.hidden_occupancy.sum()) == world.num_ship_cells


def test_sampled_states_are_independent_copies(world):
    """Mutating one draw must not corrupt the distribution or a later episode."""
    dist = world.initial_state_dist()
    first, second = dist.sample(2)
    first[:] = 0.0
    assert second.any()
    assert dist.sample()[0].any()


def test_probability_is_a_point_mass(world):
    """One on the pinned board, zero on anything else."""
    dist = world.initial_state_dist()
    pinned = dist.sample()[0]
    other = pinned.copy()
    other[world.num_cells] = 1.0  # a probed cell, so a different state
    np.testing.assert_allclose(dist.probability([pinned, other]), [1.0, 0.0])


def test_each_qa_board_is_a_distinct_layout_and_a_distinct_cache_entry():
    """Distinct boards, distinct names, distinct ``config_id``.

    A shared ``config_id`` would make twenty different boards collide onto one
    cache entry, so nineteen of the twenty episodes would be served from the
    first board's result and the whole comparison would be a single board.
    """
    worlds = [
        BattleshipFixedBoardWorld(board_id=b, discount_factor=0.99)
        for b in BATTLESHIP_QA_BOARD_IDS
    ]
    assert len({w.layout_index for w in worlds}) == len(worlds)
    assert len({w.name for w in worlds}) == len(worlds)
    assert len({w.config_id for w in worlds}) == len(worlds)


def test_world_config_id_differs_from_the_plain_environment(world):
    """The world must not share a cache identity with the planner's model."""
    assert world.config_id != BattleshipPOMDP(discount_factor=0.99).config_id


def test_dynamics_and_rewards_are_inherited_unchanged(world):
    """Pinning the start must not change what the game *is*."""
    plain = BattleshipPOMDP(discount_factor=0.99)
    assert world.reward_range == plain.reward_range
    assert world.get_actions() == plain.get_actions()
    state = world.initial_state_dist().sample()[0]
    action = int(np.flatnonzero(world.hidden_occupancy)[0])
    np.testing.assert_array_equal(
        world.sample_next_state(state, action), plain.sample_next_state(state, action)
    )
    assert world.reward(state, action) == plain.reward(state, action)
    assert world.is_terminal(state) == plain.is_terminal(state)


def test_pinning_the_world_does_not_narrow_the_planners_prior(world):
    """The fleet stays hidden.

    The pinned board lives on the world environment only; the belief handed to
    the planner is still the full uniform prior, so the planner cannot read the
    answer off its own particles.
    """
    random.seed(0)
    np.random.seed(0)
    belief = BattleshipBelief.from_environment(BattleshipPOMDP(discount_factor=0.99), 200)
    occupancies = {
        np.asarray(p, dtype=np.float64)[: world.num_cells].tobytes()
        for p in belief.particles
    }
    assert len(occupancies) > 1
    marginal = np.mean(
        [np.asarray(p, dtype=np.float64)[: world.num_cells] for p in belief.particles], axis=0
    )
    assert marginal.max() < 1.0
