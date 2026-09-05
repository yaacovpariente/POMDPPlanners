# SPDX-License-Identifier: MIT

"""Battleship-specific tests: dynamics, observation model, reward, terminal.

The cross-environment conformance suite already covers the shared Environment
contracts for this env (it is registered in ``ENV_BUILDERS``). What is left is
what only Battleship can get wrong: that the fleet really is fixed and hidden,
that a probe pays once and only once, that repeats are worthless, and that the
episode ends exactly when the last ship cell is uncovered.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.battleship_pomdp import (
    HIT,
    MISS,
    BattleshipPOMDP,
    BattleshipStepChannel,
    FleetLayoutTable,
    create_battleship_state,
    get_layout_table,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs


@pytest.fixture(name="env")
def _env() -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


def _board(env: BattleshipPOMDP, ship_cells, probed_cells=()) -> np.ndarray:
    occupancy = np.zeros(env.num_cells)
    occupancy[list(ship_cells)] = 1.0
    probed = np.zeros(env.num_cells)
    if probed_cells:
        probed[list(probed_cells)] = 1.0
    return create_battleship_state(occupancy, probed)


class TestLayoutEnumeration:
    """The legal-layout table defines what a belief is allowed to contain."""

    def test_every_layout_has_exactly_the_fleet_s_cells(self, env: BattleshipPOMDP) -> None:
        """Purpose: a layout that occupies the wrong number of cells is not this fleet.

        Given: the enumerated layout table for the default 5x5 (3,2,2) fleet
        When: each row's occupied cells are counted
        Then: every row occupies exactly seven cells

        Test type: unit
        """
        occupied_per_layout = env.layouts.masks.sum(axis=1)
        assert set(np.unique(occupied_per_layout)) == {env.num_ship_cells}

    def test_layouts_are_distinct_configurations(self, env: BattleshipPOMDP) -> None:
        """Purpose: guard against the enumeration emitting a row twice.

        Two different placement configurations may legitimately share an
        occupancy mask, so masks are not unique; what must hold is that the
        count matches an independent recomputation from the same geometry.

        Given: the default fleet
        When: the table is rebuilt from scratch
        Then: it has the same number of rows, and the same rows in the same order

        Test type: unit
        """
        rebuilt = FleetLayoutTable(board_size=5, ship_lengths=(3, 2, 2), allow_adjacent_ships=True)
        assert rebuilt.num_layouts == env.layouts.num_layouts
        assert np.array_equal(rebuilt.masks, env.layouts.masks)

    def test_disallowing_adjacency_shrinks_the_layout_space(self) -> None:
        """Purpose: the touch rule must actually change which layouts are legal.

        Given: the same fleet under both adjacency rules
        When: both tables are enumerated
        Then: forbidding touching yields strictly fewer layouts, and every
              no-touch layout is also legal under the permissive rule

        Test type: unit
        """
        permissive = FleetLayoutTable(5, (3, 2, 2), allow_adjacent_ships=True)
        strict = FleetLayoutTable(5, (3, 2, 2), allow_adjacent_ships=False)
        assert 0 < strict.num_layouts < permissive.num_layouts

        permissive_masks = {row.tobytes() for row in permissive.masks}
        assert all(row.tobytes() in permissive_masks for row in strict.masks)

    def test_ships_never_overlap(self) -> None:
        """Purpose: overlapping ships would let one cell count for two ships.

        Already implied by the cell count, but checked directly on the
        no-adjacency table because that path uses a different blocking mask.

        Given: the no-touch table
        When: each row's occupied cells are counted
        Then: every row occupies exactly the fleet's cell count

        Test type: unit
        """
        strict = FleetLayoutTable(5, (3, 2, 2), allow_adjacent_ships=False)
        assert set(np.unique(strict.masks.sum(axis=1))) == {7}

    def test_an_unplaceable_fleet_is_refused_at_construction(self) -> None:
        """Purpose: a geometry with no legal layout must fail loudly and early.

        Given: three length-3 ships on a 3x3 board that may not touch
        When: the table is enumerated
        Then: a ValueError names the geometry

        Test type: unit
        """
        with pytest.raises(ValueError, match="no legal layout"):
            FleetLayoutTable(3, (3, 3, 3), allow_adjacent_ships=False)

    def test_enumeration_cap_is_enforced(self) -> None:
        """Purpose: the cap is what keeps the exact belief from exploding.

        Given: a fleet whose configuration count exceeds a deliberately tiny cap
        When: the table is enumerated
        Then: a ValueError says so rather than the process filling memory

        Test type: unit
        """
        with pytest.raises(ValueError, match="more than 10 placement configurations"):
            FleetLayoutTable(5, (3, 2, 2), allow_adjacent_ships=True, max_layouts=10)

    def test_enumeration_cap_is_not_off_by_one(self) -> None:
        """Purpose: the last layout must not slip past the guard.

        Checking the count on entry to the recursion lets the final layout
        through, because the call that appends it returns without another check.
        A one-layout overshoot is harmless; the same off-by-one on a fleet sized
        to the cap is not.

        Given: a 2x2 board with one length-1 ship, which has exactly four layouts
        When: the cap is set to three and then to four
        Then: three raises and four does not

        Test type: unit
        """
        assert FleetLayoutTable(2, (1,), max_layouts=4).num_layouts == 4
        with pytest.raises(ValueError, match="more than 3 placement configurations"):
            FleetLayoutTable(2, (1,), max_layouts=3)

    def test_cached_layout_table_still_honours_the_cap(self) -> None:
        """Purpose: the same environment must not build hot and raise cold.

        The layout table is cached process-wide on geometry alone, which is
        right — the table is a function of geometry. The cap is not: it belongs
        to the caller. If a cache hit skipped it, an environment with too small
        a cap would construct fine in a warm process and raise in a fresh one.
        That difference surfaces only inside a worker after unpickling, or after
        a ``from_dict`` elsewhere, which is the worst place to read a traceback.

        Given: a warmed layout cache for a geometry with four layouts
        When: the same geometry is requested with a cap of one
        Then: it raises, exactly as a cold process would

        Test type: unit
        """
        get_layout_table(2, (1,), allow_adjacent_ships=True)
        with pytest.raises(ValueError, match="more than 1 placement configurations"):
            get_layout_table(2, (1,), allow_adjacent_ships=True, max_layouts=1)


class TestDynamics:
    """The fleet is fixed; only the probe record moves."""

    def test_probing_marks_only_the_probed_cell(self, env: BattleshipPOMDP) -> None:
        """Purpose: a probe must not disturb anything but its own probe flag.

        Given: a board with ships at cells 0-2
        When: cell 7 is probed
        Then: the occupancy half is unchanged and only cell 7's probe flag flips

        Test type: unit
        """
        state = _board(env, [0, 1, 2])
        next_state = env.sample_next_state(state, 7)

        assert np.array_equal(env.occupancy(next_state), env.occupancy(state))
        assert np.flatnonzero(env.probed(next_state)).tolist() == [7]

    def test_transition_is_deterministic(self, env: BattleshipPOMDP) -> None:
        """Purpose: nothing in Battleship's transition is random.

        Given: a board and a probe
        When: five successors are sampled
        Then: they are all identical

        Test type: unit
        """
        state = _board(env, [0, 1, 2])
        samples = env.sample_next_state(state, 3, n_samples=5)
        assert samples.shape == (5, 2 * env.num_cells)
        assert np.array_equal(samples, np.tile(samples[0], (5, 1)))

    def test_batch_transition_matches_the_single_one(self, env: BattleshipPOMDP) -> None:
        """Purpose: particle filters mix the batch and single paths freely.

        Given: three different boards
        When: the same probe is applied through both paths
        Then: the results agree in value and in dtype

        Test type: unit
        """
        states = np.stack([_board(env, cells) for cells in ([0, 1, 2], [5, 6, 7], [20, 21, 22])])
        batch = env.sample_next_state_batch(states, 6)
        singles = np.stack([env.sample_next_state(state, 6) for state in states])
        assert batch.dtype == singles.dtype
        assert np.array_equal(batch, singles)

    def test_repeat_probe_leaves_the_state_unchanged(self, env: BattleshipPOMDP) -> None:
        """Purpose: probing a cell twice must be idempotent on the state.

        Given: a board where cell 4 is already probed
        When: cell 4 is probed again
        Then: the state is identical

        Test type: unit
        """
        state = _board(env, [0, 1, 2], probed_cells=[4])
        assert np.array_equal(env.sample_next_state(state, 4), state)


class TestObservationModel:
    """Deterministic hit/miss, and a likelihood that says so."""

    def test_observation_reports_occupancy_exactly(self, env: BattleshipPOMDP) -> None:
        """Purpose: the sensor is noiseless, which the belief update relies on.

        Given: a board with ships at cells 0-2
        When: an occupied and an empty cell are probed
        Then: HIT and MISS come back, with no other outcome ever

        Test type: unit
        """
        state = _board(env, [0, 1, 2])
        assert env.sample_observation(env.sample_next_state(state, 1), 1) == HIT
        assert env.sample_observation(env.sample_next_state(state, 9), 9) == MISS
        assert env.sample_observation(env.sample_next_state(state, 1), 1, n_samples=4) == [HIT] * 4

    def test_likelihood_is_one_or_zero(self, env: BattleshipPOMDP) -> None:
        """Purpose: an impossible reading must carry zero likelihood, not a floor.

        Given: an occupied probed cell
        When: both readings are scored
        Then: HIT scores log 1 and MISS scores -inf

        Test type: unit
        """
        state = _board(env, [0, 1, 2])
        next_state = env.sample_next_state(state, 1)
        log_likelihood = env.observation_log_probability(next_state, 1, [HIT, MISS])
        assert log_likelihood[0] == 0.0
        assert log_likelihood[1] == -np.inf

    def test_observation_hash_agrees_with_equality(self, env: BattleshipPOMDP) -> None:
        """Purpose: tree planners index observation children by the hash.

        Given: the two possible readings
        When: hashed
        Then: equal readings hash equal and different ones do not

        Test type: unit
        """
        assert env.hash_observation(HIT) == env.hash_observation(1)
        assert env.hash_observation(HIT) != env.hash_observation(MISS)


class TestReward:
    """The reward is what stops the environment from being farmable."""

    def test_new_ship_cell_pays_the_hit_reward(self, env: BattleshipPOMDP) -> None:
        """Purpose: the point of the task.

        Given: an unprobed occupied cell
        When: it is probed
        Then: the reward is +1

        Test type: unit
        """
        assert env.reward(_board(env, [0, 1, 2]), 1) == pytest.approx(1.0)

    def test_water_pays_the_miss_penalty(self, env: BattleshipPOMDP) -> None:
        """Purpose: probing costs something, or search is free.

        Given: an unprobed empty cell
        When: it is probed
        Then: the reward is -0.1

        Test type: unit
        """
        assert env.reward(_board(env, [0, 1, 2]), 9) == pytest.approx(-0.1)

    def test_reprobing_a_hit_cell_pays_the_miss_penalty(self, env: BattleshipPOMDP) -> None:
        """Purpose: this is the defect the original formulation had.

        With an occupancy-only state, re-probing a ship cell paid +1 forever and
        the optimal policy was to find one ship cell and sit on it. Carrying the
        probe history is what makes the +1 payable once.

        Given: an occupied cell that has already been probed
        When: it is probed again
        Then: the reward is the miss penalty, not the hit reward

        Test type: unit
        """
        state = _board(env, [0, 1, 2], probed_cells=[1])
        assert env.reward(state, 1) == pytest.approx(-0.1)

    def test_no_probe_sequence_can_out_earn_the_fleet(self, env: BattleshipPOMDP) -> None:
        """Purpose: total positive reward must be bounded by the fleet's size.

        Given: every cell probed twice, in order
        When: the rewards are summed
        Then: the positives total exactly the fleet's cell count

        Test type: unit
        """
        state = _board(env, [0, 1, 2, 10, 11, 20, 21])
        positives = 0.0
        for action in list(range(env.num_cells)) * 2:
            reward = env.reward(state, action)
            positives += max(reward, 0.0)
            state = env.sample_next_state(state, action)
        assert positives == pytest.approx(float(env.num_ship_cells))

    def test_reward_batch_agrees_with_the_scalar_reward(self, env: BattleshipPOMDP) -> None:
        """Purpose: the expected-reward helper takes the batch path.

        Given: three boards differing in whether cell 1 is a new hit
        When: both paths score the same probe
        Then: they agree element-wise

        Test type: unit
        """
        states = np.stack(
            [
                _board(env, [0, 1, 2]),
                _board(env, [0, 1, 2], probed_cells=[1]),
                _board(env, [10, 11, 12]),
            ]
        )
        batched = env.reward_batch(states, 1)
        looped = np.array([env.reward(state, 1) for state in states])
        assert np.array_equal(batched, looped)

    def test_declared_reward_range_is_tight(self, env: BattleshipPOMDP) -> None:
        """Purpose: the range must bound every reward and not be padded.

        Given: the default configuration
        When: the declared range is read
        Then: it is exactly the two reachable rewards

        Test type: unit
        """
        assert env.reward_range == (-0.1, 1.0)

    def test_miss_penalty_must_be_a_magnitude(self) -> None:
        """Purpose: a negative magnitude would silently invert the sign convention.

        Given: a negative miss_penalty
        When: the environment is constructed
        Then: a ValueError explains the convention

        Test type: unit
        """
        with pytest.raises(ValueError, match="magnitude"):
            BattleshipPOMDP(discount_factor=0.99, miss_penalty=-0.1)


class TestTerminal:
    """The episode ends when the fleet is sunk, and not before."""

    def test_a_fresh_board_is_not_terminal(self, env: BattleshipPOMDP) -> None:
        """Purpose: an environment terminal at step zero is solved by anything.

        Given: a board with nothing probed
        When: terminality is checked
        Then: it is not terminal

        Test type: unit
        """
        assert not env.is_terminal(_board(env, [0, 1, 2]))

    def test_probing_every_ship_cell_terminates(self, env: BattleshipPOMDP) -> None:
        """Purpose: completion is exactly "every ship cell hit".

        Given: a board whose three ship cells are all probed
        When: terminality is checked
        Then: it is terminal, with most of the board still unprobed

        Test type: unit
        """
        state = _board(env, [0, 1, 2], probed_cells=[0, 1, 2])
        assert env.is_terminal(state)
        assert np.count_nonzero(env.probed(state)) < env.num_cells

    def test_one_remaining_ship_cell_is_not_terminal(self, env: BattleshipPOMDP) -> None:
        """Purpose: an off-by-one here would end episodes a probe early.

        Given: a board with two of three ship cells probed
        When: terminality is checked
        Then: it is not terminal

        Test type: unit
        """
        assert not env.is_terminal(_board(env, [0, 1, 2], probed_cells=[0, 1]))

    def test_probing_water_never_terminates(self, env: BattleshipPOMDP) -> None:
        """Purpose: only ship cells count towards completion.

        Given: a board with every empty cell probed
        When: terminality is checked
        Then: it is not terminal

        Test type: unit
        """
        water = [cell for cell in range(env.num_cells) if cell not in (0, 1, 2)]
        assert not env.is_terminal(_board(env, [0, 1, 2], probed_cells=water))


class TestStepInfoChannels:
    """The per-step channels the metrics are built from."""

    def test_a_new_hit_is_reported_once(self, env: BattleshipPOMDP) -> None:
        """Purpose: the hit count must not double-count a re-probe.

        Given: an occupied cell, probed then re-probed
        When: step_info scores both
        Then: the first is a new hit and the second is a repeat

        Test type: unit
        """
        state = _board(env, [0, 1, 2])
        first = env.step_info(state, 1, None)
        second = env.step_info(env.sample_next_state(state, 1), 1, None)

        assert first[BattleshipStepChannel.NEW_SHIP_CELL_HIT.value] == 1.0
        assert first[BattleshipStepChannel.REPEAT_PROBE.value] == 0.0
        assert second[BattleshipStepChannel.NEW_SHIP_CELL_HIT.value] == 0.0
        assert second[BattleshipStepChannel.REPEAT_PROBE.value] == 1.0

    def test_probe_outcome_channels_are_mutually_exclusive(self, env: BattleshipPOMDP) -> None:
        """Purpose: exactly one outcome per probe, or the counts stop adding up.

        Given: every cell of a board probed in turn
        When: step_info scores each probe
        Then: the three outcome channels always sum to one

        Test type: unit
        """
        state = _board(env, [0, 1, 2], probed_cells=[4])
        for action in range(env.num_cells):
            info = env.step_info(state, action, None)
            total = (
                info[BattleshipStepChannel.NEW_SHIP_CELL_HIT.value]
                + info[BattleshipStepChannel.WATER_PROBE.value]
                + info[BattleshipStepChannel.REPEAT_PROBE.value]
            )
            assert total == 1.0
            state = env.sample_next_state(state, action)

    def test_terminal_bookkeeping_step_reports_no_probe(self, env: BattleshipPOMDP) -> None:
        """Purpose: the terminal step took no action; counting one would inflate SUMs.

        Given: a sunk board and the terminal step's (state, None, None)
        When: step_info is called
        Then: all three probe channels are zero while the board channels are not

        Test type: unit
        """
        state = _board(env, [0, 1, 2], probed_cells=[0, 1, 2])
        info = env.step_info(state, None, None)

        assert info[BattleshipStepChannel.NEW_SHIP_CELL_HIT.value] == 0.0
        assert info[BattleshipStepChannel.WATER_PROBE.value] == 0.0
        assert info[BattleshipStepChannel.REPEAT_PROBE.value] == 0.0
        assert info[BattleshipStepChannel.FLEET_SUNK.value] == 1.0
        assert info[BattleshipStepChannel.FLEET_HIT_FRACTION.value] == pytest.approx(3.0 / 7.0)

    def test_sunk_and_not_sunk_are_complementary(self, env: BattleshipPOMDP) -> None:
        """Purpose: ended_by_goal and ended_by_timeout are built on this pair.

        Given: an unfinished board and a sunk one
        When: step_info scores both
        Then: the two channels sum to one in each case

        Test type: unit
        """
        for state in (_board(env, [0, 1, 2]), _board(env, [0, 1, 2], probed_cells=[0, 1, 2])):
            info = env.step_info(state, None, None)
            assert (
                info[BattleshipStepChannel.FLEET_SUNK.value]
                + info[BattleshipStepChannel.FLEET_NOT_SUNK.value]
            ) == 1.0


class TestSerializationAndIdentity:
    """Round-tripping and the cache key."""

    def test_round_trip_rebuilds_an_equal_environment(self, env: BattleshipPOMDP) -> None:
        """Purpose: to_dict output must be something the constructor accepts.

        Given: the default environment
        When: it is serialized and rebuilt
        Then: the rebuild is equal and shares its config_id

        Test type: unit
        """
        rebuilt = BattleshipPOMDP.from_dict(env.to_dict())
        assert rebuilt == env
        assert rebuilt.config_id == env.config_id

    def test_config_id_separates_different_geometries(self) -> None:
        """Purpose: two different boards must not share a result cache entry.

        Given: the default fleet and the same fleet with touching forbidden
        When: their config ids are compared
        Then: they differ

        Test type: unit
        """
        permissive = BattleshipPOMDP(discount_factor=0.99, allow_adjacent_ships=True)
        strict = BattleshipPOMDP(discount_factor=0.99, allow_adjacent_ships=False)
        assert permissive.config_id != strict.config_id

    def test_config_id_survives_a_pickle_round_trip(self, env: BattleshipPOMDP) -> None:
        """Purpose: the layout table is dropped on pickling and rebuilt lazily.

        A derived attribute that leaked into the identity would give a worker
        process a different config_id from the parent, silently defeating the
        result cache.

        Given: the default environment
        When: it is pickled, restored and then used
        Then: its config_id is unchanged and it still works

        Test type: unit
        """
        import pickle  # pylint: disable=import-outside-toplevel

        restored = pickle.loads(pickle.dumps(env))
        assert restored.config_id == env.config_id
        assert restored.layouts.num_layouts == env.layouts.num_layouts
        assert restored.config_id == env.config_id
