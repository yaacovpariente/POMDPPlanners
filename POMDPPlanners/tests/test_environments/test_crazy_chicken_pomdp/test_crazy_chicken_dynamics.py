# SPDX-License-Identifier: MIT

"""Dynamics, reward and termination of the Crazy Chicken POMDP.

The cross-environment conformance suite covers the shared contracts once the
environment is in ``ENV_BUILDERS``. What it cannot cover is whether this
environment's rules are the rules the design says they are, which is what this
file checks: every transition rule is exercised from a hand-built world where
the expected successor is written out rather than derived from the code under
test.

Every test here pins ``dive_probability`` to 0 or 1 where the dive coin would
otherwise decide the outcome, so a failure is a rule that changed rather than a
draw that went the other way.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.crazy_chicken_pomdp import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    MODE_DIVE,
    MODE_PATROL,
    SHIP_HIT_INDEX,
    CrazyChickenAction,
    CrazyChickenPOMDP,
    create_crazy_chicken_state,
)


def build_env(**overrides):
    """Build a small, fully deterministic world for a rule test."""
    settings = {
        "num_columns": 5,
        "num_rows": 4,
        "num_chickens": 2,
        "dive_probability": 0.0,
        "fire_cooldown": 0,
        "max_steps": 50,
        "discount_factor": 0.95,
    }
    settings.update(overrides)
    return CrazyChickenPOMDP(**settings)


def test_patrol_chicken_steps_along_its_direction():
    """A patrolling chicken moves one column the way it is walking.

    Purpose: The sideways walk is the only motion a non-diving chicken has, and
        every camera reading is a reading of where that walk left it.

    Given: A chicken at column 1 walking right, with diving switched off.
    When: The ship holds position for one step.
    Then: The chicken is at column 2, still on its row, still patrolling.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[1, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 0]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    chicken = env.chickens(successor)[0]
    assert chicken[CHICKEN_COLUMN] == 2
    assert chicken[CHICKEN_ROW] == 3
    assert chicken[CHICKEN_MODE] == MODE_PATROL


def test_patrol_chicken_bounces_off_the_wall_and_moves_in_the_same_step():
    """A chicken walking into a wall turns and takes its step, not one or the other.

    Purpose: Reversing without moving would park a chicken against a wall for a
        step, which is a different world from the one the design describes and
        one a planner could exploit by camping the edge column.

    Given: A chicken in the last column walking right.
    When: One step is taken.
    Then: It is one column back from the wall and now walking left.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[4, 3, 1, MODE_PATROL, 1], [0, 3, -1, MODE_PATROL, 0]]
    )
    chicken = env.chickens(env.sample_next_state(state, int(CrazyChickenAction.STAY)))[0]
    assert chicken[CHICKEN_COLUMN] == 3
    assert chicken[CHICKEN_DIRECTION] == -1


def test_dive_drops_one_row_and_holds_its_column():
    """A diving chicken falls straight down.

    Purpose: The radar's row reading is only informative if a dive is vertical;
        a dive that also drifted sideways would make the camera's column
        reading stale at the exact moment it matters most.

    Given: A diving chicken at column 2, row 3.
    When: One step is taken.
    Then: It is at column 2, row 2.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[2, 3, 1, MODE_DIVE, 1], [0, 3, -1, MODE_PATROL, 0]]
    )
    chicken = env.chickens(env.sample_next_state(state, int(CrazyChickenAction.STAY)))[0]
    assert (chicken[CHICKEN_COLUMN], chicken[CHICKEN_ROW]) == (2, 2)


def test_dive_coin_is_flipped_before_the_chickens_move():
    """A chicken that switches into a dive this step also drops this step.

    Purpose: This is a documented choice, not an accident. Flipping the coin
        afterwards would delay every dive by a step, which changes how much
        warning the ship gets and therefore the whole difficulty of the task.

    Given: A patrolling chicken and ``dive_probability`` of 1.
    When: One step is taken.
    Then: It is diving *and* one row lower, rather than diving on its old row.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    state = create_crazy_chicken_state(
        env, chickens=[[2, 3, 1, MODE_PATROL, 1], [0, 3, -1, MODE_PATROL, 0]]
    )
    chicken = env.chickens(env.sample_next_state(state, int(CrazyChickenAction.STAY)))[0]
    assert chicken[CHICKEN_MODE] == MODE_DIVE
    assert chicken[CHICKEN_ROW] == 2


def test_fire_kills_the_lowest_chicken_in_the_ships_column_at_any_range():
    """A shot resolves in the same step, at unlimited range, from the bottom up.

    Purpose: This is the whole hitscan rule. A shot that travelled would leave
        the top chicken alive for several steps and let either of them walk out
        of the column meanwhile; here the ship gets exactly what it aimed at.

    Given: Two chickens stacked in the ship's column, at rows 1 and 3, and a
        third elsewhere.
    When: The ship fires.
    Then: Only the row-1 chicken dies, whatever the range to the other.

    Test type: unit
    """
    env = build_env(num_rows=5, num_chickens=3)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env,
        chickens=[
            [column, 3, 1, MODE_PATROL, 1],
            [column, 1, 1, MODE_PATROL, 1],
            [0, 4, 1, MODE_PATROL, 1],
        ],
    )
    assert env.shot_target(state) == 1
    flock = env.chickens(env.sample_next_state(state, int(CrazyChickenAction.FIRE)))
    assert flock[1][CHICKEN_ALIVE] == 0.0
    assert flock[0][CHICKEN_ALIVE] == 1.0
    assert flock[2][CHICKEN_ALIVE] == 1.0


def test_fire_into_an_empty_column_misses_and_only_costs_the_shot():
    """A shot with nothing above it is charged and buys nothing.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]], ship_column=2
    )
    assert env.shot_target(state) == -1
    assert env.fires(state, int(CrazyChickenAction.FIRE))
    successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert env.live_chicken_count(successor) == 2
    assert env.reward(state, int(CrazyChickenAction.FIRE), successor) == pytest.approx(
        -env.step_cost - env.shot_cost
    )


def test_the_shot_is_resolved_before_the_chickens_move():
    """The ship hits where the flock was, not where it goes.

    Purpose: This ordering is the point of the redesign. Resolving the shot
        after the step would let a chicken step sideways out of the column and
        dodge a shot it never decided to dodge, which is what made the
        travelling bolt unusable.

    Given: A chicken in the ship's column walking right, so that after the step
        it is one column over, and a dive rate of 1 so every coin also flips.
    When: The ship fires.
    Then: It dies anyway.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 2, 1, MODE_PATROL, 1], [0, 3, 1, MODE_PATROL, 1]]
    )
    flock = env.chickens(env.sample_next_state(state, int(CrazyChickenAction.FIRE)))
    assert flock[0][CHICKEN_ALIVE] == 0.0


def test_a_chicken_shot_this_step_never_gets_to_dive():
    """Shooting is a defence as well as an attack.

    Given: A chicken one row above the ship, in its column, and a dive rate of 1
        so it would certainly reach the ship this step.
    When: The ship fires instead of dodging.
    Then: The chicken is dead and the ship is unharmed.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 1, 1, MODE_PATROL, 1], [0, 3, 1, MODE_PATROL, 1]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert env.chickens(successor)[0][CHICKEN_ALIVE] == 0.0
    assert successor[SHIP_HIT_INDEX] == 0.0


def test_the_gun_never_reaches_row_zero():
    """The shot covers rows 1 upward, which is why a pulled-up chicken exists.

    Purpose: Row 0 is the ship's own row. A chicken there has already ended the
        episode one way or the other, so including it in the gun's reach would
        be dead code that hid the pull-up rule's reason for existing.

    Test type: unit
    """
    env = build_env()
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 0, 1, MODE_PATROL, 1], [0, 3, 1, MODE_PATROL, 0]]
    )
    assert env.shot_target(state) == -1


def test_a_chicken_reaching_the_ship_ends_the_episode():
    """A dive that lands on the ship's cell destroys it.

    Given: A diving chicken on row 1 directly above the ship.
    When: One step is taken with the ship holding position.
    Then: The successor carries the hit flag, is terminal, and the reward is the
        step cost plus the hit penalty.

    Test type: unit
    """
    env = build_env()
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env,
        chickens=[[column, 1, 1, MODE_DIVE, 1], [0, 3, 1, MODE_PATROL, 0]],
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    assert successor[SHIP_HIT_INDEX] == 1.0
    assert env.is_terminal(successor)
    assert env.reward(state, int(CrazyChickenAction.STAY), successor) == pytest.approx(
        -env.step_cost - env.ship_hit_penalty
    )


def test_a_chicken_reaching_row_zero_elsewhere_pulls_up():
    """A dive that misses the ship returns to patrol at the top row.

    Purpose: This rule is an addition to the design proposal, and it is what
        keeps the task both winnable and non-trivial: without it a chicken
        either leaves the grid, clearing the flock for free, or parks on row 0,
        below the rows the gun covers.

    Given: A diving chicken on row 1, two columns away from the ship.
    When: One step is taken.
    Then: It is alive, patrolling, on the top row, in the same column, and the
        ship is unharmed.

    Test type: unit
    """
    env = build_env()
    column = max(env.ship_start_column - 2, 0)
    state = create_crazy_chicken_state(
        env, chickens=[[column, 1, -1, MODE_DIVE, 1], [0, 3, 1, MODE_PATROL, 0]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    chicken = env.chickens(successor)[0]
    assert successor[SHIP_HIT_INDEX] == 0.0
    assert chicken[CHICKEN_ALIVE] == 1.0
    assert chicken[CHICKEN_MODE] == MODE_PATROL
    assert chicken[CHICKEN_ROW] == env.num_rows - 1
    assert chicken[CHICKEN_COLUMN] == column


def test_ship_movement_is_clamped_at_both_walls():
    """The ship cannot walk off the grid.

    Test type: unit
    """
    env = build_env()
    chickens = [[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]]
    at_left = create_crazy_chicken_state(env, chickens=chickens, ship_column=0)
    at_right = create_crazy_chicken_state(env, chickens=chickens, ship_column=4)
    assert env.ship_column(env.sample_next_state(at_left, int(CrazyChickenAction.LEFT))) == 0
    assert env.ship_column(env.sample_next_state(at_right, int(CrazyChickenAction.RIGHT))) == 4


def test_clearing_the_flock_pays_the_bonus_and_ends_the_episode():
    """The last kill is worth the kill reward and the completion bonus together.

    Given: One live chicken left, standing in the ship's column.
    When: The ship fires.
    Then: The flock is empty, the state is terminal, and the reward is the kill
        plus the bonus less the step and shot costs -- which is also the
        declared maximum.

    Test type: unit
    """
    env = build_env()
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 2, 1, MODE_PATROL, 1], [0, 3, 1, MODE_PATROL, 0]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert env.live_chicken_count(successor) == 0
    assert env.is_terminal(successor)
    assert env.reward(state, int(CrazyChickenAction.FIRE), successor) == pytest.approx(
        env.kill_reward + env.clear_reward - env.step_cost - env.shot_cost
    )
    assert env.reward(state, int(CrazyChickenAction.FIRE), successor) == pytest.approx(
        env.reward_range[1]
    )


def test_timeout_is_terminal_at_max_steps():
    """An episode that neither wins nor loses stops at its budget.

    Test type: unit
    """
    env = build_env(max_steps=3)
    state = create_crazy_chicken_state(
        env, chickens=[[1, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]], step=3
    )
    assert env.is_terminal(state)


def test_firing_always_kills_when_it_could_be_overrun_so_the_minimum_is_conservative():
    """A shot and a ship hit can coincide, but never without a kill.

    Purpose: The declared minimum stacks the step cost, the shot cost and the
        ship-hit penalty. That sum is deliberately *not* reachable, and the
        reason is worth pinning because it is not obvious. The gun kills the
        lowest chicken in the ship's column over rows 1 upward, and a chicken
        can only reach the ship by diving down that same column -- so if
        anything could overrun the ship this step, the shot had a target, and
        the kill reward comes back. Firing into a genuinely empty column cannot
        be overrun at all.

        Keeping the wider bound costs nothing and survives a change to the
        gun's reach; deriving a tight one from this argument would not.

    Given: Two chickens stacked on one cell of the ship's column one row up,
        which is the closest the dynamics get to firing while being overrun,
        with a dive rate of 1.
    When: The ship fires.
    Then: One dies, the ship is destroyed, and the reward is the declared
        minimum plus exactly the kill reward -- still inside the range.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 1, 1, MODE_PATROL, 1], [column, 1, -1, MODE_PATROL, 1]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert env.live_chicken_count(successor) == 1
    assert successor[SHIP_HIT_INDEX] == 1.0
    worst_while_firing = env.reward(state, int(CrazyChickenAction.FIRE), successor)
    assert worst_while_firing == pytest.approx(env.reward_range[0] + env.kill_reward)
    assert worst_while_firing >= env.reward_range[0]


def test_the_worst_reachable_step_is_being_overrun_without_firing():
    """The lowest reward a run can actually score is the hit alone.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 1, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    overrun = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    assert overrun[SHIP_HIT_INDEX] == 1.0
    worst = env.reward(state, int(CrazyChickenAction.STAY), overrun)
    assert worst == pytest.approx(-env.step_cost - env.ship_hit_penalty)
    assert worst == pytest.approx(env.reward_range[0] + env.shot_cost)
    assert worst >= env.reward_range[0]


def test_a_shot_into_its_own_column_protects_the_ship_that_step():
    """The ordinary case: firing clears the column the flock would come down.

    Purpose: The counterpart to the test above. The declared minimum is
        reachable, but only through the stacked case -- in every ordinary
        position a shot up the ship's column removes exactly the chicken that
        was about to arrive. Recording both halves means a change to the gun's
        reach fails here rather than silently widening what the ship risks.

    Test type: unit
    """
    env = build_env(dive_probability=1.0)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env, chickens=[[column, 1, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    shot = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert shot[SHIP_HIT_INDEX] == 0.0
    assert env.chickens(shot)[0][CHICKEN_ALIVE] == 0.0


def test_reward_without_a_successor_scores_the_kill_it_can_already_see():
    """Everything but the ship hit is exact without a successor.

    Purpose: A hitscan shot resolves before anything random happens, so the
        kill and the completion bonus are functions of ``(state, action)``
        alone. A planner comparing actions at a belief node therefore sees the
        real value of a shot that connects, instead of only its cost -- which
        under the travelling bolt was all the fallback could offer.

    Test type: unit
    """
    env = build_env()
    column = env.ship_start_column
    hitting = create_crazy_chicken_state(
        env, chickens=[[column, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    assert env.reward(hitting, int(CrazyChickenAction.STAY)) == pytest.approx(-env.step_cost)
    assert env.reward(hitting, int(CrazyChickenAction.FIRE)) == pytest.approx(
        -env.step_cost - env.shot_cost + env.kill_reward
    )

    missing = create_crazy_chicken_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]], ship_column=2
    )
    assert env.reward(missing, int(CrazyChickenAction.FIRE)) == pytest.approx(
        -env.step_cost - env.shot_cost
    )

    last_one = create_crazy_chicken_state(
        env, chickens=[[column, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 0]]
    )
    assert env.reward(last_one, int(CrazyChickenAction.FIRE)) == pytest.approx(
        -env.step_cost - env.shot_cost + env.kill_reward + env.clear_reward
    )


def test_transition_log_probability_agrees_with_the_dive_coins_it_implies():
    """A successor scores the probability of the dives that produced it.

    Given: A world with two live patrolling chickens and a dive rate of 0.25.
    When: The successor in which exactly one of them dives is scored.
    Then: Its log-probability is ``log(0.25) + log(0.75)``, and a successor the
        dynamics cannot produce scores the impossible floor.

    Test type: unit
    """
    env = build_env(dive_probability=0.25)
    state = create_crazy_chicken_state(
        env, chickens=[[1, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    # pylint: disable-next=protected-access
    successor = env._step_once(
        state, int(CrazyChickenAction.STAY), switches=np.array([True, False])
    )
    score = env.transition_log_probability(state, int(CrazyChickenAction.STAY), [successor])
    assert score[0] == pytest.approx(np.log(0.25) + np.log(0.75))

    impossible = np.array(successor, copy=True)
    impossible[0] += 5.0
    assert (
        env.transition_log_probability(state, int(CrazyChickenAction.STAY), [impossible])[0] < -1e17
    )


def test_chickens_never_start_two_to_a_cell():
    """The flock prior places chickens on distinct cells.

    Purpose: Two chickens on one start cell would waste a slot, since a hitscan
        shot only ever removes the lowest chicken in its column.

    Test type: unit
    """
    env = CrazyChickenPOMDP(num_columns=4, num_rows=3, num_chickens=6, discount_factor=0.95)
    np.random.seed(5)
    for state in env.initial_state_dist().sample(50):
        flock = env.chickens(state)
        cells = {(float(c), float(r)) for c, r in flock[:, [CHICKEN_COLUMN, CHICKEN_ROW]]}
        assert len(cells) == env.num_chickens


def test_a_flock_larger_than_the_grid_is_rejected_at_construction():
    """More chickens than start cells fails loudly rather than looping forever.

    Test type: unit
    """
    with pytest.raises(ValueError, match="do not fit"):
        CrazyChickenPOMDP(num_columns=3, num_rows=2, num_chickens=4, discount_factor=0.95)


@pytest.mark.parametrize("dive_probability", [0.0, 0.15, 1.0])
@pytest.mark.parametrize("action", list(CrazyChickenAction))
def test_transition_log_probability_sums_to_one_over_reachable_successors(action, dive_probability):
    """Every action's transition is a distribution, at every dive rate.

    Purpose: This is the check that would have caught two separate scoring
        bugs, and neither showed up in a per-rule test. The coin eligibility
        was read off the *pre-shot* flock while the sampler applies coins to
        the post-shot flock, so a ``FIRE`` that killed a patroller counted the
        victim's discarded coin as "held" and summed to ``1 - dive_probability``
        instead of 1 -- and at ``dive_probability = 1`` it sent every candidate
        to the impossible floor. Separately, a chicken that dived off the bottom
        and pulled up was reset to ``MODE_PATROL``, so its coin could not be
        read back and a successor the sampler really produces scored the floor.

        Only ``FIRE`` exposes the first and only a pull-up exposes the second,
        which is why this sweeps every action rather than the one
        ``get_actions()[0]`` the shared conformance probe happens to pick.

    Given: A world where the ship's column holds a patroller the shot can kill
        and another chicken sits on row 1 off to the side, so both a kill and a
        pull-up are reachable in one step.
    When: Successors are sampled repeatedly, deduplicated, and scored.
    Then: Their probabilities sum to one, and none of them is impossible.

    Test type: integration
    """
    env = build_env(num_columns=4, num_rows=4, num_chickens=3, dive_probability=dive_probability)
    column = env.ship_start_column
    state = create_crazy_chicken_state(
        env,
        chickens=[
            [column, 2, 1, MODE_PATROL, 1],
            [0, 1, 1, MODE_PATROL, 1],
            [3, 3, -1, MODE_PATROL, 1],
        ],
    )

    np.random.seed(11)
    seen = {}
    for _ in range(3000):
        successor = env.sample_next_state(state, int(action))
        seen[successor.tobytes()] = successor

    successors = list(seen.values())
    scores = env.transition_log_probability(state, int(action), successors)
    assert np.all(scores > -1e17), (
        f"{action.name} at dive_probability={dive_probability}: "
        f"{int(np.count_nonzero(scores <= -1e17))} of {len(successors)} successors the "
        f"sampler really produced score as impossible"
    )
    assert float(np.exp(scores).sum()) == pytest.approx(1.0, abs=1e-9), (
        f"{action.name} at dive_probability={dive_probability}: transition sums to "
        f"{float(np.exp(scores).sum())}, not 1"
    )


def test_a_pulled_up_chickens_coin_is_read_back_from_its_row_jump():
    """The successor of a dive that pulled up is scoreable.

    Purpose: ``_resolve_arrivals`` resets a pulled-up chicken to ``MODE_PATROL``,
        so its dive coin cannot be read off the final mode. Reading only the
        mode rebuilt it as a sideways patrol step, the rebuild did not match,
        and a successor the sampler really produces scored the impossible floor.
        This is the exact repro, with the coin forced.

    Test type: unit
    """
    env = build_env(num_columns=4, num_rows=4, num_chickens=1, dive_probability=1.0)
    state = create_crazy_chicken_state(env, chickens=[[0, 1, 1, MODE_PATROL, 1]], ship_column=2)
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    chicken = env.chickens(successor)[0]
    # It dived off the bottom away from the ship and came back at the top.
    assert successor[SHIP_HIT_INDEX] == 0.0
    assert (chicken[CHICKEN_ROW], chicken[CHICKEN_COLUMN]) == (env.num_rows - 1, 0)
    assert chicken[CHICKEN_MODE] == MODE_PATROL

    score = env.transition_log_probability(state, int(CrazyChickenAction.STAY), [successor])
    assert score[0] == pytest.approx(0.0), "a certain pull-up must score probability 1"


def test_a_shot_victims_discarded_coin_is_not_scored():
    """Killing a patroller does not leave a phantom coin in the transition.

    Purpose: The sampler draws a coin per slot but applies it only where the
        chicken is still alive *after* the shot. Counting the victim's coin as
        held is what made ``FIRE`` sum to ``1 - dive_probability``.

    Given: The only live chicken standing in the ship's column, so the shot
        certainly kills it and no coin survives.
    When: The one reachable successor is scored.
    Then: It has probability 1, for any dive rate.

    Test type: unit
    """
    for dive_probability in (0.0, 0.15, 1.0):
        env = build_env(num_chickens=1, dive_probability=dive_probability)
        column = env.ship_start_column
        state = create_crazy_chicken_state(env, chickens=[[column, 2, 1, MODE_PATROL, 1]])
        successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
        assert env.live_chicken_count(successor) == 0
        score = env.transition_log_probability(state, int(CrazyChickenAction.FIRE), [successor])
        assert score[0] == pytest.approx(0.0), (
            f"at dive_probability={dive_probability} the shot victim's discarded coin "
            f"was scored: {score[0]}"
        )


def test_the_declared_maximum_covers_a_configuration_where_firing_never_pays():
    """The reward range holds for every configuration the constructor admits.

    Purpose: The maximum is derived from the best *firing* step, but the
        constructor only requires the five amounts to be non-negative. With
        ``shot_cost`` above ``kill_reward + clear_reward`` the best step is a
        plain ``STAY``, which scores ``-step_cost`` -- above a maximum computed
        from the firing branch alone. DESPOT consumes ``reward_range`` as a hard
        branch-and-bound bound, so a range the environment can escape voids its
        guarantee somewhere far from here.

    Test type: unit
    """
    env = build_env(kill_reward=1.0, clear_reward=1.0, shot_cost=100.0, step_cost=0.1)
    state = create_crazy_chicken_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    staying = env.reward(state, int(CrazyChickenAction.STAY), successor)
    assert staying == pytest.approx(-env.step_cost)
    assert (
        staying <= env.reward_range[1]
    ), f"a plain STAY scores {staying}, above the declared maximum {env.reward_range[1]}"


def test_the_flock_prior_rejects_a_malformed_direction_or_mode():
    """A candidate with a nonsense categorical field gets no prior mass.

    Test type: unit
    """
    env = build_env(num_chickens=1)
    prior = env.initial_state_dist()
    good = create_crazy_chicken_state(env, chickens=[[1, 2, 1, MODE_PATROL, 1]])
    assert float(prior.probability([good])[0]) > 0.0

    for field, value in ((CHICKEN_DIRECTION, 0.0), (CHICKEN_MODE, 7.0)):
        bad = np.array(good, copy=True)
        env.chickens(bad)[0][field] = value
        assert float(prior.probability([bad])[0]) == 0.0
