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
    NO_PROJECTILE,
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


def test_fire_launches_a_projectile_just_above_the_ship():
    """A shot appears on row 1 of the ship's column.

    Given: An empty sky and no cooldown.
    When: The ship fires.
    Then: Its column holds a projectile at row 1 and every other column is empty.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]]
    )
    sky = env.projectiles(env.sample_next_state(state, int(CrazyChickenAction.FIRE)))
    assert sky[env.ship_start_column] == 1.0
    assert np.count_nonzero(sky != NO_PROJECTILE) == 1


def test_fire_during_cooldown_behaves_exactly_like_stay():
    """A blocked ``FIRE`` costs nothing and launches nothing.

    Purpose: The shot cost is charged on ``fires``, not on the action label. If
        a blocked fire were still charged, a planner would learn to avoid the
        action rather than to time it.

    Given: A ship with one step of cooldown left.
    When: It fires.
    Then: No projectile appears, and the reward equals the reward for staying.

    Test type: unit
    """
    env = build_env(fire_cooldown=2)
    state = create_crazy_chicken_state(
        env,
        chickens=[[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]],
        cooldown=1,
    )
    assert not env.fires(state, int(CrazyChickenAction.FIRE))
    successor = env.sample_next_state(state, int(CrazyChickenAction.FIRE))
    assert np.all(env.projectiles(successor) == NO_PROJECTILE)
    assert env.reward(state, int(CrazyChickenAction.FIRE), successor) == pytest.approx(
        env.reward(state, int(CrazyChickenAction.STAY), successor)
    )


def test_a_column_already_holding_a_projectile_blocks_a_new_one():
    """One projectile per column, which is what makes the ship's position matter.

    Given: A ship whose own column already holds a projectile.
    When: It fires.
    Then: Nothing new is launched, and the existing shot simply rises.

    Test type: unit
    """
    env = build_env()
    column = 2
    projectiles = [NO_PROJECTILE] * 5
    projectiles[column] = 1.0
    state = create_crazy_chicken_state(
        env,
        chickens=[[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]],
        ship_column=column,
        projectiles=projectiles,
    )
    assert not env.fires(state, int(CrazyChickenAction.FIRE))
    sky = env.projectiles(env.sample_next_state(state, int(CrazyChickenAction.FIRE)))
    assert sky[column] == 2.0


def test_a_projectile_kills_the_lowest_chicken_it_reaches_not_the_first_slot():
    """A shot is stopped by the first chicken in its path, by row, not by slot.

    Purpose: Two chickens can share a column inside the one row a projectile
        crosses. Resolving by slot index there would make the outcome depend on
        the order the flock happens to be stored in, which is not a fact about
        the world and would make the transition disagree with itself after any
        reordering.

    Given: A projectile about to rise into row 2, with a chicken already in
        slot 0 at row 2 and another in slot 1 diving from row 3 to row 2.
    When: One step is taken.
    Then: Exactly one of them dies -- the one that is lower after the step --
        and the shot is spent.

    Test type: unit
    """
    env = build_env(num_rows=5, num_chickens=2)
    projectiles = [NO_PROJECTILE] * 5
    projectiles[2] = 1.0
    state = create_crazy_chicken_state(
        env,
        chickens=[[2, 3, 1, MODE_DIVE, 1], [2, 2, 1, MODE_DIVE, 1]],
        projectiles=projectiles,
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    flock = env.chickens(successor)
    assert env.live_chicken_count(successor) == 1
    # Slot 1 ends the step on row 1, below slot 0's row 2, so it is the one the
    # rising shot meets first.
    assert flock[1][CHICKEN_ALIVE] == 0.0
    assert flock[0][CHICKEN_ALIVE] == 1.0
    assert env.projectiles(successor)[2] == NO_PROJECTILE


def test_a_projectile_kills_the_chicken_it_reaches_and_dies_with_it():
    """A hit removes both the chicken and the shot.

    Given: A projectile on row 1 and a patrolling chicken on row 2 of the same
        column that the shot rises into.
    When: One step is taken.
    Then: The chicken is dead and the column is empty.

    Test type: unit
    """
    env = build_env()
    projectiles = [NO_PROJECTILE] * 5
    projectiles[3] = 1.0
    # Walking left from column 4 puts the chicken in column 3 as the shot
    # arrives on row 2, which is the co-location case.
    state = create_crazy_chicken_state(
        env,
        chickens=[[4, 2, -1, MODE_PATROL, 1], [0, 3, 1, MODE_PATROL, 0]],
        projectiles=projectiles,
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    assert env.chickens(successor)[0][CHICKEN_ALIVE] == 0.0
    assert env.projectiles(successor)[3] == NO_PROJECTILE


def test_a_diving_chicken_cannot_pass_through_a_rising_projectile():
    """A shot and a dive that swap rows still collide.

    Purpose: A naive same-cell test misses exactly this case, and it is the one
        a player would notice first: the chicken dives through the bullet.

    Given: A projectile on row 1 and a diving chicken on row 2 of that column.
    When: One step is taken, so the shot rises to 2 and the chicken drops to 1.
    Then: The chicken is dead.

    Test type: unit
    """
    env = build_env()
    projectiles = [NO_PROJECTILE] * 5
    projectiles[2] = 1.0
    state = create_crazy_chicken_state(
        env,
        chickens=[[2, 2, 1, MODE_DIVE, 1], [0, 3, 1, MODE_PATROL, 0]],
        projectiles=projectiles,
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    assert env.chickens(successor)[0][CHICKEN_ALIVE] == 0.0


def test_a_projectile_leaves_the_grid_above_the_top_row():
    """A shot that reaches the top is gone, not parked there.

    Given: A projectile on the top row.
    When: One step is taken.
    Then: Its column is empty again.

    Test type: unit
    """
    env = build_env()
    projectiles = [NO_PROJECTILE] * 5
    projectiles[1] = float(env.num_rows - 1)
    state = create_crazy_chicken_state(
        env,
        chickens=[[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]],
        projectiles=projectiles,
    )
    sky = env.projectiles(env.sample_next_state(state, int(CrazyChickenAction.STAY)))
    assert sky[1] == NO_PROJECTILE


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
        either leaves the grid, clearing the flock for free, or parks on a row
        no projectile can reach.

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

    Given: One live chicken about to be hit by a rising projectile.
    When: The step is taken.
    Then: The flock is empty, the state is terminal, and the reward is the kill
        plus the bonus less the step cost.

    Test type: unit
    """
    env = build_env()
    projectiles = [NO_PROJECTILE] * 5
    projectiles[1] = 1.0
    state = create_crazy_chicken_state(
        env,
        chickens=[[1, 2, 1, MODE_DIVE, 1], [0, 3, 1, MODE_PATROL, 0]],
        projectiles=projectiles,
    )
    successor = env.sample_next_state(state, int(CrazyChickenAction.STAY))
    assert env.live_chicken_count(successor) == 0
    assert env.is_terminal(successor)
    assert env.reward(state, int(CrazyChickenAction.STAY), successor) == pytest.approx(
        env.kill_reward + env.clear_reward - env.step_cost
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


def test_the_declared_maximum_reward_is_attained_by_the_step_it_enumerates():
    """The best step really is worth the declared maximum, to the cent.

    Purpose: A reward range that is merely *wide enough* drifts silently when
        the reward changes. Pinning the upper end to the specific step the
        constructor enumerates makes a change to the kill or clear term fail
        here rather than in a CVaR estimator downstream.

    Given: A world whose two chickens sit on separate columns, each under a
        projectile about to reach it.
    When: The ship holds position, both chickens die and the flock is cleared.
    Then: The reward equals the declared maximum.

    Test type: unit
    """
    env = build_env()
    projectiles = [NO_PROJECTILE] * 5
    projectiles[0] = 1.0
    projectiles[1] = 1.0
    before = create_crazy_chicken_state(
        env,
        chickens=[[0, 2, 1, MODE_DIVE, 1], [1, 2, 1, MODE_DIVE, 1]],
        projectiles=projectiles,
    )
    after = env.sample_next_state(before, int(CrazyChickenAction.STAY))
    assert env.live_chicken_count(after) == 0
    assert env.reward(before, int(CrazyChickenAction.STAY), after) == pytest.approx(
        env.reward_range[1]
    )


def test_firing_protects_the_ship_column_so_the_minimum_is_conservative():
    """The worst reward is a ship hit alone, one shot cost above the declared floor.

    Purpose: The declared minimum stacks the shot cost onto the ship hit even
        though the two cannot in fact coincide -- a shot rising out of the
        ship's column meets the chicken diving into it. That is a recorded
        decision, not an oversight: tightening the bound would make it depend on
        how a projectile and a dive resolve, and a later change there would put
        a real reward outside the declared range with nothing to catch it. This
        test records both halves, so if the kill rule ever changes and the two
        *do* coincide, the first assertion fails and the decision comes back
        into view.

    Given: A diving chicken one row above the ship.
    When: The ship fires.
    Then: The chicken is shot down rather than reaching the ship, and the worst
        reward actually reachable -- the same dive with the ship holding
        position -- sits exactly one shot cost above the declared minimum.

    Test type: unit
    """
    env = build_env()
    column = env.ship_start_column
    before = create_crazy_chicken_state(
        env, chickens=[[column, 1, 1, MODE_DIVE, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    shot = env.sample_next_state(before, int(CrazyChickenAction.FIRE))
    assert shot[SHIP_HIT_INDEX] == 0.0
    assert env.chickens(shot)[0][CHICKEN_ALIVE] == 0.0

    overrun = env.sample_next_state(before, int(CrazyChickenAction.STAY))
    assert overrun[SHIP_HIT_INDEX] == 1.0
    worst = env.reward(before, int(CrazyChickenAction.STAY), overrun)
    assert worst == pytest.approx(env.reward_range[0] + env.shot_cost)
    assert worst >= env.reward_range[0]


def test_reward_without_a_successor_charges_only_what_is_already_decided():
    """The no-successor fallback is the step cost plus a shot that really fired.

    Purpose: The fallback is the number a belief-space planner compares actions
        with. Making it an expectation over kills would be a different
        objective; making it zero would hide the cost of firing.

    Test type: unit
    """
    env = build_env()
    state = create_crazy_chicken_state(
        env, chickens=[[1, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    assert env.reward(state, int(CrazyChickenAction.STAY)) == pytest.approx(-env.step_cost)
    assert env.reward(state, int(CrazyChickenAction.FIRE)) == pytest.approx(
        -env.step_cost - env.shot_cost
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

    Purpose: Two chickens on one cell would let a single projectile score two
        kills, which is the one way a step could exceed the declared reward
        maximum.

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
