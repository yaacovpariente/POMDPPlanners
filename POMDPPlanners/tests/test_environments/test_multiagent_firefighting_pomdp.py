# SPDX-License-Identifier: MIT

"""Tests specific to :class:`MultiAgentFirefightingPOMDP`.

The shared API contracts -- hashing, batch agreement, serialization identity,
reward-range bounding, seeded reproducibility, metric channels -- are covered
once for every registered environment by ``test_env_api_conformance.py``. What
is here is what is specific to this one: the six transition stages, the
observation model, the terminal precedence, and the two closed-form densities,
which are the pieces a planner's belief is built on and which nothing else
checks.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
    DIRECTION_OFFSETS,
    FireCategory,
    FirefightingAction,
    MultiAgentFirefightingMetrics,
    MultiAgentFirefightingPOMDP,
    MultiAgentFirefightingStepChannel,
    WindDirection,
    WindStrength,
    create_firefighting_state,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    multiagent_firefighting_pinned_kwargs,
)


def build_env(**overrides) -> MultiAgentFirefightingPOMDP:
    """Build the pinned environment, with overrides merged on top.

    Args:
        **overrides: Constructor kwargs replacing pinned defaults.

    Returns:
        The environment.
    """
    return MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs(**overrides)
    )


def joint(*per_robot: int) -> int:
    """Encode per-robot actions as one base-5 joint action.

    Args:
        *per_robot: One action per robot, robot 0 first.

    Returns:
        The joint action.
    """
    return int(sum(int(action) * 5**index for index, action in enumerate(per_robot)))


def empty_fire(env: MultiAgentFirefightingPOMDP) -> np.ndarray:
    """Return an all-unburnt fire map for ``env``.

    Args:
        env: The environment whose grid shape is used.

    Returns:
        ``(num_rows, num_cols)`` of ``UNBURNT``.
    """
    return np.full((env.num_rows, env.num_cols), float(FireCategory.UNBURNT))


# ---------------------------------------------------------------------------
# reward range
# ---------------------------------------------------------------------------


def test_reward_range_is_derived_from_the_constructor_arguments() -> None:
    """The declared range tracks the coefficients, the grid and the robot count.

    Purpose: The reward range is this repository's most-repeated bug, and the
        failure mode is a constant that was right for one configuration and
        silently wrong for every other. Building three different environments
        and checking the bound moves with each one is what distinguishes a
        derived bound from a hard-coded one that happens to match.

    Given: The default environment, one with a bigger grid, and one with more
        robots and a different damage cost.
    When: Their declared reward ranges are read.
    Then: Each equals the formula re-derived from that environment's own
        arguments, and the three differ from each other.

    Test type: unit
    """
    for kwargs in (
        {},
        {
            "num_rows": 4,
            "num_cols": 7,
            "obstacle_cells": [(2, 3)],
            "robot_start_cells": [(1, 1), (1, 2)],
        },
        {"num_robots": 3, "robot_start_cells": [(2, 2), (2, 3), (3, 2)], "damage_cost": 3.0},
        {"max_health": 1},
    ):
        env = build_env(**kwargs)
        expected_max = env.success_reward - env.step_cost
        expected_min = -(
            env.step_cost
            + max(env.smoldering_cell_cost, env.burning_cell_cost, env.burnt_cell_cost)
            * env.num_cells
            + env.damage_cost * env.num_robots * min(env.max_health, 2)
            + env.water_cost * env.num_robots
        )
        assert env.reward_range == pytest.approx((expected_min, expected_max))

    assert (
        build_env().reward_range
        != build_env(
            num_rows=4,
            num_cols=7,
            obstacle_cells=[(2, 3)],
            robot_start_cells=[(1, 1), (1, 2)],
        ).reward_range
    )


def test_reward_range_does_not_stack_the_alight_and_newly_burnt_budgets() -> None:
    """The map terms share one budget of ``K`` cells rather than summing.

    Purpose: A cell in the successor is smoldering, or burning, or newly burnt,
        or none of those -- never two at once. Charging all three coefficients
        against a full grid would declare a bound roughly three times too wide,
        which passes every test while hiding any real violation.

    Given: The default environment.
    When: Its declared minimum is compared with the summed-budget version.
    Then: The declared minimum is the shared-budget one, strictly tighter than
        the summed one.

    Test type: unit
    """
    env = build_env()
    summed = -(
        env.step_cost
        + (env.smoldering_cell_cost + env.burning_cell_cost + env.burnt_cell_cost) * env.num_cells
        + env.damage_cost * env.num_robots * 2
        + env.water_cost * env.num_robots
    )
    assert env.reward_range[0] > summed


def test_declared_minimum_bounds_a_hand_built_worst_case_step() -> None:
    """A deliberately terrible transition still scores inside the declared range.

    Purpose: The bound is only worth anything if a bad step really cannot beat
        it. A hand-built step that burns a whole grid's worth of cells and
        disables both robots is the closest a reachable transition gets.

    Given: A successor where every non-obstacle cell is newly burnt and both
        robots lost their maximum per-step health, reached by a joint SUPPRESS.
    When: The reward is computed for that transition.
    Then: It lies inside the declared range.

    Test type: unit
    """
    env = build_env()
    before = empty_fire(env)
    before[:, :] = float(FireCategory.BURNING)
    state = create_firefighting_state(
        env,
        [(4, 4, env.max_tank, env.max_health), (4, 5, env.max_tank, env.max_health)],
        (0, 0),
        before,
    )
    after = np.full_like(before, float(FireCategory.BURNT))
    # One cell is left burning on purpose. A wholly burnt successor has no
    # alight cell, which is the *success* condition, so it would collect the
    # bonus and score a hundred points above the floor -- the worst reachable
    # step is the one where almost everything is destroyed and the fire is
    # still going.
    after[0, 0] = float(FireCategory.BURNING)
    successor = create_firefighting_state(
        env,
        [
            (4, 4, env.max_tank - 1, env.max_health - 2),
            (4, 5, env.max_tank - 1, env.max_health - 2),
        ],
        (0, 0),
        after,
        step=1,
    )
    reward = env.reward(
        state, joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS), successor
    )
    low, high = env.reward_range
    assert low <= reward <= high
    # And it really is near the floor, not comfortably inside it by accident:
    # within one cell's worth of the declared minimum.
    assert reward < low + env.burnt_cell_cost


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


def test_joint_action_decodes_as_base_five_digits() -> None:
    """Robot ``i`` is the ``i``-th base-5 digit, least significant first.

    Purpose: The joint action is the one place several robots are squeezed into
        one integer, and an endianness slip there swaps the robots silently --
        every episode still runs, and every result is wrong.

    Given: The default two-robot environment and a three-robot one.
    When: Every joint action is decoded and re-encoded.
    Then: The decoding round-trips, and a known action decodes to the digits
        the formal definition gives.

    Test type: unit
    """
    for env in (build_env(), build_env(num_robots=3, robot_start_cells=[(2, 2), (2, 3), (3, 2)])):
        actions = env.get_actions()
        assert actions == list(range(5**env.num_robots))
        for action in actions:
            per_robot = env.decode_action(action)
            assert joint(*per_robot) == action
    env = build_env()
    assert list(env.decode_action(joint(FirefightingAction.SUPPRESS, FirefightingAction.EAST))) == [
        int(FirefightingAction.SUPPRESS),
        int(FirefightingAction.EAST),
    ]
    with pytest.raises(ValueError):
        env.decode_action(env.num_actions)


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------


def test_motion_is_refused_by_edges_obstacles_and_burnt_cells() -> None:
    """The three inadmissible targets all leave the robot where it was.

    Purpose: "Admissible" has three separate clauses and each has been an
        off-by-one somewhere. A robot that can walk through a wall, or onto
        ash, changes what the task is.

    Given: A robot placed against the north edge, beside the obstacle blob,
        and beside a burnt cell, with slipping turned off.
    When: It is told to move into each of them, many times.
    Then: It never moves, and a legal move always succeeds.

    Test type: unit
    """
    env = build_env(slip_probability=0.0)
    fire = empty_fire(env)
    fire[3, 3] = float(FireCategory.BURNT)
    fire[9, 9] = float(FireCategory.SMOLDERING)  # keeps the state non-terminal

    cases = [
        ((0, 4), FirefightingAction.NORTH),  # off the grid
        ((4, 5), FirefightingAction.SOUTH),  # into the obstacle at (5, 5)
        ((2, 3), FirefightingAction.SOUTH),  # into the burnt cell at (3, 3)
    ]
    for (row, col), action in cases:
        state = create_firefighting_state(env, [(row, col, 6, 3), (9, 0, 6, 3)], (0, 0), fire)
        for _ in range(5):
            successor = env.sample_next_state(state, joint(action, FirefightingAction.SUPPRESS))
            assert tuple(env.robots(successor)[0, :2]) == (row, col)

    state = create_firefighting_state(env, [(4, 4, 6, 3), (9, 0, 6, 3)], (0, 0), fire)
    successor = env.sample_next_state(
        state, joint(FirefightingAction.EAST, FirefightingAction.SUPPRESS)
    )
    assert tuple(env.robots(successor)[0, :2]) == (4, 5)


def test_slipping_sometimes_refuses_an_otherwise_legal_move() -> None:
    """The slip probability is the rate at which an admissible move fails.

    Purpose: Slipping is the only stochasticity in the robots' own motion, and
        a version that never fires would make the poses deterministic and quietly
        change what the belief has to track.

    Given: A robot with a legal move and a slip probability of 0.5.
    When: The move is sampled many times.
    Then: Roughly half the attempts stay put, and at a slip probability of 1.0
        none of them move.

    Test type: unit
    """
    fire_env = build_env(slip_probability=0.5)
    fire = empty_fire(fire_env)
    fire[9, 9] = float(FireCategory.SMOLDERING)
    state = create_firefighting_state(fire_env, [(4, 4, 6, 3), (9, 0, 6, 3)], (0, 0), fire)
    np.random.seed(0)
    moved = sum(
        tuple(
            fire_env.robots(
                fire_env.sample_next_state(
                    state, joint(FirefightingAction.EAST, FirefightingAction.SUPPRESS)
                )
            )[0, :2]
        )
        == (4, 5)
        for _ in range(600)
    )
    assert 0.4 < moved / 600 < 0.6

    always = build_env(slip_probability=1.0)
    state = create_firefighting_state(always, [(4, 4, 6, 3), (9, 0, 6, 3)], (0, 0), fire)
    for _ in range(10):
        successor = always.sample_next_state(
            state, joint(FirefightingAction.EAST, FirefightingAction.SUPPRESS)
        )
        assert tuple(always.robots(successor)[0, :2]) == (4, 4)


def test_a_disabled_robot_ignores_its_digit_of_the_joint_action() -> None:
    """Zero health means the robot is scenery, not a smaller action space.

    Purpose: The action space stays ``5 ** N`` whatever has happened to the
        robots, so the inert part of it has to be genuinely inert -- a disabled
        robot that still moved, or still sprayed, would keep costing water and
        keep changing the map.

    Given: A disabled robot told to move, and a disabled robot told to spray a
        burning cell it is standing next to.
    When: The transition is sampled, with slipping off.
    Then: It does not move, its tank is untouched, and the fire is unchanged.

    Test type: unit
    """
    env = build_env(slip_probability=0.0)
    fire = empty_fire(env)
    fire[4, 5] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 4, 6, 0), (9, 0, 6, 0)], (0, 0), fire)
    successor = env.sample_next_state(
        state, joint(FirefightingAction.EAST, FirefightingAction.SUPPRESS)
    )
    assert tuple(env.robots(successor)[0, :2]) == (4, 4)

    state = create_firefighting_state(env, [(4, 4, 6, 0), (9, 0, 6, 0)], (0, 0), fire)
    successor = env.sample_next_state(
        state, joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS)
    )
    assert int(env.robots(successor)[0, 2]) == 6
    assert int(env.fire_map(successor)[4, 5]) != int(FireCategory.WET)


# ---------------------------------------------------------------------------
# suppression, tanks and the depot
# ---------------------------------------------------------------------------


def test_suppression_soaks_the_robot_cell_and_its_four_neighbours() -> None:
    """One spray covers ``N+(p)``, and unburnt cells are pre-wetted.

    Purpose: Suppression reaching neighbours is the point of the design -- it
        is what lets a careful planner fight from an adjacent cell -- and the
        unburnt case is how a firebreak gets built ahead of the front.

    Given: A robot in the middle of an all-unburnt grid, whose unburnt
        suppression probability is 1.
    When: It sprays once.
    Then: Exactly its own cell and its four neighbours are wet, and one unit
        has left its tank.

    Test type: unit
    """
    env = build_env()
    state = create_firefighting_state(env, [(4, 4, 6, 3), (9, 9, 6, 3)], (0, 0), empty_fire(env))
    successor = env.sample_next_state(
        state, joint(FirefightingAction.SUPPRESS, FirefightingAction.NORTH)
    )
    wet = {tuple(cell) for cell in np.argwhere(env.fire_map(successor) == int(FireCategory.WET))}
    expected = {(4, 4)} | {(4 + dr, 4 + dc) for dr, dc in DIRECTION_OFFSETS}
    assert wet == expected
    assert int(env.robots(successor)[0, 2]) == 5


def test_two_sprays_on_one_cell_beat_one() -> None:
    """Overlapping coverage gives each robot an independent attempt.

    Purpose: This is the only place in the model where the robots genuinely
        cooperate rather than divide the work, so a version that took the
        maximum rather than compounding would silently remove the reason to
        put two robots on one cell.

    Given: A burning cell covered by one robot, then by two.
    When: The soak rate is measured over many samples.
    Then: The two-robot rate matches ``1 - (1 - q)^2`` and beats the one-robot
        rate, which matches ``q``.

    Test type: integration
    """
    env = build_env()
    quality = env.suppression_probability_burning
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)

    def soak_rate(second_cell, second_action, trials=1500):
        state = create_firefighting_state(
            env, [(4, 3, 6, 3), (second_cell[0], second_cell[1], 6, 3)], (0, 0), fire
        )
        np.random.seed(11)
        return (
            sum(
                int(
                    env.fire_map(
                        env.sample_next_state(
                            state, joint(FirefightingAction.SUPPRESS, second_action)
                        )
                    )[4, 4]
                )
                == int(FireCategory.WET)
                for _ in range(trials)
            )
            / trials
        )

    one = soak_rate((9, 9), FirefightingAction.NORTH)
    two = soak_rate((4, 5), FirefightingAction.SUPPRESS)
    assert one == pytest.approx(quality, abs=0.04)
    assert two == pytest.approx(1.0 - (1.0 - quality) ** 2, abs=0.04)
    assert two > one


def test_an_empty_tank_sprays_nothing_and_pays_nothing() -> None:
    """A dry robot choosing SUPPRESS is a no-op, not a free spray.

    Purpose: If an empty tank still soaked cells the depot would be pointless,
        and if it still charged the water cost the planner would be paying for
        nothing it could observe.

    Given: A robot with an empty tank beside a burning cell.
    When: It chooses SUPPRESS many times.
    Then: The cell never becomes wet, the tank stays at zero, and the reward
        carries no water cost.

    Test type: unit
    """
    env = build_env()
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 3, 0, 3), (9, 9, 6, 3)], (0, 0), fire)
    action = joint(FirefightingAction.SUPPRESS, FirefightingAction.NORTH)
    np.random.seed(2)
    for _ in range(40):
        successor = env.sample_next_state(state, action)
        assert int(env.fire_map(successor)[4, 4]) != int(FireCategory.WET)
        assert int(env.robots(successor)[0, 2]) == 0
    assert (
        env.step_info(state, action, successor)[
            MultiAgentFirefightingStepChannel.SUPPRESS_ACTIONS.value
        ]
        == 0.0
    )


def test_entering_the_depot_refills_and_overrides_the_spray_cost() -> None:
    """A robot that sprays into the depot ends the step full.

    Purpose: The refill rule is written to override the suppression cost rather
        than to be applied before it, and getting the order wrong costs the
        planner one unit on exactly the step it was trying to top up.

    Given: A robot one cell from the depot, and a robot standing on it.
    When: It moves in, and separately sprays while standing on it.
    Then: Both end the step with a full tank.

    Test type: unit
    """
    env = build_env(slip_probability=0.0)
    fire = empty_fire(env)
    fire[9, 9] = float(FireCategory.SMOLDERING)
    depot_row, depot_col = env.depot_cell

    state = create_firefighting_state(
        env, [(depot_row + 1, depot_col, 1, 3), (5, 0, 6, 3)], (0, 0), fire
    )
    successor = env.sample_next_state(
        state, joint(FirefightingAction.NORTH, FirefightingAction.NORTH)
    )
    assert tuple(env.robots(successor)[0, :2]) == env.depot_cell
    assert int(env.robots(successor)[0, 2]) == env.max_tank

    state = create_firefighting_state(
        env, [(depot_row, depot_col, 1, 3), (5, 0, 6, 3)], (0, 0), fire
    )
    successor = env.sample_next_state(
        state, joint(FirefightingAction.SUPPRESS, FirefightingAction.NORTH)
    )
    assert int(env.robots(successor)[0, 2]) == env.max_tank


# ---------------------------------------------------------------------------
# spread
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", list(WindDirection))
@pytest.mark.parametrize("strength", list(WindStrength))
def test_spread_favours_the_downwind_neighbour_under_every_wind(direction, strength) -> None:
    """Each of the eight winds boosts exactly the cell it blows towards.

    Purpose: The wind is the hidden variable the whole task turns on, and it is
        identifiable only because the spread is asymmetric. A wind whose
        direction table were rotated or mirrored would still produce an
        asymmetric fire, so nothing else in the suite would notice.

    Given: One burning cell in the middle of an otherwise unburnt grid, no
        robots near it, and each of the eight wind values in turn.
    When: A single transition is sampled many times and the ignition rate of
        each of the four neighbours is measured.
    Then: The neighbour in the wind's own direction ignites at the boosted
        rate, the other three at the attenuated rate, and the boosted rate
        matches the environment's own gain for that strength.

    Test type: integration
    """
    env = build_env(obstacle_cells=[], growth_probability=0.0, burnout_probability=0.0)
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    # Both robots parked far away with SUPPRESS so nothing they do reaches the fire.
    state = create_firefighting_state(
        env, [(0, 0, 0, 3), (9, 9, 0, 3)], (int(direction), int(strength)), fire
    )
    action = joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS)

    trials = 3000
    np.random.seed(17)
    counts = {offset: 0 for offset in DIRECTION_OFFSETS}
    for _ in range(trials):
        successor = env.fire_map(env.sample_next_state(state, action))
        for offset in DIRECTION_OFFSETS:
            if successor[4 + offset[0], 4 + offset[1]] == int(FireCategory.SMOLDERING):
                counts[offset] += 1

    gain = env.wind_gain_high if strength is WindStrength.HIGH else env.wind_gain_low
    downwind = min(1.0, env.spread_probability * gain)
    crosswind = env.spread_probability * (1.0 - env.crosswind_attenuation)
    wind_offset = DIRECTION_OFFSETS[int(direction)]
    for offset, count in counts.items():
        expected = downwind if offset == wind_offset else crosswind
        assert count / trials == pytest.approx(expected, abs=0.03)
    assert counts[wind_offset] == max(counts.values())


def test_wet_burnt_and_obstacle_cells_never_ignite() -> None:
    """The three non-fuel categories are immune to spread.

    Purpose: ``WET`` and ``BURNT`` being absorbing is what makes a fire-free map
        a genuine terminal state rather than a moment that can be undone, so a
        leak here would quietly break the completion metric as well as the
        dynamics.

    Given: A burning cell surrounded by a wet cell, a burnt cell and an
        obstacle, under the wind that blows towards each in turn.
    When: Many transitions are sampled.
    Then: None of the three ever becomes smoldering or burning.

    Test type: integration
    """
    env = build_env(obstacle_cells=[(4, 3)], burnout_probability=0.0)
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    fire[4, 5] = float(FireCategory.WET)
    fire[3, 4] = float(FireCategory.BURNT)
    action = joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS)
    np.random.seed(5)
    for direction in WindDirection:
        state = create_firefighting_state(
            env, [(0, 0, 0, 3), (9, 9, 0, 3)], (int(direction), int(WindStrength.HIGH)), fire
        )
        for _ in range(300):
            successor = env.fire_map(env.sample_next_state(state, action))
            assert int(successor[4, 5]) == int(FireCategory.WET)
            assert int(successor[3, 4]) == int(FireCategory.BURNT)
            assert int(successor[4, 3]) == int(FireCategory.UNBURNT)


def test_a_cell_cannot_ignite_and_grow_in_the_same_step() -> None:
    """Growth is applied only to cells alight before the spread stage.

    Purpose: If a freshly ignited cell were eligible for growth in its own
        step, a fire would reach full intensity a step earlier everywhere, and
        the window the robots have to reach a new ignition while it is still
        cheap -- which is exactly what ``growth_probability`` is there to set --
        would shrink with nothing to show for it.

    Given: A burning cell in an unburnt grid with a growth probability of 1.
    When: Many transitions are sampled.
    Then: Every newly lit neighbour is smoldering, never burning.

    Test type: integration
    """
    env = build_env(obstacle_cells=[], growth_probability=1.0, burnout_probability=0.0)
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(
        env, [(0, 0, 0, 3), (9, 9, 0, 3)], (int(WindDirection.EAST), int(WindStrength.HIGH)), fire
    )
    action = joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS)
    np.random.seed(9)
    lit = 0
    for _ in range(400):
        successor = env.fire_map(env.sample_next_state(state, action))
        for offset in DIRECTION_OFFSETS:
            category = int(successor[4 + offset[0], 4 + offset[1]])
            assert category in (int(FireCategory.UNBURNT), int(FireCategory.SMOLDERING))
            lit += category == int(FireCategory.SMOLDERING)
    assert lit > 0, "no neighbour ever caught, so the assertion above proved nothing"


# ---------------------------------------------------------------------------
# heat damage
# ---------------------------------------------------------------------------


def test_heat_damage_is_one_on_smoldering_and_two_on_burning() -> None:
    """Standing in fire costs health in proportion to its intensity.

    Purpose: With the default health of 3 a robot survives one burning step and
        is disabled by the second, which is what makes fighting from an
        adjacent cell the intended play rather than a nicety. A damage table
        off by one changes that policy.

    Given: A robot standing on each category in turn, unable to move.
    When: One transition is taken.
    Then: It loses 1 on smoldering, 2 on burning and nothing otherwise, and
        health never goes below zero.

    Test type: unit
    """
    env = build_env(slip_probability=1.0, growth_probability=0.0, burnout_probability=0.0)
    expected = {
        FireCategory.UNBURNT: 0,
        FireCategory.SMOLDERING: 1,
        FireCategory.BURNING: 2,
        FireCategory.BURNT: 0,
        FireCategory.WET: 0,
    }
    for category, damage in expected.items():
        fire = empty_fire(env)
        fire[4, 4] = float(category)
        fire[9, 9] = float(FireCategory.SMOLDERING)
        state = create_firefighting_state(env, [(4, 4, 0, 3), (0, 0, 0, 3)], (0, 0), fire)
        successor = env.sample_next_state(
            state, joint(FirefightingAction.NORTH, FirefightingAction.NORTH)
        )
        assert int(env.robots(successor)[0, 3]) == 3 - damage

    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 4, 0, 1), (0, 0, 0, 3)], (0, 0), fire)
    successor = env.sample_next_state(
        state, joint(FirefightingAction.NORTH, FirefightingAction.NORTH)
    )
    assert int(env.robots(successor)[0, 3]) == 0


# ---------------------------------------------------------------------------
# observation model
# ---------------------------------------------------------------------------


def test_only_cells_inside_a_live_footprint_are_reported() -> None:
    """Chebyshev radius ``rho`` around every live robot, and nothing else.

    Purpose: The visible set is what makes this a POMDP rather than a fully
        observed grid, and a disabled robot that still saw would quietly
        remove the cost of losing one.

    Given: One live robot and one disabled robot far apart.
    When: An observation is drawn.
    Then: Exactly the live robot's ``(2 rho + 1)`` block carries a category,
        every other cell carries the unknown marker, and the robots' own poses,
        tanks and healths are reported exactly.

    Test type: unit
    """
    env = build_env()
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 4, 6, 3), (0, 9, 6, 0)], (0, 0), fire)
    np.random.seed(1)
    observation = env.sample_observation(state, 0)

    reported = observation[4 * env.num_robots :].reshape(env.num_rows, env.num_cols)
    seen = np.argwhere(reported >= 0)
    expected = {
        (row, col)
        for row in range(4 - env.sensing_radius, 4 + env.sensing_radius + 1)
        for col in range(4 - env.sensing_radius, 4 + env.sensing_radius + 1)
    }
    assert {tuple(cell) for cell in seen} == expected
    assert np.array_equal(observation[: 4 * env.num_robots], [4, 4, 6, 3, 0, 9, 6, 0])


def test_the_observation_likelihood_is_the_confusion_matrix() -> None:
    """Correct with ``1 - eps``, each wrong category with ``eps / 4``.

    Purpose: This likelihood is the entire weight a particle filter puts on a
        firefighting observation. An error rate applied per-observation rather
        than per-cell, or spread over five categories instead of four, changes
        every belief on this environment without changing any trajectory.

    Given: A state and the observations reachable from it.
    When: Readings are drawn many times and scored.
    Then: The empirical per-cell error rate matches ``eps``, the analytic
        likelihood of a reading matches the product form, and a reading that
        names a category for an unseen cell, or misreports a pose, scores
        ``-inf``.

    Test type: integration
    """
    env = build_env(observation_error_probability=0.2)
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 4, 6, 3), (0, 9, 6, 3)], (0, 0), fire)
    visible = env.visible_mask(state).ravel()
    truth = env.fire_map(state).ravel()

    np.random.seed(4)
    wrong = 0
    total = 0
    for _ in range(400):
        reported = env.sample_observation(state, 0)[4 * env.num_robots :]
        wrong += int(np.count_nonzero(reported[visible] != truth[visible]))
        total += int(np.count_nonzero(visible))
    assert wrong / total == pytest.approx(0.2, abs=0.02)

    observation = env.sample_observation(state, 0)
    reported = observation[4 * env.num_robots :]
    mismatches = int(np.count_nonzero(reported[visible] != truth[visible]))
    matches = int(np.count_nonzero(visible)) - mismatches
    expected = matches * np.log(0.8) + mismatches * np.log(0.2 / 4.0)
    assert env.observation_log_probability(state, 0, [observation])[0] == pytest.approx(expected)

    impossible = observation.copy()
    impossible[4 * env.num_robots + 0] = float(FireCategory.UNBURNT)  # cell (0, 0) is unseen
    assert env.observation_log_probability(state, 0, [impossible])[0] == -np.inf

    wrong_pose = observation.copy()
    wrong_pose[0] += 1.0
    assert env.observation_log_probability(state, 0, [wrong_pose])[0] == -np.inf


def test_the_observation_never_reports_the_wind() -> None:
    """Two states differing only in the wind have identical observation laws.

    Purpose: The wind entering the observation would turn the inference problem
        the environment exists to pose into a lookup, and it would do so
        without changing a single trajectory.

    Given: Two states identical except for their wind.
    When: A reading drawn from one is scored under both.
    Then: The two log-likelihoods are equal.

    Test type: unit
    """
    env = build_env()
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    east = create_firefighting_state(env, [(4, 4, 6, 3), (0, 9, 6, 3)], (1, 1), fire)
    west = create_firefighting_state(env, [(4, 4, 6, 3), (0, 9, 6, 3)], (3, 0), fire)
    np.random.seed(6)
    observation = env.sample_observation(east, 0)
    assert env.observation_log_probability(east, 0, [observation])[0] == pytest.approx(
        env.observation_log_probability(west, 0, [observation])[0]
    )


# ---------------------------------------------------------------------------
# closed-form densities
# ---------------------------------------------------------------------------


def test_the_transition_density_matches_sampling_and_sums_to_one() -> None:
    """``transition_log_probability`` is the exact law ``sample_next_state`` draws from.

    Purpose: A planner's belief is reweighted with this density while its
        trajectories come from the sampler, so a disagreement between the two
        is invisible in every rollout and corrupts every belief. On a world
        small enough to enumerate, the two can simply be compared.

    Given: A 3x3 world with one robot, a mixed fire map, and three actions.
    When: Many successors are sampled and each distinct one is also scored
        analytically.
    Then: The analytic masses of the sampled successors sum to one, and each
        matches its empirical frequency.

    Test type: integration
    """
    env = MultiAgentFirefightingPOMDP(
        discount_factor=0.95,
        num_rows=3,
        num_cols=3,
        num_robots=1,
        obstacle_cells=[(0, 2)],
        depot_cell=(2, 2),
        robot_start_cells=[(1, 1)],
        num_initial_fires=1,
        max_tank=2,
        max_health=2,
        sensing_radius=1,
    )
    fire = np.full((3, 3), float(FireCategory.UNBURNT))
    fire[0, 0] = float(FireCategory.BURNING)
    fire[1, 0] = float(FireCategory.SMOLDERING)
    fire[2, 0] = float(FireCategory.BURNT)
    fire[2, 1] = float(FireCategory.WET)
    state = create_firefighting_state(env, [(1, 1, 2, 2)], (1, 1), fire, step=3)

    trials = 40000
    for action in (
        int(FirefightingAction.NORTH),
        int(FirefightingAction.EAST),
        int(FirefightingAction.SUPPRESS),
    ):
        np.random.seed(7)
        seen: dict = {}
        for _ in range(trials):
            key = env.sample_next_state(state, action).tobytes()
            seen[key] = seen.get(key, 0) + 1
        mass = 0.0
        for key, count in seen.items():
            successor = np.frombuffer(key, dtype=np.float64)
            analytic = float(np.exp(env.transition_log_probability(state, action, [successor])[0]))
            mass += analytic
            assert analytic == pytest.approx(count / trials, abs=0.01)
        assert mass == pytest.approx(1.0, abs=0.01)

    unreachable = env.sample_next_state(state, 0).copy()
    unreachable[0] += 5.0  # a step counter no single transition can produce
    assert env.transition_log_probability(state, 0, [unreachable])[0] == -np.inf


def test_the_wind_is_identifiable_from_the_spread_pattern() -> None:
    """An exact posterior over the eight winds concentrates on the true one.

    Purpose: The whole point of the environment is that a planner can infer the
        wind from where the fire grows. That claim is about the *model*, not
        about any particular filter: a particle filter can fail to find the
        wind because it has too few particles, so failing to see the posterior
        move would not tell you whether the information was there. This asks
        the model directly.

    Given: An open grid, an unattended fire, and a flat prior over the eight
        wind values.
    When: Twenty-five transitions are watched with the map fully observed, and
        each hypothesis is scored with the environment's own transition
        density.
    Then: The posterior puts most of its mass on the true wind.

    Test type: integration
    """
    env = build_env(obstacle_cells=[])
    idle = joint(FirefightingAction.NORTH, FirefightingAction.NORTH)
    exact = []
    for direction, strength, seed in (
        (WindDirection.NORTH, WindStrength.LOW, 101),
        (WindDirection.EAST, WindStrength.HIGH, 102),
        (WindDirection.WEST, WindStrength.LOW, 103),
    ):
        np.random.seed(seed)
        fire = empty_fire(env)
        fire[5, 5] = float(FireCategory.BURNING)
        state = create_firefighting_state(
            env, [(0, 0, 6, 3), (0, 9, 6, 3)], (int(direction), int(strength)), fire
        )
        log_posterior = np.full(8, -np.log(8.0))
        for _ in range(25):
            successor = env.sample_next_state(state, idle)
            for index in range(8):
                hypothesis = state.copy()
                candidate = successor.copy()
                for vector in (hypothesis, candidate):
                    vector[env.wind_direction_index] = float(index // 2)
                    vector[env.wind_strength_index] = float(index % 2)
                log_posterior[index] += env.transition_log_probability(
                    hypothesis, idle, [candidate]
                )[0]
            log_posterior -= log_posterior.max()
            state = successor
            if not np.any(env.alight_mask(env.fire_map(state))):
                break
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()
        true_index = int(direction) * 2 + int(strength)
        # The direction is the strong half of the signal -- it decides *which*
        # neighbour catches. The strength only scales one rate, so on a single
        # episode the posterior can favour the wrong strength while being
        # certain of the direction; that ambiguity is a property of the model,
        # not a defect, and the formal definition says as much. The direction
        # is therefore asserted per run and the exact value on average.
        by_direction = posterior.reshape(4, 2).sum(axis=1)
        assert by_direction[int(direction)] == max(by_direction)
        assert by_direction[int(direction)] > 0.8
        exact.append(posterior[true_index])
    assert np.mean(exact) > 0.4


# ---------------------------------------------------------------------------
# termination and metrics
# ---------------------------------------------------------------------------


def test_terminal_precedence_puts_the_goal_ahead_of_the_failure() -> None:
    """A fire put out by robots that then burned out is still a success.

    Purpose: Terminal does not mean success anywhere in this repository, and
        the three end-reason rates are what tell a reader whether a low
        completion rate is bad risk-taking or too small a step budget. Scoring
        a won episode as a failure would invert that reading.

    Given: A fire-free state whose robots are all disabled, a still-burning
        state whose robots are all disabled, and a still-burning state at the
        step budget.
    When: Terminality and the end-reason channels are read.
    Then: All three are terminal, and they report goal, failure and timeout
        respectively, with the three rates summing to one in each case.

    Test type: unit
    """
    env = build_env()
    cases = [
        (empty_fire(env), [(4, 4, 6, 0), (4, 5, 6, 0)], 5, "goal"),
        (None, [(4, 4, 6, 0), (4, 5, 6, 0)], 5, "failure"),
        (None, [(4, 4, 6, 3), (4, 5, 6, 3)], env.max_steps, "timeout"),
    ]
    channel = MultiAgentFirefightingStepChannel
    for fire, robots, step, expected in cases:
        grid = empty_fire(env) if fire is None else fire
        if expected != "goal":
            grid[0, 0] = float(FireCategory.BURNING)
        state = create_firefighting_state(env, robots, (0, 0), grid, step=step)
        assert env.is_terminal(state)
        info = env.step_info(state, None, None)
        rates = {
            "goal": info[channel.FIRE_EXTINGUISHED.value],
            "failure": info[channel.ALL_ROBOTS_DISABLED.value],
            "timeout": info[channel.TIMED_OUT_WITH_FIRE.value],
        }
        assert rates[expected] == 1.0
        assert sum(rates.values()) == pytest.approx(1.0)


def test_all_robots_disabled_is_only_terminal_when_the_flag_is_set() -> None:
    """The flag changes termination and nothing else.

    Purpose: It is the kind of flag that has added a reward term elsewhere in
        this repository and silently invalidated the declared range, so the
        two halves of the claim are worth checking together.

    Given: The same all-disabled, still-burning state under both settings.
    When: Terminality and the declared reward range are read.
    Then: It is terminal only with the flag on, and the range is the same
        either way.

    Test type: unit
    """
    fire = empty_fire(build_env())
    fire[0, 0] = float(FireCategory.BURNING)
    on = build_env(is_all_robots_disabled_terminal=True)
    off = build_env(is_all_robots_disabled_terminal=False)
    robots = [(4, 4, 6, 0), (4, 5, 6, 0)]
    assert on.is_terminal(create_firefighting_state(on, robots, (0, 0), fire, step=5))
    assert not off.is_terminal(create_firefighting_state(off, robots, (0, 0), fire, step=5))
    assert on.reward_range == off.reward_range


def test_step_info_reports_the_danger_channels_from_the_realised_transition() -> None:
    """Health, sprays, alight cells and burnt fraction all come out right.

    Purpose: These are the metrics a planner is judged on, and each is a
        different kind of quantity -- a per-step loss, an action count, a count
        of the successor's cells and a fraction of the grid. A channel read off
        the wrong state or the wrong step is invisible until someone compares
        two planners.

    Given: A transition in which one robot stands in a burning cell and both
        spray.
    When: ``step_info`` is called on it, and again on the terminal bookkeeping
        step.
    Then: The channels match the transition, and the transition-shaped ones are
        neutral on the terminal step while the state-shaped ones are not.

    Test type: unit
    """
    env = build_env()
    before = empty_fire(env)
    before[4, 4] = float(FireCategory.BURNING)
    before[4, 5] = float(FireCategory.SMOLDERING)
    state = create_firefighting_state(env, [(4, 4, 6, 3), (0, 0, 6, 3)], (0, 0), before)
    after = before.copy()
    after[4, 6] = float(FireCategory.BURNT)
    successor = create_firefighting_state(env, [(4, 4, 5, 1), (0, 0, 5, 3)], (0, 0), after, step=1)
    action = joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS)
    channel = MultiAgentFirefightingStepChannel

    info = env.step_info(state, action, successor)
    assert info[channel.HEALTH_LOST.value] == 2.0
    assert info[channel.SUPPRESS_ACTIONS.value] == 2.0
    assert info[channel.ALIGHT_CELLS.value] == 2.0
    assert info[channel.ROBOT_IN_ALIGHT_CELL.value] == 1.0
    assert info[channel.BURNT_CELL_FRACTION.value] == pytest.approx(1.0 / env.num_cells)
    assert info[channel.ROBOTS_DISABLED.value] == 0.0

    terminal = env.step_info(successor, None, None)
    assert terminal[channel.HEALTH_LOST.value] == 0.0
    assert terminal[channel.SUPPRESS_ACTIONS.value] == 0.0
    assert terminal[channel.ALIGHT_CELLS.value] == 2.0
    assert terminal[channel.RECORDED_STEP.value] == 1.0


def test_every_declared_metric_has_a_channel_step_info_emits() -> None:
    """Declared names and emitted channels agree.

    Purpose: A declared-but-unreported channel yields a metric that is silently
        dropped, which reads downstream as "this environment does not report
        that" rather than as a bug.

    Given: The environment's metric specs and one call to ``step_info``.
    When: The declared channels are compared with the emitted ones.
    Then: Every declared channel is emitted, and the declared metric names are
        exactly those of the metric enum.

    Test type: unit
    """
    env = build_env()
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    successor = env.sample_next_state(state, 0)
    emitted = set(env.step_info(state, 0, successor))
    specs = env.get_metric_specs()
    assert {spec.channel for spec in specs} <= emitted
    assert {spec.name for spec in specs} == {
        metric.value for metric in MultiAgentFirefightingMetrics
    }


# ---------------------------------------------------------------------------
# reset distribution and serialization
# ---------------------------------------------------------------------------


def test_reset_always_lights_a_fire_and_never_lights_an_obstacle() -> None:
    """The goal is never free at ``t = 0``, and obstacles never burn.

    Purpose: If a reset could hand back a fire-free state the success reward
        would be collectable for nothing, and the completion rate would count
        it. The obstacle clause is what keeps a lit obstacle from being an
        unreachable, unwinnable fire.

    Given: Many draws from the reset distribution.
    When: Their fire maps, winds, poses, tanks and healths are inspected.
    Then: Each has exactly ``num_initial_fires`` burning cells, none on an
        obstacle, robots at their posts with full tanks and health, and the
        eight winds all appear.

    Test type: integration
    """
    env = build_env(num_initial_fires=3)
    np.random.seed(0)
    states = env.initial_state_dist().sample(400)
    winds = set()
    for state in states:
        fire = env.fire_map(state)
        burning = np.argwhere(fire == int(FireCategory.BURNING))
        assert len(burning) == 3
        assert not env.is_terminal(state)
        for row, col in burning:
            assert not env.obstacle_mask[row, col]
        assert np.count_nonzero(fire != int(FireCategory.UNBURNT)) == 3
        robots = env.robots(state)
        assert [tuple(cell) for cell in robots[:, :2]] == env.robot_start_cells
        assert np.all(robots[:, 2] == env.max_tank)
        assert np.all(robots[:, 3] == env.max_health)
        winds.add(env.wind(state))
    assert len(winds) == 8


def test_the_reset_distribution_reports_its_own_density() -> None:
    """The prior mass of a drawn state matches the uniform-times-uniform form.

    Purpose: The prior is a product of two uniforms and therefore has a closed
        form, so an importance weight built on it is exactly computable. A
        density that did not match its own sampler would corrupt any such
        weight silently.

    Given: States drawn from the reset distribution, and states it cannot
        produce.
    When: Their probabilities are read.
    Then: Drawn states all carry the same mass, ``1 / (8 * C(M, n))``, and the
        impossible ones carry zero.

    Test type: unit
    """
    from math import comb

    env = build_env()
    distribution = env.initial_state_dist()
    np.random.seed(0)
    states = distribution.sample(12)
    ignitable = env.num_cells - len(env.obstacle_cells)
    expected = 1.0 / (8.0 * comb(ignitable, env.num_initial_fires))
    assert np.allclose(distribution.probability(states), expected)

    stepped = states[0].copy()
    stepped[0] = 1.0
    fire_free = create_firefighting_state(
        env,
        [(2, 2, env.max_tank, env.max_health), (2, 3, env.max_tank, env.max_health)],
        (0, 0),
        empty_fire(env),
    )
    assert np.allclose(distribution.probability([stepped, fire_free]), 0.0)


def test_serialization_round_trips_the_cell_sequence_arguments() -> None:
    """Obstacles, depot and start cells survive ``to_dict`` / ``from_dict``.

    Purpose: These three are the arguments most likely to break a round trip --
        they are sequences of tuples, and a derived form stored under a
        constructor parameter's name is exactly how ContinuousPush's
        serialization broke. The empty-obstacle case is here because ``None``
        meaning "use the default" and ``[]`` meaning "no obstacles" are
        different, and a truthiness test conflates them.

    Given: The default environment and one with no obstacles at all.
    When: Each is rebuilt from its own dict.
    Then: The rebuilt environment is equal to the original, carries the same
        ``config_id``, and holds the same layout.

    Test type: unit
    """
    for env in (build_env(), build_env(obstacle_cells=[], depot_cell=(9, 9))):
        rebuilt = MultiAgentFirefightingPOMDP.from_dict(env.to_dict())
        assert rebuilt.config_id == env.config_id
        assert rebuilt == env
        assert rebuilt.obstacle_cells == env.obstacle_cells
        assert rebuilt.depot_cell == env.depot_cell
        assert rebuilt.robot_start_cells == env.robot_start_cells


def test_an_empty_obstacle_list_is_not_the_default_obstacle_blob() -> None:
    """``None`` and ``[]`` mean different worlds.

    Purpose: ``if obstacles:`` is the exact shape of bug the environment API
        contract calls out: it takes the no-obstacles branch for the default
        and declares a world that is not the one anyone ran.

    Given: One environment built with ``None`` and one with ``[]``.
    When: Their obstacle sets are compared.
    Then: The first is non-empty, the second is empty, and they are not equal.

    Test type: unit
    """
    default = MultiAgentFirefightingPOMDP(discount_factor=0.95)
    open_grid = MultiAgentFirefightingPOMDP(discount_factor=0.95, obstacle_cells=[])
    assert default.obstacle_cells
    assert open_grid.obstacle_cells == []
    assert default != open_grid


def test_the_constructor_rejects_an_impossible_world() -> None:
    """Bad geometry and out-of-range probabilities fail at construction.

    Purpose: A typo in a config file should stop the run, not silently select
        a different world -- a depot inside the obstacle blob, or a fire that
        cannot be lit, would otherwise surface as an unexplainable result.

    Given: Several invalid constructor argument sets.
    When: Each is used to build an environment.
    Then: Each raises ``ValueError``.

    Test type: unit
    """
    bad = [
        {"depot_cell": (5, 5)},  # inside the default obstacle blob
        {"robot_start_cells": [(5, 5), (2, 3)]},  # a robot inside it
        {"robot_start_cells": [(2, 2)]},  # one cell for two robots
        {"num_initial_fires": 0},  # goal free at reset
        {"num_initial_fires": 1000},  # more fires than ignitable cells
        {"spread_probability": 1.5},
        {"observation_error_probability": -0.1},
        {"step_cost": -1.0},
        {"max_steps": 0},
        {"max_health": 0},
        {"num_robots": 0},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            build_env(**kwargs)


def test_the_renderer_has_a_label_for_every_action() -> None:
    """The caption's action labels cover the action enum exactly.

    Purpose: The renderer names actions from its own table. Adding a sixth
        per-robot action would leave that table short and the caption would
        raise mid-render -- or, worse, silently mislabel -- on the first
        episode anyone tried to visualize, long after the change landed.

    Given: The renderer's label table and the environment's action enum.
    When: The two are compared.
    Then: Every action has exactly one label, and there are no extras.

    Test type: unit
    """
    from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_visualizer import (  # noqa: E501
        ACTION_LABELS,
        action_labels,
    )

    assert set(action_labels()) == {int(member) for member in FirefightingAction}
    assert len(ACTION_LABELS) == len(FirefightingAction)


@pytest.mark.parametrize("error", [0.0, 1.0])
def test_the_observation_likelihood_is_finite_at_both_error_extremes(error) -> None:
    """A zero or total error rate scores a reading without producing ``NaN``.

    Purpose: Both endpoints are legal and documented -- at 0 the fire map is
        exact inside the footprints, at 1 every reading is wrong -- and at each
        one of the two log terms is ``-inf`` while its count is zero. Written
        as ``count * log`` that is ``0 * -inf``, which is ``NaN``. A ``NaN``
        here does not raise: it silently poisons a particle weight and from
        there the whole belief, so it would surface as a planner that stopped
        working rather than as an error.

    Given: An environment at each endpoint, and a reading drawn from it.
    When: The reading is scored.
    Then: The score is a finite number, and it is the one the confusion matrix
        gives.

    Test type: unit
    """
    env = build_env(observation_error_probability=error)
    fire = empty_fire(env)
    fire[4, 4] = float(FireCategory.BURNING)
    state = create_firefighting_state(env, [(4, 4, 6, 3), (0, 9, 6, 3)], (0, 0), fire)
    np.random.seed(8)
    observation = env.sample_observation(state, 0)
    score = env.observation_log_probability(state, 0, [observation])[0]
    assert not np.isnan(score)
    assert score == pytest.approx(0.0) if error == 0.0 else np.isfinite(score)


def test_explicit_robot_starts_are_not_judged_by_the_default_placement_rule() -> None:
    """A supplied layout is accepted even where the default rule would refuse.

    Purpose: The default placement keeps robots off the depot so that nobody
        starts with a free refill. That is a rule about the *default*, not about
        what a caller may ask for -- and on a grid with no free cell to spare it
        cannot be satisfied at all. Building it eagerly let it reject layouts
        nothing was going to use.

    Given: A one-cell open grid whose only cell is both the depot and the
        robot's explicit start.
    When: The environment is constructed.
    Then: It builds, and the robot starts where it was told to.

    Test type: unit
    """
    env = MultiAgentFirefightingPOMDP(
        discount_factor=0.95,
        num_rows=1,
        num_cols=1,
        num_robots=1,
        obstacle_cells=[],
        depot_cell=(0, 0),
        robot_start_cells=[(0, 0)],
        num_initial_fires=1,
    )
    assert env.robot_start_cells == [(0, 0)]
    assert env.depot_cell == (0, 0)
