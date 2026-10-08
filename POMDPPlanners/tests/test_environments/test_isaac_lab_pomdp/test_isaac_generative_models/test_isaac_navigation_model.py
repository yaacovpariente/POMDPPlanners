# SPDX-License-Identifier: MIT

"""Behaviour tests for the goal-relative navigation model.

The load-bearing claim is that propagating a base-frame goal is the same motion as driving a base
around a floor, only bookkept in the moving frame. The first test pins exactly that: it integrates
a base in world coordinates with the validated unicycle, recomputes where the goal *would* appear
in the new base frame, and demands the goal-relative model already said so. Everything the planner
does rests on that identity, and nothing else in the code would notice if it broke.

The action-separation tests are the regression guard for the measured failure this model exists to
fix -- under the fitted linear surrogate every action scored alike and the planner emitted one
index for a whole episode.
"""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    IsaacChannelSchema,
    NavigationIsaacModel,
    NavigationRewardModel,
    UnicycleTransition,
    wrap_angle,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model import (  # noqa: E501  pylint: disable=line-too-long
    GoalRelativeTransition,
)

STEP_DT = 0.2
SCHEMA = IsaacChannelSchema((("base_lin_vel", 3), ("projected_gravity", 3), ("pose_command", 4)))
PRESETS = [
    np.zeros(3),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.6, -0.4, 0.5]),
]
CLEAN_NOISE = {
    "velocity_noise_std": 1e-9,
    "position_noise_std": 1e-9,
    "heading_noise_std": 1e-9,
}


def _transition(**overrides: Any) -> GoalRelativeTransition:
    settings: dict = {"step_dt": STEP_DT, **CLEAN_NOISE}
    settings.update(overrides)
    return GoalRelativeTransition(**settings)


def _model(**overrides: Any) -> NavigationIsaacModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": PRESETS,
        "discount_factor": 0.99,
        "step_dt": STEP_DT,
        **CLEAN_NOISE,
    }
    settings.update(overrides)
    return NavigationIsaacModel(**settings)


def _driven(goal_xy: Any, heading: float) -> np.ndarray:
    """A driven block ``[base_lin_vel(3), pose_command(4)]`` with the base at rest."""
    return np.array([0.0, 0.0, 0.0, goal_xy[0], goal_xy[1], 0.0, heading])


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array(
        [
            cos_a * vector[0] - sin_a * vector[1],
            sin_a * vector[0] + cos_a * vector[1],
        ]
    )


# ── The identity the whole model rests on ───────────────────────────────


@pytest.mark.parametrize("action", PRESETS)
def test_goal_propagation_agrees_with_driving_the_base_in_world_coordinates(
    action: np.ndarray,
) -> None:
    """Propagating the goal must be the same motion as driving the base, seen from the base.

    Purpose: Pins the goal-relative update to the validated unicycle integration in world
        coordinates, which is the only reason the model is entitled to plan without a base position

    Given: A base at a non-trivial pose with a fixed world-frame goal and goal heading
    When: The base is driven one step by UnicycleTransition and the goal is re-expressed in the
        new base frame by hand, while GoalRelativeTransition predicts the same quantity directly
    Then: The two base-frame goals and headings agree to floating-point tolerance

    Test type: unit
    """
    base_xy, base_yaw = np.array([1.0, -2.0]), 0.7
    goal_world, goal_heading_world = np.array([4.0, 1.0]), 2.0

    goal_base = _rotate(goal_world - base_xy, -base_yaw)
    heading_base = float(wrap_angle(goal_heading_world - base_yaw))

    unicycle = UnicycleTransition(step_dt=STEP_DT, process_noise_std=1e-9)
    moved = unicycle.body_frame_delta(action)
    next_xy = base_xy + _rotate(moved[:2], base_yaw)
    next_yaw = base_yaw + moved[2]

    expected_goal = _rotate(goal_world - next_xy, -next_yaw)
    expected_heading = float(wrap_angle(goal_heading_world - next_yaw))

    predicted = _transition().predict_next(_driven(goal_base, heading_base), action)[0]

    np.testing.assert_allclose(predicted[3:5], expected_goal, atol=1e-9)
    assert float(wrap_angle(predicted[6] - expected_heading)) == pytest.approx(0.0, abs=1e-9)


# ── Dynamics ────────────────────────────────────────────────────────────


def test_driving_at_the_goal_closes_the_gap_and_driving_away_opens_it() -> None:
    """A model whose distance does not respond to the command gives a search nothing to climb.

    Purpose: Validates the sign of the position update under opposed commands

    Given: A goal two metres straight ahead of a stationary base
    When: A full-forward and a full-reverse command are each applied for one step
    Then: The forward command reduces the goal distance and the reverse command increases it

    Test type: unit
    """
    transition = _transition()
    state = _driven(np.array([2.0, 0.0]), 0.0)
    forward = transition.predict_next(state, np.array([1.0, 0.0, 0.0]))[0]
    reverse = transition.predict_next(state, np.array([-1.0, 0.0, 0.0]))[0]
    assert float(np.linalg.norm(forward[3:5])) < 2.0
    assert float(np.linalg.norm(reverse[3:5])) > 2.0


def test_turning_swings_the_goal_the_opposite_way_and_closes_the_heading_error() -> None:
    """Turning is the half a fitted linear map over a position-free observation cannot learn.

    Purpose: Validates the rotation half of the goal-relative update

    Given: A goal ahead of the base and a positive commanded heading error
    When: A pure left-turn command is applied for one step
    Then: The goal rotates clockwise in the base frame and the heading error shrinks

    Test type: unit
    """
    transition = _transition()
    state = _driven(np.array([2.0, 0.0]), 0.6)
    turned = transition.predict_next(state, np.array([0.0, 0.0, 1.0]))[0]
    assert turned[4] < 0.0  # a left turn puts a goal that was dead ahead on the right
    assert abs(turned[6]) < 0.6


def test_tracking_scales_shorten_the_predicted_step() -> None:
    """Assuming perfect tracking is the model's most likely quiet failure, so it must be a knob.

    Purpose: Validates that the calibrated scales actually reduce the predicted motion

    Given: A pure translation and a pure turn, each under perfect and under half tracking
    When: Both models predict one step from an identical state
    Then: The half-tracking model covers half the ground and turns half as far

    Test type: unit
    """
    state = _driven(np.array([3.0, 0.0]), 0.0)
    # Measured one command at a time: with a simultaneous turn the goal's x entry mixes the
    # translation and the rotation, so it is not linear in either scale on its own.
    forward = np.array([1.0, 0.0, 0.0])
    perfect = _transition().predict_next(state, forward)[0]
    halved = _transition(linear_scale=0.5).predict_next(state, forward)[0]
    assert 3.0 - halved[3] == pytest.approx((3.0 - perfect[3]) * 0.5, rel=1e-6)

    turn = np.array([0.0, 0.0, 1.0])
    perfect_turn = _transition().predict_next(state, turn)[0]
    halved_turn = _transition(angular_scale=0.5).predict_next(state, turn)[0]
    assert halved_turn[6] == pytest.approx(perfect_turn[6] * 0.5, rel=1e-6)


def test_the_tracked_base_velocity_follows_the_scaled_command() -> None:
    """The velocity block is observed, so a model that ignores it mis-weights every particle.

    Purpose: Validates that base_lin_vel is predicted as the scaled linear command

    Given: A model tracking 80% of its linear command
    When: A sideways-and-forward command is applied
    Then: The predicted body-frame velocity is 0.8 times the command, with zero vertical component

    Test type: unit
    """
    predicted = _transition(linear_scale=0.8).predict_next(
        _driven(np.array([1.0, 1.0]), 0.0), np.array([0.5, -0.25, 1.0])
    )[0]
    np.testing.assert_allclose(predicted[:3], [0.4, -0.2, 0.0], atol=1e-9)


def test_distinct_commands_lead_to_distinct_predicted_goals() -> None:
    """The fitted linear model failed by scoring every action alike; this is that regression guard.

    Purpose: Validates that the preset action set is separated by the model's own predictions

    Given: The five preset velocity commands and one shared state
    When: Each is applied for a step
    Then: Every predicted base-frame goal position is distinct

    Test type: unit
    """
    transition = _transition()
    state = _driven(np.array([2.3, 1.4]), 0.5)
    goals = np.asarray([transition.predict_next(state, action)[0][3:5] for action in PRESETS])
    separations = np.linalg.norm(goals[:, None, :] - goals[None, :, :], axis=-1)
    off_diagonal = separations[~np.eye(len(PRESETS), dtype=bool)]
    assert float(off_diagonal.min()) > 1e-6


def test_the_transition_density_treats_the_heading_as_an_angle() -> None:
    """An unwrapped heading residual would call a numerically identical candidate impossible.

    Purpose: Validates that the log-density wraps the heading residual

    Given: A predicted next state and the same state with its heading shifted by a full turn
    When: Both are scored under the transition density
    Then: They receive the same log-density

    Test type: unit
    """
    transition = _transition(heading_noise_std=0.05)
    state = _driven(np.array([1.0, 0.5]), 0.2)
    action = np.array([0.5, 0.0, 0.3])
    predicted = transition.predict_next(state, action)[0]
    wrapped_away = predicted.copy()
    wrapped_away[6] += 2.0 * np.pi
    scores = transition.log_probability(state, action, np.stack([predicted, wrapped_away]))
    assert scores[0] == pytest.approx(scores[1], rel=1e-9)


def test_sampled_headings_stay_inside_one_revolution() -> None:
    """A heading drifting outside (-pi, pi] would break every downstream angle comparison.

    Purpose: Validates that sampling re-wraps the heading after adding noise

    Given: A transition with heading noise wide enough to push samples past pi
    When: Many next states are sampled from a state already near the wrap boundary
    Then: Every sampled heading lies in (-pi, pi]

    Test type: unit
    """
    np.random.seed(0)
    transition = _transition(heading_noise_std=0.8)
    samples = transition.sample_next_state(_driven(np.array([1.0, 0.0]), 3.0), PRESETS[3], 512)
    assert np.all(samples[:, 6] > -np.pi) and np.all(samples[:, 6] <= np.pi)


def test_projected_gravity_is_carried_through_the_transition() -> None:
    """Level ground is configuration, not dynamics; resampling it would inject noise for nothing.

    Purpose: Validates that undriven state blocks survive a model step unchanged

    Given: A state whose projected_gravity block is the flat-ground value
    When: The model takes one step under a non-trivial command
    Then: The projected_gravity block is bit-identical

    Test type: unit
    """
    state = SCHEMA.pack(
        {
            "base_lin_vel": np.zeros(3),
            "projected_gravity": [0.0, 0.0, -1.0],
            "pose_command": [2.0, 0.0, 0.0, 0.3],
        }
    )
    moved = _model().sample_next_state(state, PRESETS[4])
    assert SCHEMA.block(moved, "projected_gravity").tolist() == [0.0, 0.0, -1.0]


# ── Reward ──────────────────────────────────────────────────────────────


def test_the_reward_reproduces_the_task_reward_terms() -> None:
    """The point of an analytic model is optimizing the task's objective, not a regression onto it.

    Purpose: Pins the reward to IsaacLab's own position_command_error_tanh and
        heading_command_error_abs terms at the task's configured weights

    Given: A base-frame command two metres out with a 0.4 rad heading error
    When: The model scores it
    Then: The reward equals 0.5*(1-tanh(d/2)) + 0.5*(1-tanh(d/0.2)) - 0.2*|heading|

    Test type: unit
    """
    reward_model = NavigationRewardModel(state_schema=SCHEMA)
    state = SCHEMA.pack(
        {
            "base_lin_vel": np.zeros(3),
            "projected_gravity": [0.0, 0.0, -1.0],
            "pose_command": [1.6, 1.2, 0.0, 0.4],
        }
    )
    distance = 2.0
    expected = (
        0.5 * (1.0 - np.tanh(distance / 2.0)) + 0.5 * (1.0 - np.tanh(distance / 0.2)) - 0.2 * 0.4
    )
    assert reward_model.reward(state, None, state) == pytest.approx(float(expected), rel=1e-9)


def test_the_reward_rises_monotonically_as_the_base_approaches_the_goal() -> None:
    """A non-monotone objective would reward stopping short, which no search could recover from.

    Purpose: Validates the reward's ordering across the distances an episode actually spans

    Given: Base-frame commands from three metres in to zero, all with no heading error
    When: Each is scored
    Then: The rewards strictly increase as the distance falls

    Test type: unit
    """
    reward_model = NavigationRewardModel(state_schema=SCHEMA)
    rewards = [
        reward_model.reward(
            SCHEMA.pack(
                {
                    "base_lin_vel": np.zeros(3),
                    "projected_gravity": [0.0, 0.0, -1.0],
                    "pose_command": [metres, 0.0, 0.0, 0.0],
                }
            ),
            None,
            None,
        )
        for metres in [3.0, 2.0, 1.0, 0.5, 0.1, 0.0]
    ]
    assert all(later > earlier for earlier, later in zip(rewards, rewards[1:]))


def test_the_planar_goal_distance_ignores_the_height_gap() -> None:
    """The success predicate is planar; scoring it on a 3-D norm would credit an unreachable gap.

    Purpose: Validates that planar_goal_distance drops the z entry the objective keeps

    Given: A command 3 m ahead and 4 m above the base
    When: Both distance accessors are read
    Then: The objective's distance is 5 m and the planar distance is 3 m

    Test type: unit
    """
    reward_model = NavigationRewardModel(state_schema=SCHEMA)
    state = SCHEMA.pack(
        {
            "base_lin_vel": np.zeros(3),
            "projected_gravity": [0.0, 0.0, -1.0],
            "pose_command": [3.0, 0.0, 4.0, 0.0],
        }
    )
    assert reward_model.goal_distance(state) == pytest.approx(5.0)
    assert reward_model.planar_goal_distance(state) == pytest.approx(3.0)


# ── Configuration guards ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"step_dt": 0.0},
        {"linear_scale": 0.0},
        {"angular_scale": -0.5},
        {"velocity_noise_std": 0.0},
        {"position_noise_std": -1.0},
        {"heading_noise_std": 0.0},
    ],
)
def test_the_transition_rejects_a_configuration_it_cannot_represent(overrides: dict) -> None:
    """A silently accepted zero std makes every particle weight -inf, far from where it was set.

    Purpose: Validates constructor validation of the transition's numeric arguments

    Given: A configuration with a non-positive step, tracking scale or noise std
    When: The transition is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError):
        _transition(**overrides)


@pytest.mark.parametrize(
    "channels",
    [
        (("base_lin_vel", 2), ("projected_gravity", 3), ("pose_command", 4)),
        (("base_lin_vel", 3), ("projected_gravity", 3), ("pose_command", 3)),
    ],
)
def test_the_model_rejects_a_schema_whose_driven_blocks_are_the_wrong_width(
    channels: tuple,
) -> None:
    """A mis-sized block would not raise, it would silently read a neighbouring channel as a goal.

    Purpose: Validates that the model checks both driven channel widths against the schema

    Given: A schema whose velocity or command block is the wrong width
    When: The model is constructed over it
    Then: ValueError is raised naming the offending channel

    Test type: unit
    """
    with pytest.raises(ValueError, match="wide"):
        _model(state_schema=IsaacChannelSchema(channels))


def test_a_supplied_reward_model_replaces_the_analytic_one() -> None:
    """The vectorized mirror refuses a model it cannot mirror, so this flag must be honest.

    Purpose: Validates that navigation_reward is None when a different objective is supplied

    Given: A model built with an explicit non-navigation reward model
    When: Its navigation_reward attribute is read
    Then: It is None, and the default construction sets it

    Test type: unit
    """

    class _Flat:
        def reward(self, state: Any, action: Any, next_state: Any) -> float:
            del state, action, next_state
            return 0.0

    assert _model(reward_model=_Flat()).navigation_reward is None
    assert _model().navigation_reward is not None
