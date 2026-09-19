# SPDX-License-Identifier: MIT

"""Per-step channels and metrics of the Snake POMDP.

A completion rate on its own is ambiguous: 20% completion with the rest dying to
walls is a reckless planner, and 20% with the rest starving is a planner that
never found the food. These tests check the channels that tell those apart, and
that the three end-reason rates really do partition the episodes.
"""

import numpy as np
import pytest

from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction
from POMDPPlanners.environments.snake_pomdp.snake_belief import SnakeBelief
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    SnakeAction,
    SnakePOMDP,
    SnakePOMDPMetrics,
    SnakeStepChannel,
    create_snake_state,
)


def build_env(**overrides):
    """Return a small Snake environment, with overrides applied."""
    kwargs = {
        "grid_size": 7,
        "target_length": 6,
        "starvation_limit": 8,
        "discount_factor": 0.98,
    }
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs)


def state_for(env, body, food, steps_since_food=0):
    """Build a running state for ``env``."""
    return create_snake_state(
        body=body,
        food=food,
        steps_since_food=steps_since_food,
        target_length=env.target_length,
    )


def history_of(env, steps, terminated):
    """Wrap recorded steps into a :class:`History`."""
    return History(
        history=steps,
        discount_factor=env.discount_factor,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=len(steps),
        reach_terminal_state=terminated,
        policy_run_data=[],
    )


def run_episode(env, actions, state):
    """Play a fixed action sequence and record it the way the runner does."""
    belief = SnakeBelief.from_environment(env, n_particles=8)
    steps = []
    terminated = False
    for action in actions:
        if env.is_terminal(state):
            terminated = True
            break
        next_state, observation, reward = env.sample_next_step(state, int(action))
        steps.append(
            StepData(
                state=state,
                action=int(action),
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=belief,
                info=env.step_info(state, int(action), next_state),
            )
        )
        state = next_state
    if env.is_terminal(state):
        terminated = True
        steps.append(
            StepData(
                state=state,
                action=None,
                next_state=None,
                observation=None,
                reward=None,
                belief=belief,
                info=env.step_info(state, None, None),
            )
        )
    return history_of(env, steps, terminated)


def metrics_of(env, histories):
    """Return the environment's metrics keyed by name."""
    return {metric.name: metric.value for metric in env.compute_metrics(histories)}


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def test_every_declared_channel_is_emitted():
    """A declared-but-unreported channel yields a silently dropped metric.

    Test type: unit
    """
    env = build_env()
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    reported = set(env.step_info(state, int(SnakeAction.GO_STRAIGHT), state))
    declared = {spec.channel for spec in env.get_metric_specs()}
    assert declared <= reported
    assert reported == {channel.value for channel in SnakeStepChannel}


def test_step_info_tolerates_the_terminal_bookkeeping_step():
    """The final state is recorded with no action and no successor.

    Test type: unit
    """
    env = build_env()
    dead = env.sample_next_state(
        state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5)), int(SnakeAction.GO_STRAIGHT)
    )
    info = env.step_info(dead, None, None)
    assert info[SnakeStepChannel.FOOD_EATEN.value] == 0.0
    assert info[SnakeStepChannel.WALL_DEATH.value] == 1.0
    assert info[SnakeStepChannel.EPISODE_FAILURE.value] == 1.0


def test_step_info_consumes_no_randomness():
    """A draw here would shift the stream for every later transition.

    Test type: unit
    """
    env = build_env()
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    np.random.seed(5)
    before = np.random.random()
    np.random.seed(5)
    env.step_info(state, int(SnakeAction.GO_STRAIGHT), state)
    assert np.random.random() == before


def test_the_eat_channel_reports_the_step_that_reaches_the_food():
    """Food eaten is scored against the state the step was taken from.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4))
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    info = env.step_info(state, int(SnakeAction.GO_STRAIGHT), successor)
    assert info[SnakeStepChannel.FOOD_EATEN.value] == 1.0


def test_the_death_channels_name_the_cause():
    """Wall, self and starvation are reported separately, not as one failure.

    Purpose: They call for opposite fixes -- a planner that walks into walls is
        searching badly, one that starves is not searching at all.

    Test type: unit
    """
    env = build_env(starvation_limit=4)
    wall = env.sample_next_state(
        state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5)), int(SnakeAction.GO_STRAIGHT)
    )
    starved = env.sample_next_state(
        state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 6), steps_since_food=3),
        int(SnakeAction.GO_STRAIGHT),
    )
    assert env.step_info(wall, None, None)[SnakeStepChannel.WALL_DEATH.value] == 1.0
    assert env.step_info(wall, None, None)[SnakeStepChannel.STARVATION_DEATH.value] == 0.0
    assert env.step_info(starved, None, None)[SnakeStepChannel.STARVATION_DEATH.value] == 1.0


def test_progress_is_read_from_the_successor_when_there_is_one():
    """An episode whose last allowed step wins records no bookkeeping step.

    Purpose: The episode runner checks its step budget before terminality, so
        reading progress from ``state`` alone would score that win as a timeout.

    Test type: unit
    """
    env = build_env(target_length=4)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4))
    won = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    info = env.step_info(state, int(SnakeAction.GO_STRAIGHT), won)
    assert info[SnakeStepChannel.TARGET_LENGTH_REACHED.value] == 1.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_the_declared_metric_names_are_the_ones_produced():
    """``get_metric_names`` and ``compute_metrics`` cannot drift apart.

    Test type: integration
    """
    env = build_env()
    np.random.seed(0)
    history = run_episode(
        env, [int(SnakeAction.GO_STRAIGHT)] * 6, env.initial_state_dist().sample()[0]
    )
    assert set(metrics_of(env, [history])) == set(env.get_metric_names())
    assert set(env.get_metric_names()) == {metric.value for metric in SnakePOMDPMetrics}


def test_completion_reduces_with_any_and_the_end_reasons_with_last():
    """Winning is a once-and-stays-true event; the end reasons are the last step.

    Test type: unit
    """
    env = build_env()
    specs = {spec.name: spec for spec in env.get_metric_specs()}
    assert specs[SnakePOMDPMetrics.TASK_COMPLETION_RATE.value].per_episode is EpisodeReduction.ANY
    for name in (
        SnakePOMDPMetrics.ENDED_BY_GOAL,
        SnakePOMDPMetrics.ENDED_BY_FAILURE,
        SnakePOMDPMetrics.ENDED_BY_TIMEOUT,
    ):
        assert specs[name.value].per_episode is EpisodeReduction.LAST
    assert specs[SnakePOMDPMetrics.AVERAGE_EPISODE_LENGTH.value].per_episode is EpisodeReduction.SUM


def test_a_won_episode_scores_as_a_goal():
    """The three end-reason rates partition a completed episode onto the goal.

    Test type: integration
    """
    env = build_env(target_length=4)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4))
    np.random.seed(0)
    history = run_episode(env, [int(SnakeAction.GO_STRAIGHT)] * 3, state)
    values = metrics_of(env, [history])
    assert values[SnakePOMDPMetrics.TASK_COMPLETION_RATE.value] == 1.0
    assert values[SnakePOMDPMetrics.ENDED_BY_GOAL.value] == 1.0
    assert values[SnakePOMDPMetrics.ENDED_BY_FAILURE.value] == 0.0
    assert values[SnakePOMDPMetrics.ENDED_BY_TIMEOUT.value] == 0.0
    assert values[SnakePOMDPMetrics.AVERAGE_FOOD_EATEN.value] == 1.0
    assert values[SnakePOMDPMetrics.MAX_SNAKE_LENGTH.value] == 4.0


def test_a_wall_death_scores_as_a_failure():
    """The end reasons still sum to one when the episode ended badly.

    Test type: integration
    """
    env = build_env()
    state = state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5))
    np.random.seed(0)
    history = run_episode(env, [int(SnakeAction.GO_STRAIGHT)] * 3, state)
    values = metrics_of(env, [history])
    assert values[SnakePOMDPMetrics.ENDED_BY_FAILURE.value] == 1.0
    assert values[SnakePOMDPMetrics.ENDED_BY_GOAL.value] == 0.0
    assert values[SnakePOMDPMetrics.ENDED_BY_TIMEOUT.value] == 0.0
    assert values[SnakePOMDPMetrics.WALL_DEATH_RATE.value] == 1.0


def test_an_unfinished_episode_scores_as_a_timeout_and_not_a_failure():
    """Running out of the runner's steps is neither a win nor a death.

    Test type: integration
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (6, 6))
    np.random.seed(0)
    history = run_episode(env, [int(SnakeAction.TURN_LEFT), int(SnakeAction.TURN_LEFT)], state)
    values = metrics_of(env, [history])
    assert values[SnakePOMDPMetrics.ENDED_BY_TIMEOUT.value] == 1.0
    assert values[SnakePOMDPMetrics.ENDED_BY_FAILURE.value] == 0.0
    assert values[SnakePOMDPMetrics.ENDED_BY_GOAL.value] == 0.0
    assert values[SnakePOMDPMetrics.AVERAGE_EPISODE_LENGTH.value] == 2.0


def test_the_starvation_severity_metric_is_the_worst_moment():
    """``max_steps_since_food`` is how close the episode came to starving.

    Purpose: The three death counts say whether starvation happened; this says
        how close the episodes that survived came to it, which is the only
        danger signal an episode that did not die reports at all.

    Test type: integration
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (6, 6), steps_since_food=2)
    np.random.seed(0)
    history = run_episode(env, [int(SnakeAction.GO_STRAIGHT)] * 3, state)
    values = metrics_of(env, [history])
    assert values[SnakePOMDPMetrics.MAX_STEPS_SINCE_FOOD.value] == pytest.approx(5.0)
