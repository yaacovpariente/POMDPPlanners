# SPDX-License-Identifier: MIT

"""Tests for the Battleship metrics.

The conformance suite checks that every declared channel is emitted and every
declared name is produced. What it cannot check is whether the reductions were
chosen correctly, and a wrong reduction is invisible: it reports a plausible
number for the wrong quantity. These tests build the two episode shapes that
matter — one that sinks the fleet and one that runs out of steps — and assert
what each metric reads in both.
"""

from typing import Dict, List

import numpy as np
import pytest

from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.battleship_pomdp import (
    BattleshipPOMDP,
    BattleshipPOMDPMetrics,
    create_battleship_state,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs


@pytest.fixture(name="env")
def _env() -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


def _belief(state) -> WeightedParticleBelief:
    return WeightedParticleBelief(particles=[state, state], log_weights=np.array([0.0, -0.1]))


def _episode(env: BattleshipPOMDP, ship_cells, actions) -> History:
    """Run a fixed probe sequence, mirroring what EpisodeRunner records."""
    occupancy = np.zeros(env.num_cells)
    occupancy[list(ship_cells)] = 1.0
    state = create_battleship_state(occupancy)

    steps: List[StepData] = []
    terminated = False
    for action in actions:
        if env.is_terminal(state):
            terminated = True
            break
        next_state, observation, reward = env.sample_next_step(state, action)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=_belief(state),
                info=env.step_info(state, action, next_state),
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
                belief=_belief(state),
                info=env.step_info(state, None, None),
            )
        )
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


def _metrics(env: BattleshipPOMDP, histories: List[History]) -> Dict[str, float]:
    return {metric.name: metric.value for metric in env.compute_metrics(histories)}


SHIPS = (0, 1, 2, 10, 11, 20, 21)


class TestCompletedEpisode:
    """An episode that sinks the fleet."""

    def test_completion_and_end_reason(self, env: BattleshipPOMDP) -> None:
        """Purpose: a sunk fleet must be a goal, not a timeout.

        Given: an episode probing every ship cell and two water cells
        When: metrics are computed
        Then: completion is 1, ended_by_goal is 1, and the other two reasons are 0

        Test type: unit
        """
        history = _episode(env, SHIPS, list(SHIPS) + [5, 6])
        values = _metrics(env, [history])

        assert values[BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value] == 1.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_GOAL.value] == 1.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_TIMEOUT.value] == 0.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_FAILURE.value] == 0.0

    def test_probe_counts_and_length(self, env: BattleshipPOMDP) -> None:
        """Purpose: the counts must exclude the terminal bookkeeping step.

        Given: an episode of seven hits, one water probe and one repeat
        When: metrics are computed
        Then: the counts are exactly seven, one and one, and the length counts
              the eight probes plus the terminal step

        Test type: unit
        """
        actions = [SHIPS[0], SHIPS[0], 5] + list(SHIPS[1:])
        history = _episode(env, SHIPS, actions)
        values = _metrics(env, [history])

        assert values[BattleshipPOMDPMetrics.AVERAGE_UNIQUE_SHIP_CELL_HITS.value] == 7.0
        assert values[BattleshipPOMDPMetrics.AVERAGE_REPEAT_PROBES.value] == 1.0
        assert values[BattleshipPOMDPMetrics.AVERAGE_WATER_PROBES.value] == 1.0
        assert values[BattleshipPOMDPMetrics.AVERAGE_EPISODE_LENGTH.value] == float(
            len(actions) + 1
        )

    def test_hit_fraction_reaches_one(self, env: BattleshipPOMDP) -> None:
        """Purpose: the partial-progress metric must saturate on completion.

        Given: a completed episode
        When: metrics are computed
        Then: the maximum hit fraction is 1

        Test type: unit
        """
        values = _metrics(env, [_episode(env, SHIPS, list(SHIPS))])
        assert values[BattleshipPOMDPMetrics.MAX_FLEET_HIT_FRACTION.value] == 1.0


class TestTimedOutEpisode:
    """An episode that runs out of probes."""

    def test_completion_and_end_reason(self, env: BattleshipPOMDP) -> None:
        """Purpose: an unfinished episode must read as a timeout, not a failure.

        Given: an episode probing only three of the seven ship cells
        When: metrics are computed
        Then: completion is 0, ended_by_timeout is 1, ended_by_failure is 0

        Test type: unit
        """
        values = _metrics(env, [_episode(env, SHIPS, [0, 1, 2, 5, 6])])

        assert values[BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value] == 0.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_GOAL.value] == 0.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_TIMEOUT.value] == 1.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_FAILURE.value] == 0.0

    def test_partial_progress_is_reported(self, env: BattleshipPOMDP) -> None:
        """Purpose: a completion rate of zero alone hides how close the run got.

        Given: an episode that hit three of seven ship cells
        When: metrics are computed
        Then: the maximum hit fraction is 3/7

        Test type: unit
        """
        values = _metrics(env, [_episode(env, SHIPS, [0, 1, 2, 5, 6])])
        assert values[BattleshipPOMDPMetrics.MAX_FLEET_HIT_FRACTION.value] == pytest.approx(3 / 7)


class TestHorizonBoundary:
    """The episode whose last allowed probe sinks the fleet."""

    def test_completing_on_the_final_allowed_probe_counts_as_a_goal(
        self, env: BattleshipPOMDP
    ) -> None:
        """Purpose: the runner never records a terminal step for this episode.

        ``EpisodeRunner._should_continue`` checks its step budget before it
        checks terminality, so an episode that sinks the fleet with its last
        allowed probe stops without appending the terminal bookkeeping step.
        Scoring the board channels from the pre-transition state would then
        report that episode as an unfinished timeout, missing its last hit — the
        single worst place for the completion metric to be wrong, because it is
        exactly the boundary a horizon is chosen to sit near.

        Given: an episode whose final recorded probe is the seventh hit, with no
               terminal step appended
        When: metrics are computed
        Then: completion and ended_by_goal are 1, timeout is 0, and all seven
              hits are counted

        Test type: unit
        """
        occupancy = np.zeros(env.num_cells)
        occupancy[list(SHIPS)] = 1.0
        state = create_battleship_state(occupancy)

        steps: List[StepData] = []
        for action in SHIPS:
            next_state, observation, reward = env.sample_next_step(state, action)
            steps.append(
                StepData(
                    state=state,
                    action=action,
                    next_state=next_state,
                    observation=observation,
                    reward=reward,
                    belief=_belief(state),
                    info=env.step_info(state, action, next_state),
                )
            )
            state = next_state
        assert env.is_terminal(state)

        history = History(
            history=steps,
            discount_factor=env.discount_factor,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(steps),
            reach_terminal_state=False,
            policy_run_data=[],
        )
        values = _metrics(env, [history])

        assert values[BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value] == 1.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_GOAL.value] == 1.0
        assert values[BattleshipPOMDPMetrics.ENDED_BY_TIMEOUT.value] == 0.0
        assert values[BattleshipPOMDPMetrics.AVERAGE_UNIQUE_SHIP_CELL_HITS.value] == 7.0
        assert values[BattleshipPOMDPMetrics.MAX_FLEET_HIT_FRACTION.value] == 1.0


class TestAcrossEpisodes:
    """What the averages mean over a batch."""

    def test_end_reason_rates_sum_to_one(self, env: BattleshipPOMDP) -> None:
        """Purpose: the three reasons partition the episodes, or one is missing.

        Given: one completed and one timed-out episode
        When: metrics are computed
        Then: the three end-reason rates sum to 1 and completion is 0.5

        Test type: unit
        """
        histories = [
            _episode(env, SHIPS, list(SHIPS)),
            _episode(env, SHIPS, [0, 1, 2]),
        ]
        values = _metrics(env, histories)
        total = (
            values[BattleshipPOMDPMetrics.ENDED_BY_GOAL.value]
            + values[BattleshipPOMDPMetrics.ENDED_BY_FAILURE.value]
            + values[BattleshipPOMDPMetrics.ENDED_BY_TIMEOUT.value]
        )
        assert total == pytest.approx(1.0)
        assert values[BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value] == pytest.approx(0.5)

    def test_declared_names_match_produced_names(self, env: BattleshipPOMDP) -> None:
        """Purpose: a metric nobody can look up by name is a dropped metric.

        Given: one episode
        When: metrics are computed
        Then: the produced names are exactly the declared ones

        Test type: unit
        """
        produced = {metric.name for metric in env.compute_metrics([_episode(env, SHIPS, [0, 1])])}
        assert produced == set(env.get_metric_names())
        assert BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value in produced
