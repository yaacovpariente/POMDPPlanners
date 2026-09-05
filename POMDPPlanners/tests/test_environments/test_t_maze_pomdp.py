# SPDX-License-Identifier: MIT

"""Tests for the T-Maze POMDP.

The shared Environment contracts (hashing, batch/single agreement, serialization
round trip, reward range, seed determinism, step_info purity) are covered once for
every registered environment by ``test_env_api_conformance.py``; ``TMazePOMDP`` is
in its ``ENV_BUILDERS``. What is left, and what this file covers, is what is
specific to this environment: the corridor's geometry, the absorbing endpoints,
wall collisions, and above all the single-use cue — its frequency, its
normalisation, and the fact that nothing can make it speak twice.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import (
    ACTIONS,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    OBSERVATIONS,
    STATE_CUE_PHASE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
    TMazeMetric,
    TMazePOMDP,
    TMazeStepChannel,
    create_t_maze_state,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import t_maze_pinned_kwargs


def _env(**overrides) -> TMazePOMDP:
    return TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs(**overrides))


def _walk(env: TMazePOMDP, state, actions):
    """Apply a sequence of actions, returning the final state."""
    for action in actions:
        state = env.sample_next_state(state, action)
    return state


class TestGeometry:
    """The T-shaped corridor and the constructor's validation of it."""

    def test_cell_set_is_a_symmetric_t(self):
        """The valid cells are the stem plus both arms and nothing else.

        Purpose: The geometry is what every other rule is stated against — the cue
            cell, the junction and the endpoints are all positions in it — so an
            off-by-one here would quietly change the task rather than break it

        Given: A default T-Maze
        When: Its cell set and landmark cells are read
        Then: The stem runs from the start to the junction, both arms are present
            at equal length, and the landmarks sit where the model says

        Test type: unit
        """
        env = _env()
        expected = {(0, y) for y in range(env.stem_length + 1)} | {
            (x, env.stem_length) for x in range(-env.arm_length, env.arm_length + 1)
        }
        assert set(env.valid_cells) == expected
        assert env.start_cell == (0, 0)
        assert env.cue_cell == (0, 1)
        assert env.junction == (0, env.stem_length)
        assert env.left_endpoint == (-env.arm_length, env.stem_length)
        assert env.right_endpoint == (env.arm_length, env.stem_length)

    def test_cue_sits_strictly_below_the_junction(self):
        """There is at least one identical-observation step between cue and choice.

        Purpose: If the cue cell were the junction the task would collapse from a
            memory problem into a reactive one, because the agent would still be
            hearing the cue at the moment it has to choose

        Given: The default maze
        When: The cue row and the junction row are compared
        Then: The cue is strictly lower, leaving corridor steps in between

        Test type: unit
        """
        env = _env()
        assert env.cue_cell[1] < env.junction[1]
        assert env.junction[1] - env.cue_cell[1] >= 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stem_length": 1},
            {"stem_length": 0},
            {"arm_length": 0},
            {"cue_accuracy": 0.4},
            {"cue_accuracy": 1.5},
        ],
    )
    def test_degenerate_configurations_are_rejected(self, kwargs):
        """Degenerate sizes and out-of-range accuracies raise rather than run.

        Purpose: A maze with no corridor, or with an anti-correlated cue, is a
            different task wearing this one's name; accepting it silently would
            make two configurations mean one environment

        Given: A constructor argument outside the model's stated range
        When: The environment is constructed
        Then: ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError):
            _env(**kwargs)

    def test_deterministic_cue_accuracy_is_allowed(self):
        """``cue_accuracy=1.0`` constructs and is a pure-memory task.

        Purpose: The noiseless setting is the one that isolates memory from noise
            handling, so it has to be reachable rather than excluded by the range
            check

        Given: A maze with a perfectly accurate cue
        When: The cue is observed from an emitting state
        Then: It always names the true side

        Test type: unit
        """
        env = _env(cue_accuracy=1.0)
        emitting = create_t_maze_state(env.cue_cell, GOAL_RIGHT, CUE_EMITTING)
        np.random.seed(0)
        assert {env.sample_observation(emitting, "up") for _ in range(50)} == {
            OBSERVATION_RIGHT_CUE
        }


class TestTransitions:
    """Movement, wall collisions and the absorbing endpoints."""

    def test_moves_follow_the_corridor(self):
        """Each action moves one cell when the target is inside the corridor.

        Given: The default maze and an agent on the stem
        When: ``up`` is applied repeatedly
        Then: The agent walks the stem to the junction, one row per action

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.start_cell, GOAL_LEFT)
        for expected_row in range(1, env.stem_length + 1):
            state = env.sample_next_state(state, "up")
            assert (int(state[STATE_X]), int(state[STATE_Y])) == (0, expected_row)

    def test_wall_collisions_leave_the_position_unchanged(self):
        """A move into a wall stays put with probability 1.

        Purpose: The model says an invalid move is a no-op rather than an error or
            a random deflection, and the collision metric counts exactly these

        Given: An agent on the stem, where left, right and down are all walls
        When: Each blocked action is applied
        Then: The position is unchanged and the transition density puts all its
            mass on the unchanged state

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state((0, 2), GOAL_LEFT, CUE_CONSUMED)
        for action in ("left", "right"):
            next_state = env.sample_next_state(state, action)
            assert (int(next_state[STATE_X]), int(next_state[STATE_Y])) == (0, 2)
            log_prob = env.transition_log_probability(state, action, [next_state])
            assert float(log_prob[0]) == pytest.approx(0.0)

    def test_endpoints_are_terminal_and_absorbing(self):
        """Both endpoints end the episode and cannot be left.

        Purpose: An endpoint that is terminal but not absorbing would let an
            over-long episode walk back out and be paid a second time

        Given: An agent standing on each endpoint
        When: Every action is applied
        Then: ``is_terminal`` holds and the state is unchanged by every action

        Test type: unit
        """
        env = _env()
        for endpoint in (env.left_endpoint, env.right_endpoint):
            state = create_t_maze_state(endpoint, GOAL_LEFT, CUE_CONSUMED)
            assert env.is_terminal(state)
            for action in ACTIONS:
                assert np.array_equal(env.sample_next_state(state, action), state)

    def test_goal_side_never_changes(self):
        """No transition rewrites the hidden goal side.

        Given: Both goal sides and a long random action sequence
        When: The states are rolled forward
        Then: The goal slot is unchanged throughout

        Test type: unit
        """
        env = _env()
        rng = np.random.default_rng(0)
        for goal_side in (GOAL_LEFT, GOAL_RIGHT):
            state = create_t_maze_state(env.start_cell, goal_side)
            for _ in range(30):
                state = env.sample_next_state(state, ACTIONS[int(rng.integers(len(ACTIONS)))])
                assert float(state[STATE_GOAL]) == goal_side


class TestReward:
    """Payouts at the endpoints, the step cost, and paying only once."""

    def test_correct_endpoint_pays_the_goal_reward(self):
        """Entering the correct endpoint pays ``+goal_reward``, replacing the step cost.

        Given: An agent at the junction with the goal on the left
        When: It moves left onto the endpoint
        Then: The reward is exactly the goal reward, not the goal reward less the
            step penalty

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.junction, GOAL_LEFT, CUE_CONSUMED)
        assert env.reward(state, "left") == pytest.approx(env.goal_reward)

    def test_wrong_endpoint_pays_the_penalty(self):
        """Entering the wrong endpoint pays ``-wrong_goal_penalty``.

        Given: An agent at the junction with the goal on the left
        When: It moves right onto the other endpoint
        Then: The reward is the wrong-endpoint penalty

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.junction, GOAL_LEFT, CUE_CONSUMED)
        assert env.reward(state, "right") == pytest.approx(-env.wrong_goal_penalty)

    def test_every_other_action_costs_one_step_including_collisions(self):
        """Ordinary moves and wall bumps both cost ``-step_penalty``.

        Given: An agent on the stem
        When: A legal move and a blocked move are each scored
        Then: Both pay the step penalty, so a collision is not free and not extra

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state((0, 2), GOAL_LEFT, CUE_CONSUMED)
        assert env.reward(state, "up") == pytest.approx(-env.step_penalty)
        assert env.reward(state, "left") == pytest.approx(-env.step_penalty)

    def test_terminal_states_are_paid_only_once(self):
        """Actions taken from an endpoint pay nothing.

        Purpose: Without this, an episode that runs past its terminal state
            accumulates the goal reward once per remaining step, and the return of
            a successful episode would depend on the step budget

        Given: An agent standing on the correct endpoint
        When: Every action is scored from there
        Then: Each pays 0.0

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.left_endpoint, GOAL_LEFT, CUE_CONSUMED)
        for action in ACTIONS:
            assert env.reward(state, action) == pytest.approx(0.0)

    def test_declared_reward_range_covers_every_payout(self):
        """The declared bounds contain each of the four rewards the model defines.

        Purpose: A too-narrow reward range is the most repeated bug in this
            repository, and downstream CVaR code treats the range as a hard bound

        Given: The default maze
        When: The goal reward, the wrong-endpoint penalty, the step penalty and the
            terminal 0.0 are compared against ``reward_range``
        Then: All four lie inside

        Test type: unit
        """
        env = _env()
        assert env.reward_range is not None
        low, high = env.reward_range
        for value in (env.goal_reward, -env.wrong_goal_penalty, -env.step_penalty, 0.0):
            assert low <= value <= high


class TestCue:
    """The single-use cue: when it speaks, how often it is right, and that it stops."""

    def test_cue_is_emitted_on_entering_the_cue_cell(self):
        """Entering the cue cell sets the phase to emitting, and nothing else does.

        Given: An agent at the start cell
        When: It moves up onto the cue cell, then up again
        Then: The phase runs unseen -> emitting -> consumed

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.start_cell, GOAL_LEFT)
        assert float(state[STATE_CUE_PHASE]) == CUE_UNSEEN
        state = env.sample_next_state(state, "up")
        assert float(state[STATE_CUE_PHASE]) == CUE_EMITTING
        state = env.sample_next_state(state, "up")
        assert float(state[STATE_CUE_PHASE]) == CUE_CONSUMED

    def test_no_cue_before_the_cue_cell_or_after_it(self):
        """Every non-emitting state observes ``"empty"`` with probability 1.

        Given: The start cell, a corridor cell after the cue, and an endpoint
        When: Observations are drawn and scored
        Then: Only ``"empty"`` has non-zero probability

        Test type: unit
        """
        env = _env()
        for state in (
            create_t_maze_state(env.start_cell, GOAL_LEFT, CUE_UNSEEN),
            create_t_maze_state((0, 3), GOAL_LEFT, CUE_CONSUMED),
            create_t_maze_state(env.left_endpoint, GOAL_LEFT, CUE_CONSUMED),
        ):
            np.random.seed(1)
            assert {env.sample_observation(state, "up") for _ in range(25)} == {OBSERVATION_EMPTY}
            log_probs = env.observation_log_probability(state, "up", list(OBSERVATIONS))
            assert float(np.exp(log_probs[OBSERVATIONS.index(OBSERVATION_EMPTY)])) == pytest.approx(
                1.0
            )

    def test_initial_observation_carries_no_cue(self):
        """The initial observation distribution is ``"empty"`` with probability 1.

        Purpose: The cue is earned by entering the cue cell; handing it out before
            the first action would give the answer away for free

        Given: The default maze
        When: The initial observation distribution is sampled
        Then: Only ``"empty"`` is ever produced

        Test type: unit
        """
        env = _env()
        np.random.seed(2)
        assert set(env.initial_observation_dist().sample(20)) == {OBSERVATION_EMPTY}

    @pytest.mark.parametrize("goal_side", [GOAL_LEFT, GOAL_RIGHT])
    def test_cue_observation_distribution_is_normalized(self, goal_side):
        """``Z(o | s')`` sums to 1 on the emitting state and names the side correctly.

        Given: An emitting state for each goal side
        When: The likelihood of every observation is evaluated
        Then: The matching cue carries ``cue_accuracy``, the opposite carries
            ``1 - cue_accuracy``, ``"empty"`` carries 0, and the three sum to 1

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.cue_cell, goal_side, CUE_EMITTING)
        probs = np.exp(env.observation_log_probability(state, "up", list(OBSERVATIONS)))
        matching = OBSERVATION_LEFT_CUE if goal_side == GOAL_LEFT else OBSERVATION_RIGHT_CUE
        opposite = OBSERVATION_RIGHT_CUE if goal_side == GOAL_LEFT else OBSERVATION_LEFT_CUE
        assert float(probs[OBSERVATIONS.index(matching)]) == pytest.approx(env.cue_accuracy)
        assert float(probs[OBSERVATIONS.index(opposite)]) == pytest.approx(1 - env.cue_accuracy)
        assert float(probs[OBSERVATIONS.index(OBSERVATION_EMPTY)]) == pytest.approx(0.0)
        assert float(np.sum(probs)) == pytest.approx(1.0)

    def test_cue_is_right_at_about_its_stated_accuracy(self):
        """Sampling the cue reproduces ``cue_accuracy`` empirically.

        Purpose: The density and the sampler are written separately, so a test of
            the density alone would not catch a sampler that ignores the accuracy

        Given: An emitting state with the goal on the left
        When: 4000 observations are drawn
        Then: The fraction naming the left side is within sampling error of 0.9

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.cue_cell, GOAL_LEFT, CUE_EMITTING)
        np.random.seed(3)
        draws = [env.sample_observation(state, "up") for _ in range(4000)]
        share_left = draws.count(OBSERVATION_LEFT_CUE) / len(draws)
        assert share_left == pytest.approx(env.cue_accuracy, abs=0.03)
        assert draws.count(OBSERVATION_EMPTY) == 0

    @pytest.mark.parametrize("second_action", list(ACTIONS))
    def test_cue_is_consumed_by_whatever_action_follows_it(self, second_action):
        """Any action consumes an emitting cue, wall bumps included.

        Purpose: This is the rule that makes the cue single-use. If a blocked move
            left the phase emitting, an agent could stand on the cue cell bumping
            into the wall and average the noise away, which would delete the task

        Given: An agent that has just entered the cue cell
        When: Each of the four actions is applied — three of which are walls there
        Then: The phase is consumed in every case and the next observation is
            ``"empty"``

        Test type: unit
        """
        env = _env()
        state = env.sample_next_state(create_t_maze_state(env.start_cell, GOAL_LEFT), "up")
        assert float(state[STATE_CUE_PHASE]) == CUE_EMITTING
        after = env.sample_next_state(state, second_action)
        assert float(after[STATE_CUE_PHASE]) == CUE_CONSUMED
        np.random.seed(4)
        assert {env.sample_observation(after, second_action) for _ in range(20)} == {
            OBSERVATION_EMPTY
        }

    def test_revisiting_the_cue_cell_reveals_nothing(self):
        """A second visit to the cue cell does not re-arm the cue.

        Given: An agent that reads the cue, walks up the stem and comes back down
        When: It re-enters the cue cell
        Then: The phase stays consumed and the observation is ``"empty"``

        Test type: unit
        """
        env = _env()
        state = _walk(
            env,
            create_t_maze_state(env.start_cell, GOAL_LEFT),
            ["up", "up", "up", "down", "down"],
        )
        assert (int(state[STATE_X]), int(state[STATE_Y])) == env.cue_cell
        assert float(state[STATE_CUE_PHASE]) == CUE_CONSUMED
        np.random.seed(5)
        assert {env.sample_observation(state, "down") for _ in range(20)} == {OBSERVATION_EMPTY}


class TestBeliefMemory:
    """The cue moves the belief, and the corridor leaves it where the cue put it."""

    def test_left_cue_moves_a_uniform_prior_to_the_stated_posterior(self):
        """One left cue takes a 0.5 / 0.5 prior to 0.9 / 0.1.

        Purpose: This is the number the environment promises. It is what a planner
            has to act on, and it falls straight out of the observation likelihood

        Given: A uniform prior over the goal side, at the start cell
        When: The belief is updated with ``up`` and a ``"left_cue"``
        Then: The mass on the left goal is 0.9

        Test type: integration
        """
        env = _env()
        belief = WeightedParticleBelief(
            particles=[
                create_t_maze_state(env.start_cell, GOAL_LEFT),
                create_t_maze_state(env.start_cell, GOAL_RIGHT),
            ],
            log_weights=np.log(np.array([0.5, 0.5])),
            resampling=False,
        )
        updated = belief.update(action="up", observation=OBSERVATION_LEFT_CUE, pomdp=env)
        assert self._left_mass(updated) == pytest.approx(env.cue_accuracy, abs=1e-9)

    def test_empty_corridor_observations_preserve_the_posterior(self):
        """Walking the corridor after the cue does not move the belief.

        Purpose: The whole task is carrying one reading across steps that say
            nothing. If ``"empty"`` reweighted the particles the memory would decay
            for reasons the model does not describe

        Given: The 0.9 / 0.1 posterior produced by a left cue
        When: The belief is updated with ``up`` and ``"empty"`` for the rest of the
            stem
        Then: The mass on the left goal is unchanged

        Test type: integration
        """
        env = _env()
        belief = WeightedParticleBelief(
            particles=[
                create_t_maze_state(env.start_cell, GOAL_LEFT),
                create_t_maze_state(env.start_cell, GOAL_RIGHT),
            ],
            log_weights=np.log(np.array([0.5, 0.5])),
            resampling=False,
        )
        belief = belief.update(action="up", observation=OBSERVATION_LEFT_CUE, pomdp=env)
        for _ in range(env.stem_length - 1):
            belief = belief.update(action="up", observation=OBSERVATION_EMPTY, pomdp=env)
        assert self._left_mass(belief) == pytest.approx(env.cue_accuracy, abs=1e-9)
        assert all(
            (int(p[STATE_X]), int(p[STATE_Y])) == env.junction for p in belief.particles
        )

    def test_deterministic_cue_collapses_the_belief(self):
        """At ``cue_accuracy=1.0`` one cue settles the goal side outright.

        Given: A noiseless maze and a uniform prior
        When: The belief is updated with a right cue
        Then: All the mass sits on the right goal

        Test type: integration
        """
        env = _env(cue_accuracy=1.0)
        belief = WeightedParticleBelief(
            particles=[
                create_t_maze_state(env.start_cell, GOAL_LEFT),
                create_t_maze_state(env.start_cell, GOAL_RIGHT),
            ],
            log_weights=np.log(np.array([0.5, 0.5])),
            resampling=False,
        )
        belief = belief.update(action="up", observation=OBSERVATION_RIGHT_CUE, pomdp=env)
        assert self._left_mass(belief) == pytest.approx(0.0, abs=1e-9)

    @staticmethod
    def _left_mass(belief: WeightedParticleBelief) -> float:
        weights = np.asarray(belief.normalized_weights, dtype=np.float64)
        weights = weights / float(np.sum(weights))
        return float(
            np.sum(
                [
                    weights[index]
                    for index, particle in enumerate(belief.particles)
                    if float(np.asarray(particle)[STATE_GOAL]) == GOAL_LEFT
                ]
            )
        )


class TestMetrics:
    """The per-step channels the completion and episode-end metrics are built from."""

    def test_entering_the_correct_endpoint_reports_completion(self):
        """The completion and goal channels fire on the step that reaches the goal.

        Given: An agent at the junction with the goal on the left
        When: It moves onto the left endpoint and ``step_info`` is read
        Then: The correct-endpoint and ended-by-goal channels are 1 and the wrong
            and timeout channels are 0

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.junction, GOAL_LEFT, CUE_CONSUMED)
        next_state = env.sample_next_state(state, "left")
        info = env.step_info(state, "left", next_state)
        assert info[TMazeStepChannel.CORRECT_ENDPOINT.value] == 1.0
        assert info[TMazeStepChannel.ENDED_BY_GOAL.value] == 1.0
        assert info[TMazeStepChannel.WRONG_ENDPOINT.value] == 0.0
        assert info[TMazeStepChannel.ENDED_BY_TIMEOUT.value] == 0.0

    def test_entering_the_wrong_endpoint_reports_failure_not_completion(self):
        """A wrong choice is reported as a failure, distinct from a timeout.

        Purpose: A completion rate alone cannot separate a planner that guessed
            from one that never got to the junction, and the two need opposite fixes

        Given: An agent at the junction with the goal on the left
        When: It moves onto the right endpoint
        Then: The failure channels fire and completion does not

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.junction, GOAL_LEFT, CUE_CONSUMED)
        next_state = env.sample_next_state(state, "right")
        info = env.step_info(state, "right", next_state)
        assert info[TMazeStepChannel.WRONG_ENDPOINT.value] == 1.0
        assert info[TMazeStepChannel.ENDED_BY_FAILURE.value] == 1.0
        assert info[TMazeStepChannel.CORRECT_ENDPOINT.value] == 0.0
        assert info[TMazeStepChannel.ENDED_BY_TIMEOUT.value] == 0.0

    def test_corridor_steps_report_a_timeout_until_something_ends(self):
        """A step that ends nothing reports the timeout channel.

        Given: An ordinary corridor step
        When: ``step_info`` is read
        Then: Only the timeout end-channel is set, so a ``LAST`` reduction over an
            episode that never terminated reports a timeout

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state((0, 2), GOAL_LEFT, CUE_CONSUMED)
        info = env.step_info(state, "up", env.sample_next_state(state, "up"))
        assert info[TMazeStepChannel.ENDED_BY_TIMEOUT.value] == 1.0
        assert info[TMazeStepChannel.ENDED_BY_GOAL.value] == 0.0
        assert info[TMazeStepChannel.ENDED_BY_FAILURE.value] == 0.0

    def test_wall_collisions_are_counted(self):
        """A blocked move reports a collision; a legal move does not.

        Given: An agent on the stem
        When: A blocked and a legal action are each measured
        Then: Only the blocked one sets the collision channel

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state((0, 2), GOAL_LEFT, CUE_CONSUMED)
        blocked = env.step_info(state, "left", env.sample_next_state(state, "left"))
        legal = env.step_info(state, "up", env.sample_next_state(state, "up"))
        assert blocked[TMazeStepChannel.WALL_COLLISION.value] == 1.0
        assert legal[TMazeStepChannel.WALL_COLLISION.value] == 0.0

    def test_terminal_bookkeeping_step_reports_the_ending(self):
        """``step_info(state, None, None)`` still names how the episode ended.

        Purpose: The episode runner records the final state only on this call, and
            the end-reason metrics reduce with ``LAST``, so a neutral answer here
            would report every successful episode as a timeout

        Given: The final state of a successful episode
        When: ``step_info`` is called with no action and no successor
        Then: The goal channel is set, the collision channel is neutral, and every
            value is a plain finite scalar

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.left_endpoint, GOAL_LEFT, CUE_CONSUMED)
        info = env.step_info(state, None, None)
        assert info[TMazeStepChannel.ENDED_BY_GOAL.value] == 1.0
        assert info[TMazeStepChannel.ENDED_BY_TIMEOUT.value] == 0.0
        assert info[TMazeStepChannel.WALL_COLLISION.value] == 0.0
        assert all(isinstance(value, float) and np.isfinite(value) for value in info.values())

    def test_declared_metric_names_match_the_specs(self):
        """Every declared metric maps to a channel the environment emits.

        Given: The default maze and one measured transition
        When: The declared specs are compared against the emitted channels
        Then: Each spec's channel is emitted and each metric name is declared

        Test type: unit
        """
        env = _env()
        state = create_t_maze_state(env.start_cell, GOAL_LEFT)
        emitted = set(env.step_info(state, "up", env.sample_next_state(state, "up")))
        specs = env.get_metric_specs()
        assert {spec.channel for spec in specs} <= emitted
        assert {spec.name for spec in specs} == {metric.value for metric in TMazeMetric}
        assert TMazeMetric.TASK_COMPLETION_RATE.value in env.get_metric_names()


class TestSerialization:
    """Round trip and configuration identity for this environment specifically."""

    def test_round_trip_rebuilds_an_equal_environment(self):
        """``from_dict(to_dict(env))`` returns an equal env with the same identity.

        Given: A non-default T-Maze
        When: It is serialized and rebuilt
        Then: The rebuild is equal, shares its ``config_id``, and keeps its geometry

        Test type: unit
        """
        env = _env(stem_length=6, arm_length=2, cue_accuracy=1.0)
        rebuilt = TMazePOMDP.from_dict(env.to_dict())
        assert rebuilt == env
        assert rebuilt.config_id == env.config_id
        assert rebuilt.junction == env.junction
        assert set(rebuilt.valid_cells) == set(env.valid_cells)

    def test_config_id_separates_different_mazes(self):
        """Two mazes that differ in any configured value get different identities.

        Purpose: ``config_id`` is the experiment cache key; two different tasks
            sharing one id would serve each other's cached episodes

        Given: The default maze and one with a different cue accuracy
        When: Their ``config_id`` values are compared
        Then: They differ

        Test type: unit
        """
        assert _env().config_id != _env(cue_accuracy=0.7).config_id
        assert _env().config_id != _env(stem_length=5).config_id

    def test_config_id_survives_use(self):
        """Using the environment does not change its identity.

        Given: A maze and its ``config_id``
        When: It is rolled forward and both batch paths are exercised
        Then: The id is unchanged, so nothing is being memoized into it

        Test type: unit
        """
        env = _env()
        before = env.config_id
        states = env.initial_state_dist().sample(4)
        env.reward_batch(states, "up")
        env.sample_next_state_batch(states, "up")
        _walk(env, states[0], ["up"] * 5)
        assert env.config_id == before
