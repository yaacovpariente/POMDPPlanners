# SPDX-License-Identifier: MIT

"""What a Chicheck Invaders episode reports, and how its belief recovers.

The conformance suite checks that every declared channel is emitted and every
declared metric name is produced. What it cannot check is whether the numbers
*mean* what their names say, which is what this file does: each metric is
computed over hand-built episodes whose correct value is written out.
"""

from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation.history import History, StepData
from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_ROW,
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_SHIP_WIDTH,
    ChicheckInvadersAction,
    ChicheckInvadersMetrics,
    ChicheckInvadersPOMDP,
    ChicheckInvadersStepChannel,
    ObservationMode,
    create_chicheck_invaders_belief,
    create_chicheck_invaders_state,
    noiseless_preset,
)


def build_env(**overrides):
    """A small world for metric and belief checks."""
    settings = {
        "num_columns": 5,
        "num_rows": 4,
        "num_chickens": 2,
        "dive_probability": 0.0,
        "fire_cooldown": 0,
        "max_steps": 20,
        "discount_factor": 0.95,
    }
    settings.update(overrides)
    return ChicheckInvadersPOMDP(**settings)


def placeholder_belief(state) -> WeightedParticleBelief:
    """A one-particle belief, so ``StepData`` has the ``Belief`` it requires.

    The metric code reads the belief at most for summary statistics, never for
    correctness, so a point mass on the true state is enough here -- the belief
    itself is exercised in its own tests below.
    """
    # A non-zero log-weight: WeightedParticleBelief rejects an all-zero vector,
    # since a belief with no weight anywhere cannot be normalised.
    return WeightedParticleBelief(
        particles=[np.array(state, copy=True)], log_weights=np.array([1.0])
    )


def run_episode(env, actions, state) -> History:
    """Roll ``actions`` from ``state`` and package the result as a ``History``."""
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
                reward=float(reward),
                belief=placeholder_belief(state),
                info=env.step_info(state, action, next_state),
            )
        )
        state = next_state
    if env.is_terminal(state):
        terminated = True
    if terminated:
        steps.append(
            StepData(
                state=state,
                action=None,
                next_state=None,
                observation=None,
                reward=None,
                belief=placeholder_belief(state),
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


def metric(metrics, name: str) -> float:
    """Return the value of the metric called ``name``.

    Raises rather than returning ``None`` for a missing name: a metric that was
    silently dropped is exactly the failure these tests exist to catch, and an
    assertion against ``None`` reads as a value mismatch rather than an absence.

    Raises:
        AssertionError: If no metric of that name was produced.
    """
    for value in metrics:
        if value.name == name:
            return float(value.value)
    raise AssertionError(f"compute_metrics produced no metric called {name!r}")


def test_a_cleared_episode_reports_a_goal_ending_and_its_kills():
    """Clearing the flock is a completion, a goal ending, and two kills.

    Given: Two chickens stacked in the ship's column, so two shots clear them.
    When: The episode runs to its terminal state.
    Then: Completion is 1, ``ended_by_goal`` is 1, the other two endings are 0,
        and two kills are counted.

    Test type: integration
    """
    env = build_env(num_rows=6)
    column = env.ship_start_column
    state = create_chicheck_invaders_state(
        env, chickens=[[column, 4, 1, MODE_PATROL, 1], [column, 5, 1, MODE_PATROL, 1]]
    )
    metrics = env.compute_metrics([run_episode(env, [int(ChicheckInvadersAction.FIRE)] * 5, state)])
    assert metric(metrics, ChicheckInvadersMetrics.TASK_COMPLETION_RATE.value) == 1.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_GOAL.value) == 1.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_FAILURE.value) == 0.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_TIMEOUT.value) == 0.0
    assert metric(metrics, ChicheckInvadersMetrics.AVERAGE_CHICKENS_KILLED.value) == 2.0


def test_a_lost_episode_reports_a_failure_ending_and_a_hit_taken():
    """Being overrun is a failure ending and one hit taken, not a timeout.

    Purpose: A completion rate alone is ambiguous -- the difference between a
        planner taking bad risks and one given too small a budget is exactly
        this channel.

    Test type: integration
    """
    env = build_env()
    column = env.ship_start_column
    state = create_chicheck_invaders_state(
        env, chickens=[[column, 1, 1, MODE_DIVE, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    metrics = env.compute_metrics([run_episode(env, [int(ChicheckInvadersAction.STAY)] * 5, state)])
    assert metric(metrics, ChicheckInvadersMetrics.TASK_COMPLETION_RATE.value) == 0.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_FAILURE.value) == 1.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_GOAL.value) == 0.0
    assert metric(metrics, ChicheckInvadersMetrics.ENDED_BY_TIMEOUT.value) == 0.0
    assert metric(metrics, ChicheckInvadersMetrics.AVERAGE_HITS_TAKEN.value) == 1.0


def test_an_episode_that_runs_out_of_steps_reports_a_timeout():
    """Neither winning nor losing is a timeout, and the three endings sum to one.

    Test type: integration
    """
    env = build_env(max_steps=4)
    state = create_chicheck_invaders_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    metrics = env.compute_metrics(
        [run_episode(env, [int(ChicheckInvadersAction.STAY)] * 10, state)]
    )
    endings = [
        metric(metrics, ChicheckInvadersMetrics.ENDED_BY_GOAL.value),
        metric(metrics, ChicheckInvadersMetrics.ENDED_BY_FAILURE.value),
        metric(metrics, ChicheckInvadersMetrics.ENDED_BY_TIMEOUT.value),
    ]
    assert endings == [0.0, 0.0, 1.0]
    assert sum(endings) == pytest.approx(1.0)
    assert metric(metrics, ChicheckInvadersMetrics.AVERAGE_EPISODE_LENGTH.value) == 5.0


def test_shot_accuracy_is_kills_per_shot_over_the_episode():
    """Accuracy is the episode's ratio, not the mean of per-step ratios.

    Purpose: This is the one metric a channel reduction cannot express, which
        is why it lives in ``compute_metrics``. A mean of per-step ratios would
        read 0.5 here -- one step with ratio 1 and one with ratio 0 -- rather
        than the episode's true 0.5 only by coincidence, so the check uses an
        episode where the two differ.

    Given: An episode that fires repeatedly at a column holding one chicken:
        the first shot connects, the rest hit nothing.
    When: The metrics are computed.
    Then: ``shot_accuracy`` is the episode's kills-over-shots ratio, it is
        strictly below one, and it agrees with the two counts.

    Test type: integration
    """
    env = build_env(num_rows=6, num_chickens=2, max_steps=20)
    column = env.ship_start_column
    state = create_chicheck_invaders_state(
        env, chickens=[[column, 5, 1, MODE_PATROL, 1], [0, 1, 1, MODE_PATROL, 1]]
    )
    history = run_episode(env, [int(ChicheckInvadersAction.FIRE)] * 6, state)
    metrics = env.compute_metrics([history])
    shots = metric(metrics, ChicheckInvadersMetrics.AVERAGE_SHOTS_FIRED.value)
    kills = metric(metrics, ChicheckInvadersMetrics.AVERAGE_CHICKENS_KILLED.value)
    accuracy = metric(metrics, ChicheckInvadersMetrics.SHOT_ACCURACY.value)
    assert shots > kills > 0
    assert accuracy == pytest.approx(kills / shots)
    assert accuracy < 1.0


def test_shot_accuracy_is_reported_even_when_no_episode_fired():
    """A batch in which nothing was fired still reports the name it declares.

    Purpose: A declared metric name that ``compute_metrics`` sometimes omits is
        silently dropped by every consumer that looks it up.

    Test type: integration
    """
    env = build_env()
    state = create_chicheck_invaders_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]]
    )
    metrics = env.compute_metrics([run_episode(env, [int(ChicheckInvadersAction.STAY)] * 3, state)])
    assert metric(metrics, ChicheckInvadersMetrics.SHOT_ACCURACY.value) == 0.0
    assert set(env.get_metric_names()) <= {value.name for value in metrics}


def test_encroachment_is_the_grid_span_less_the_nearest_chicken_distance():
    """The severity channel is recoverable as a distance, and is zero when clear.

    Test type: unit
    """
    env = build_env()
    span = env.max_chicken_distance
    close = create_chicheck_invaders_state(
        env, chickens=[[2, 1, 1, MODE_PATROL, 1], [4, 3, -1, MODE_PATROL, 1]], ship_column=2
    )
    info = env.step_info(close, None, None)
    assert info[ChicheckInvadersStepChannel.CHICKEN_ENCROACHMENT_CELLS.value] == span - 1.0

    cleared = create_chicheck_invaders_state(
        env, chickens=[[2, 1, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]]
    )
    cleared_info = env.step_info(cleared, None, None)
    assert cleared_info[ChicheckInvadersStepChannel.CHICKEN_ENCROACHMENT_CELLS.value] == 0.0


def test_step_info_tolerates_the_terminal_bookkeeping_step():
    """The final recorded step reports the state's outcome and neutral transitions.

    Test type: unit
    """
    env = build_env()
    state = create_chicheck_invaders_state(
        env, chickens=[[0, 3, 1, MODE_PATROL, 0], [4, 3, -1, MODE_PATROL, 0]]
    )
    info = env.step_info(state, None, None)
    assert info[ChicheckInvadersStepChannel.FLOCK_CLEARED.value] == 1.0
    assert info[ChicheckInvadersStepChannel.CHICKENS_KILLED.value] == 0.0
    assert info[ChicheckInvadersStepChannel.SHOT_FIRED.value] == 0.0


def test_the_belief_concentrates_on_the_true_flock_under_exact_sensors():
    """With perfect sensors the filter's weight collapses onto the truth.

    Purpose: This is the end-to-end check that the likelihood, the transition
        and the filter agree. If any of the three disagreed, the weight on the
        true flock would not rise, and it is the one failure that makes every
        planning result on this environment meaningless.

    Given: The noiseless preset, a world small enough that 400 particles cover
        it, and a ship that holds position so the whole flock stays in reach.
    When: Five observations are filtered.
    Then: Most of the belief's weight sits on particles whose live chickens
        occupy exactly the true cells.

    Test type: integration
    """
    env = ChicheckInvadersPOMDP(
        num_columns=4,
        num_rows=3,
        num_chickens=1,
        dive_probability=0.2,
        max_steps=20,
        discount_factor=0.95,
        **noiseless_preset(),
    )
    np.random.seed(4)
    belief = create_chicheck_invaders_belief(env, n_particles=400)
    state = env.initial_state_dist().sample()[0]
    for _ in range(5):
        next_state, observation, _ = env.sample_next_step(state, int(ChicheckInvadersAction.STAY))
        belief = belief.update(
            action=int(ChicheckInvadersAction.STAY), observation=observation, pomdp=env
        )
        state = next_state

    truth = {
        (float(slot[CHICKEN_COLUMN]), float(slot[CHICKEN_ROW]))
        for slot in env.chickens(state)
        if slot[CHICKEN_ALIVE] > 0.0
    }
    weights = np.asarray(belief.normalized_weights, dtype=np.float64).ravel()
    on_truth = 0.0
    for particle, weight in zip(belief.particles, weights):
        cells = {
            (float(slot[CHICKEN_COLUMN]), float(slot[CHICKEN_ROW]))
            for slot in env.chickens(np.asarray(particle, dtype=np.float64))
            if slot[CHICKEN_ALIVE] > 0.0
        }
        if cells == truth:
            on_truth += float(weight)
    assert on_truth > 0.6, f"only {on_truth:.2f} of the belief's weight is on the true flock"


def test_reinvigoration_leaves_a_reported_chicken_alone():
    """A chicken the sensors just located is not jittered.

    Purpose: Perturbing a chicken the camera and radar both pinned would throw
        away the only hard information the step produced -- the refresh exists
        for the chickens nothing reported.

    Given: A belief whose particles all agree, and a reading that reports the
        single chicken slot.
    When: Reinvigoration runs with every particle eligible.
    Then: No particle's chicken moved.

    Test type: unit
    """
    env = build_env(num_chickens=1)
    state = create_chicheck_invaders_state(env, chickens=[[2, 2, 1, MODE_PATROL, 1]])
    belief = create_chicheck_invaders_belief(env, n_particles=8, reinvigoration_fraction=1.0)
    belief.particles = [np.array(state, copy=True) for _ in range(8)]

    reported = np.zeros(env.observation_size, dtype=np.float64)
    reported[0] = float(env.ship_column(state))
    reported[OBSERVATION_SHIP_WIDTH + 0] = 1.0
    reported[OBSERVATION_SHIP_WIDTH + 2] = 1.0

    np.random.seed(0)
    refreshed = belief.reinvigorate(
        action=int(ChicheckInvadersAction.STAY), observation=reported, pomdp=env, belief=belief
    )
    for particle in refreshed.particles:
        slot = env.chickens(np.asarray(particle, dtype=np.float64))[0]
        assert (slot[CHICKEN_COLUMN], slot[CHICKEN_ROW]) == (2.0, 2.0)


def test_reinvigoration_moves_an_unreported_chicken():
    """A chicken nothing reported is re-drawn, which is what buys diversity back.

    Test type: unit
    """
    env = build_env(num_chickens=1)
    state = create_chicheck_invaders_state(env, chickens=[[2, 2, 1, MODE_PATROL, 1]])
    belief = create_chicheck_invaders_belief(env, n_particles=40, reinvigoration_fraction=1.0)
    belief.particles = [np.array(state, copy=True) for _ in range(40)]
    silent = np.zeros(env.observation_size, dtype=np.float64)
    silent[0] = float(env.ship_column(state))

    np.random.seed(1)
    refreshed = belief.reinvigorate(
        action=int(ChicheckInvadersAction.STAY), observation=silent, pomdp=env, belief=belief
    )
    cells = {
        (
            float(env.chickens(np.asarray(p, dtype=np.float64))[0][CHICKEN_COLUMN]),
            float(env.chickens(np.asarray(p, dtype=np.float64))[0][CHICKEN_ROW]),
        )
        for p in refreshed.particles
    }
    assert len(cells) > 1
    # Never onto the ship's own row: a particle there asserts the episode has
    # already been lost, which no jitter should be able to claim.
    assert all(row >= 1.0 for _, row in cells)


def test_the_belief_panel_renders_occupancy_probability_not_expected_count():
    """A cell two chickens share is painted once, not twice.

    Purpose: The panel's docstring promises the weighted chance that a cell
        holds a chicken. Summing one weight per live chicken renders the
        expected chicken *count* instead, and a ``clip`` to 1 hides the
        overflow rather than fixing it -- a single certain particle with two
        chickens stacked would paint that cell exactly as a certain particle
        with one. Sideways patrols walk into each other, so stacked cells are
        reachable rather than theoretical.

    Given: One particle, certain, with two of its three chickens on one cell
        and the third elsewhere.
    When: The marginal is computed.
    Then: Both occupied cells read 1.0, and the grid sums to 2.0 -- the number
        of occupied cells -- rather than 3.0, the number of chickens.

    Test type: unit
    """
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.chicheck_invaders_pomdp import ChicheckInvadersVisualizer

    env = build_env(num_chickens=3)
    state = create_chicheck_invaders_state(
        env,
        chickens=[
            [1, 2, 1, MODE_PATROL, 1],
            [1, 2, -1, MODE_PATROL, 1],
            [3, 1, 1, MODE_PATROL, 1],
        ],
    )
    belief = placeholder_belief(state)
    # pylint: disable-next=protected-access
    grid = ChicheckInvadersVisualizer(env)._occupancy_marginal(belief)

    assert grid[2, 1] == pytest.approx(1.0)
    assert grid[1, 3] == pytest.approx(1.0)
    assert float(grid.sum()) == pytest.approx(2.0)


def test_a_fully_observable_belief_collapses_onto_the_observed_state():
    """In FULL mode the belief is a point mass on what was observed.

    Purpose: The likelihood is one only on an exact array match, so every
        particle drawn from the flock prior floors, and
        ``WeightedParticleBelief`` normalises an all-floor vector to a *uniform*
        one. That is indistinguishable from a healthy prior, so the registered
        ``ChicheckInvadersPOMDP[fully_observable]`` baseline would have run on a
        belief that was uniform over particles all known to be wrong.

    Given: The fully observable environment and one filtered step.
    When: The belief is updated with the observation.
    Then: Every particle equals the true state.

    Test type: integration
    """
    env = build_env(num_chickens=2, observation_mode=ObservationMode.FULL)
    np.random.seed(2)
    belief = create_chicheck_invaders_belief(env, n_particles=25)
    state = env.initial_state_dist().sample()[0]

    next_state, observation, _ = env.sample_next_step(state, int(ChicheckInvadersAction.STAY))
    belief = belief.update(
        action=int(ChicheckInvadersAction.STAY), observation=observation, pomdp=env
    )
    assert all(
        np.array_equal(np.asarray(p, dtype=np.float64), next_state) for p in belief.particles
    ), "a fully observable belief must be a point mass on the observed state"


def test_a_belief_every_particle_contradicts_is_rebuilt_from_the_reading():
    """When no particle can explain the reading, the particles are replaced.

    Purpose: Resampling cannot recover here -- it draws from the particles that
        are already wrong -- and the normalise-to-uniform path makes the
        collapse invisible. Under the noiseless preset one contradicted slot is
        enough to reach it.

    Given: The noiseless preset, a reading taken from the true state, and a
        belief whose particles all put the flock somewhere that reading rules
        out.
    When: The belief is updated.
    Then: Every particle can explain the reading afterwards.

    Test type: integration
    """
    env = ChicheckInvadersPOMDP(
        num_columns=5,
        num_rows=4,
        num_chickens=1,
        dive_probability=0.0,
        max_steps=20,
        discount_factor=0.95,
        **noiseless_preset(),
    )
    truth = create_chicheck_invaders_state(env, chickens=[[2, 1, 1, MODE_PATROL, 1]], ship_column=2)
    observation = env.sample_observation(truth, None)

    # Every particle puts the chicken where the reading says it is not.
    wrong = create_chicheck_invaders_state(env, chickens=[[4, 3, 1, MODE_PATROL, 1]], ship_column=2)
    np.random.seed(0)
    belief = create_chicheck_invaders_belief(env, n_particles=30)
    belief.particles = [np.array(wrong, copy=True) for _ in range(30)]
    assert all(
        env.observation_log_probability_single(p, None, observation) <= -1e17
        for p in belief.particles
    ), "fixture is wrong: the particles must all contradict the reading"

    refreshed = belief.update(
        action=int(ChicheckInvadersAction.STAY), observation=observation, pomdp=env
    )
    scores = [
        env.observation_log_probability_single(p, None, observation) for p in refreshed.particles
    ]
    assert all(score > -1e17 for score in scores), (
        "after a total collapse every particle should be able to explain the reading, "
        f"but {sum(score <= -1e17 for score in scores)} of {len(scores)} still cannot"
    )
