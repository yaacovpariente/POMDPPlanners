# SPDX-License-Identifier: MIT

"""Tests for the LaserTag episode trace exporter and the visualizer move.

Two things are worth testing here and nothing else is.

The first is that the exported beams are the observation model's beams. In
this environment the eight ranges *are* the observation, and the package
already carries two implementations of them: the model the agent is given, and
the ray walk the discrete GIF draws, which ignores the opponent. A viewer fed
re-derived ranges would be a third. These tests pin the exported ranges to the
environment's own functions and, specifically, to the fact that a beam stops
at the opponent.

The second is that moving the renderers into ``visualizer/`` changed nothing
about the GIFs. The byte-level pin lives in the golden-file suite, which only
runs inside the project's Docker image; what is checked here is what can be
checked anywhere — that the renderers still resolve from the new package, that
the sprite sheet travels with them, and that a render is reproducible.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    LASER_DIRECTIONS,
    compute_laser_measurements,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    _LASER_DIRECTIONS,
    LaserTagPOMDP,
)
from POMDPPlanners.environments.laser_tag_pomdp.visualizer import (
    ContinuousLaserTagVisualizer,
    LaserTagVisualizer,
)
from POMDPPlanners.environments.laser_tag_pomdp.visualizer.trace_exporter import (
    LASER_TAG_PAYLOAD_KIND,
    build_continuous_laser_tag_trace,
    build_laser_tag_trace,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_laser_tag_pinned_kwargs,
    laser_tag_pinned_kwargs,
)


def _belief(states: List[Any]) -> WeightedParticleBelief:
    particles = [np.asarray(s, dtype=float) for s in states]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _step(state, action, next_state, observation, reward=-1.0) -> StepData:
    # Two particles, not one: a single-particle cloud has log weight zero,
    # which WeightedParticleBelief refuses as a degenerate belief.
    return StepData(
        state=np.asarray(state, dtype=float),
        action=action,
        next_state=None if next_state is None else np.asarray(next_state, dtype=float),
        observation=observation,
        reward=reward,
        belief=_belief([state, state]),
    )


@pytest.fixture(name="discrete_env")
def discrete_env_fixture() -> LaserTagPOMDP:
    """A discrete LaserTag environment on the pinned default arena."""
    return LaserTagPOMDP(discount_factor=0.95, **laser_tag_pinned_kwargs())


@pytest.fixture(name="continuous_env")
def continuous_env_fixture() -> ContinuousLaserTagPOMDP:
    """A continuous LaserTag environment on the pinned default arena."""
    return ContinuousLaserTagPOMDP(discount_factor=0.95, **continuous_laser_tag_pinned_kwargs())


def _rollout(env: Any, actions: List[Any], seed: int = 42) -> List[StepData]:
    """Run a fixed action sequence and record it the way the episode loop does."""
    np.random.seed(seed)
    state = env.initial_state_dist().sample()[0]
    history: List[StepData] = []
    for action in actions:
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=_belief([state, next_state]),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break
    return history


def test_discrete_ranges_are_the_environments_own(discrete_env):
    """Test that exported discrete ranges come from the environment's model.

    Purpose: The eight ranges are the whole observation, so a viewer must not
        be handed a second, drifting copy of them. Recomputing them here from
        the environment's own cell walk is what makes the payload pinned to
        the model rather than to whatever the exporter felt like writing.

    Given: A recorded discrete episode
    When: Its trace is built
    Then: Every exported range equals ``_laser_distance_inline`` at that state

    Test type: unit
    """
    history = _rollout(discrete_env, [0, 1, 2, 3, 0, 1])
    payload = build_laser_tag_trace(discrete_env, history, 0).payload

    for index, step in enumerate(history):
        robot = (int(step.state[0]), int(step.state[1]))
        opponent = (int(step.state[2]), int(step.state[3]))
        expected = [
            # pylint: disable-next=protected-access
            discrete_env._laser_distance_inline(robot, direction, opponent)
            for direction in _LASER_DIRECTIONS
        ]
        assert payload["laser_ranges"][index] == pytest.approx(expected)


def test_continuous_ranges_are_the_environments_own(continuous_env):
    """Test that exported continuous ranges come from the environment's model.

    Purpose: Same reason as the discrete case — the ray/AABB/circle cast lives
        in one place and the payload must quote it, not re-implement it.

    Given: A recorded continuous episode
    When: Its trace is built
    Then: Every exported range equals ``compute_laser_measurements``

    Test type: unit
    """
    actions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])]
    history = _rollout(continuous_env, actions)
    payload = build_continuous_laser_tag_trace(continuous_env, history, 0).payload

    for index, step in enumerate(history):
        expected = compute_laser_measurements(
            np.asarray(step.state[:2], dtype=float),
            np.asarray(step.state[2:4], dtype=float),
            continuous_env.opponent_radius,
            np.asarray(continuous_env.walls, dtype=float).reshape(-1, 4),
            np.asarray(continuous_env.grid_size, dtype=float).reshape(-1),
        )
        assert payload["laser_ranges"][index] == pytest.approx(list(expected))


def test_discrete_beam_stops_at_the_opponent(discrete_env):
    """Test that a discrete beam ends at the opponent, not behind it.

    Purpose: The exported range must follow the observation model, because
        that is what the agent is given and what the viewer claims to be
        drawing. This test used to assert the opposite of its last clause: the
        GIF renderer discarded the opponent and walked on to the next wall
        while ``sample_observation`` stopped at it, and the test pinned that
        divergence. The renderer was fixed to stop at the opponent too, so the
        assertion is now that the two AGREE — which is the property actually
        worth protecting.

    Given: A state with the opponent in clear line of sight due east, with
        free cells behind it
    When: The trace is built
    Then: The east range stops one cell short of the opponent, is flagged as
        an opponent hit, and equals the length the GIF renderer draws

    Test type: unit
    """
    # Row 0 is free from column 0 to column 6, so a ray east from (0, 0) meets
    # the opponent at (0, 3) and would otherwise run to the arena edge.
    state = [0.0, 0.0, 0.0, 3.0, 0.0]
    history = [_step(state, 2, state, tuple([1.0] * 8))]
    payload = build_laser_tag_trace(discrete_env, history, 0).payload

    east = _LASER_DIRECTIONS.index((0, 1))
    assert payload["laser_ranges"][0][east] == pytest.approx(2.0)
    assert payload["hit_opponent"][0][east] is True

    visualizer = LaserTagVisualizer(
        floor_shape=discrete_env.floor_shape,
        walls=discrete_env.walls,
        dangerous_areas=list(discrete_env.dangerous_areas),
        dangerous_area_radius=discrete_env.dangerous_area_radius,
    )
    # pylint: disable-next=protected-access
    drawn = visualizer._laser_segments(np.asarray(state[:2]), np.asarray(state[2:4]))
    drawn_east = float(np.linalg.norm(drawn[east][1] - drawn[east][0]))
    assert drawn_east == pytest.approx(payload["laser_ranges"][0][east]), (
        "The drawn beam and the measured range must end in the same place: "
        "the renderer stops at the opponent now, as the observation model "
        "always did"
    )


def test_continuous_beam_stops_at_the_opponent(continuous_env):
    """Test that a continuous beam ends on the opponent's circle.

    Purpose: The continuous range model already stops at the opponent disc.
        Pinning it here keeps the two variants' payloads meaning the same
        thing, so the viewer can draw both the same way.

    Given: A state with the opponent straight along one ray, in the open
    When: The trace is built
    Then: That ray's range is the gap to the opponent's circle and is flagged

    Test type: unit
    """
    # Beam 0 is +y in the continuous frame; place the opponent along it.
    beam = 0
    direction = LASER_DIRECTIONS[beam]
    # x = 2.0 lies in no wall's x span, so nothing but the opponent is on the
    # ray before the far boundary five units away.
    robot = np.array([2.0, 2.0])
    opponent = robot + direction * 2.0
    state = [robot[0], robot[1], opponent[0], opponent[1], 0.0]
    history = [_step(state, np.array([0.0, 1.0, 0.0]), state, np.ones(8))]
    payload = build_continuous_laser_tag_trace(continuous_env, history, 0).payload

    assert payload["laser_ranges"][0][beam] == pytest.approx(
        2.0 - continuous_env.opponent_radius, abs=1e-6
    )
    assert payload["hit_opponent"][0][beam] is True


def test_a_clear_beam_is_not_flagged_as_an_opponent_hit(discrete_env):
    """Test that ``hit_opponent`` is false for a beam the opponent does not stop.

    Purpose: The flag drives the colour that says "the agent has just seen the
        opponent", which is the single most informative event in an episode.
        A flag that were true everywhere would be worse than no flag.

    Given: A state with the opponent off every ray but one
    When: The trace is built
    Then: Only that ray is flagged

    Test type: unit
    """
    state = [0.0, 0.0, 0.0, 3.0, 0.0]
    history = [_step(state, 2, state, tuple([1.0] * 8))]
    payload = build_laser_tag_trace(discrete_env, history, 0).payload

    east = _LASER_DIRECTIONS.index((0, 1))
    flags = payload["hit_opponent"][0]
    assert flags[east] is True
    assert [i for i, f in enumerate(flags) if f] == [east]


def test_observed_ranges_are_paired_with_the_state_they_were_taken_at(discrete_env):
    """Test that a recorded reading sits beside the state it was measured at.

    Purpose: ``sample_observation`` is called on a step's ``next_state``, so
        the reading recorded on step ``i`` belongs to state ``i + 1``. Writing
        it against state ``i`` would put the agent's reading a whole step away
        from the beams it is compared with, and the two would look wrong by a
        move every time.

    Given: A recorded episode
    When: The trace is built
    Then: State 0 carries no reading, and state ``i`` carries step ``i - 1``'s

    Test type: unit
    """
    history = _rollout(discrete_env, [0, 1, 2, 3])
    observed = build_laser_tag_trace(discrete_env, history, 0).payload["observed_ranges"]

    assert observed[0] is None
    for index in range(1, len(history)):
        assert observed[index] == pytest.approx(list(history[index - 1].observation))


def test_a_terminal_state_reports_no_measurement(discrete_env):
    """Test that a terminal state gets no ranges rather than ranges of zero.

    Purpose: A terminal state emits the all ``-1`` sentinel, which is not a
        measurement. Exporting it as eight zeros would draw a robot boxed in on
        all eight sides — a picture of something that did not happen.

    Given: An episode whose last recorded state is terminal
    When: The trace is built
    Then: That state's ranges and hit flags are empty, and its reading is None

    Test type: unit
    """
    live = [0.0, 0.0, 0.0, 3.0, 0.0]
    dead = [0.0, 3.0, 0.0, 3.0, 1.0]
    history = [
        _step(live, 4, dead, tuple([-1.0] * 8), reward=10.0),
        _step(dead, None, None, None, reward=None),
    ]
    payload = build_laser_tag_trace(discrete_env, history, 0).payload

    assert payload["terminals"] == [False, True]
    assert payload["laser_ranges"][1] == []
    assert payload["hit_opponent"][1] == []
    # The sentinel is not a range, so it is not written as one.
    assert payload["observed_ranges"][1] is None


def test_the_world_block_comes_from_the_instance():
    """Test that the arena in the payload is the one the episode ran on.

    Purpose: A configured run may not be using the class defaults, and a
        viewer that drew the defaults would show a different arena from the
        one the numbers came from.

    Given: An environment with a non-default wall set
    When: The trace is built
    Then: The payload's walls and directions are that instance's

    Test type: unit
    """
    env = LaserTagPOMDP(
        discount_factor=0.9, **laser_tag_pinned_kwargs(walls={(2, 2)}, dangerous_areas={(4, 4)})
    )
    history = _rollout(env, [0, 1])
    world = build_laser_tag_trace(env, history, 0).payload["world"]

    assert world["variant"] == "discrete"
    assert world["walls"] == [[2, 2]]
    assert world["hazards"] == [[4.0, 4.0]]
    assert world["laser_directions"] == [list(d) for d in _LASER_DIRECTIONS]


def test_the_two_variants_keep_their_own_direction_tables(discrete_env, continuous_env):
    """Test that each variant exports its own table, unconverted.

    Purpose: The two tables index the same eight beams but are written in
        different frames, and the arenas they are used against are stored
        differently too. Converting one into the other in the exporter would
        silently rotate one variant's beams.

    Given: One trace of each variant
    When: Their world blocks are read
    Then: The discrete table is the integer cell steps and the continuous one
        is the unit vectors, each as its own source defines it

    Test type: unit
    """
    discrete = build_laser_tag_trace(discrete_env, _rollout(discrete_env, [0]), 0)
    continuous = build_continuous_laser_tag_trace(
        continuous_env, _rollout(continuous_env, [np.array([1.0, 0.0, 0.0])]), 0
    )

    assert discrete.payload["world"]["laser_directions"] == [list(d) for d in _LASER_DIRECTIONS]
    assert np.asarray(continuous.payload["world"]["laser_directions"]) == pytest.approx(
        LASER_DIRECTIONS
    )


def test_the_belief_is_serialized_by_core(discrete_env):
    """Test that the exporter writes no belief format of its own.

    Purpose: The belief is a core abstraction with a closed family of
        implementations. An environment that serialized it itself would be one
        more place to update when a belief class is added, and one more place
        to get it wrong.

    Given: An episode whose belief is a weighted particle belief
    When: The trace is built
    Then: The payload carries core's particle payload, with the full 5-vector
        states core wrote and not a projection the exporter chose

    Test type: unit
    """
    history = _rollout(discrete_env, [0, 1])
    belief = build_laser_tag_trace(discrete_env, history, 0).payload["beliefs"][0]

    assert belief["kind"] == "particles"
    assert belief["belief_class"] == "WeightedParticleBelief"
    assert all(len(p) == 5 for p in belief["particles"])


def test_the_trace_round_trips_through_json(discrete_env, tmp_path: Path):
    """Test that a written trace reads back as the same trace.

    Purpose: The payload is only useful if a browser can read it. Anything
        numpy left in it would serialize as text a reader cannot parse back
        into numbers.

    Given: A discrete episode's trace
    When: It is written and read back
    Then: The payload, envelope and summary fields all survive

    Test type: unit
    """
    history = _rollout(discrete_env, [0, 1, 2, 4])
    trace = build_laser_tag_trace(discrete_env, history, 3, policy_name="POMCPOW")
    written = trace.write(tmp_path / "trace.json")
    restored = EpisodeTrace.read(written)

    assert restored.payload_kind == LASER_TAG_PAYLOAD_KIND
    assert restored.episode_index == 3
    assert restored.policy == "POMCPOW"
    assert restored.payload == json.loads(json.dumps(trace.payload))
    assert restored.discounted_return == pytest.approx(trace.discounted_return)


def test_both_environments_write_a_trace_file(discrete_env, continuous_env, tmp_path: Path):
    """Test that ``cache_trace`` produces a file for either variant.

    Purpose: The exporter is only wired in if the environment's own hook calls
        it. A helper nobody calls would pass every test above and still leave
        every real run without a trace.

    Given: One recorded episode of each variant
    When: ``cache_trace`` is called
    Then: Each writes a ``trace_<index>.json`` carrying the LaserTag kind

    Test type: integration
    """
    discrete_path = discrete_env.cache_trace(
        history=_rollout(discrete_env, [0, 1]), output_dir=tmp_path, episode_index=0
    )
    continuous_path = continuous_env.cache_trace(
        history=_rollout(continuous_env, [np.array([1.0, 0.0, 0.0])]),
        output_dir=tmp_path,
        episode_index=1,
    )

    assert discrete_path == tmp_path / "trace_0.json"
    assert continuous_path == tmp_path / "trace_1.json"
    for path in (discrete_path, continuous_path):
        assert json.loads(path.read_text())["payload_kind"] == LASER_TAG_PAYLOAD_KIND


def test_an_empty_history_is_refused(discrete_env):
    """Test that an empty episode raises rather than writing an empty trace.

    Purpose: An empty trace would load in the viewer and draw an arena with
        nothing in it, which reads as "the planner did nothing" rather than as
        "there is no episode here".

    Given: An empty history
    When: A trace is built from it
    Then: ``ValueError`` is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty history"):
        build_laser_tag_trace(discrete_env, [], 0)


@pytest.mark.parametrize("variant", ["discrete", "continuous"])
def test_the_moved_renderers_still_produce_identical_gifs(variant, tmp_path: Path):
    """Test that the GIF renderers survived the move into ``visualizer/``.

    Purpose: The migration is supposed to change where these modules live and
        nothing else. The byte-level pin against the checked-in golden GIFs
        runs in the project's Docker image, where the font and PIL versions
        are fixed; what is checked here, and runs anywhere, is that the
        renderers and their sprite sheet still resolve from the new package
        and that a render is reproducible.

    Given: One deterministic episode per variant
    When: It is rendered twice through the moved visualizer
    Then: Both renders exist and are byte-for-byte identical

    Test type: integration
    """
    # Imported here: this module is the golden suite's neighbour, not its
    # dependency, and importing every environment it registers at module scope
    # would make this file's collection cost the whole package.
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.tests.test_environments import (
        test_environment_visualizations_golden_files as golden,
    )

    if variant == "discrete":
        env = LaserTagPOMDP(
            discount_factor=0.95, **laser_tag_pinned_kwargs(transition_error_prob=0.0)
        )
        history = golden.create_deterministic_laser_tag_episode(seed=42)
        render = LaserTagVisualizer(
            floor_shape=env.floor_shape,
            walls=env.walls,
            dangerous_areas=list(env.dangerous_areas),
            dangerous_area_radius=env.dangerous_area_radius,
        ).create_visualization
    else:
        env = ContinuousLaserTagPOMDP(
            discount_factor=0.95,
            **continuous_laser_tag_pinned_kwargs(
                robot_transition_cov_matrix=np.eye(2) * 0.01,
                opponent_transition_cov_matrix=np.eye(2) * 0.01,
            ),
        )
        history = golden.create_deterministic_continuous_laser_tag_episode(seed=42)
        render = ContinuousLaserTagVisualizer(
            grid_size=env.grid_size,
            walls=env.walls,
            robot_radius=env.robot_radius,
            opponent_radius=env.opponent_radius,
            dangerous_areas=env.dangerous_areas,
            dangerous_area_radius=env.dangerous_area_radius,
        ).create_visualization

    first, second = tmp_path / "a.gif", tmp_path / "b.gif"
    render(history, first)
    render(history, second)

    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(second.read_bytes()).hexdigest()
    assert first.stat().st_size > 0
