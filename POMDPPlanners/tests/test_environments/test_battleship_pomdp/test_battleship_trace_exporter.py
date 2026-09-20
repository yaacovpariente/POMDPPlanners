# SPDX-License-Identifier: MIT

"""Tests for the Battleship episode trace and for the visualizer's move.

The trace is the interface between Python and the browser viewer: a field that
does not survive the round trip is a field the viewer silently never sees, and
a field that carries the wrong thing is a page that lies convincingly. So the
payload is checked against the episode it was built from rather than against
itself.

The move of ``battleship_visualizer`` into the ``visualizer`` package is
checked here too. The GIF's bytes are pinned by a golden hash that only runs
inside the project's Docker image, so the risk a local suite can still catch is
the one the move actually created: the environment's own lazy import reaching a
different renderer from the package export, or reaching none at all.
"""

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.battleship_pomdp import (
    BattleshipBelief,
    BattleshipPOMDP,
    BattleshipVisualizer,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_visualization.trace_exporter import (
    BATTLESHIP_PAYLOAD_KIND,
    MARGINAL_SOURCE_EXACT,
    MARGINAL_SOURCE_PARTICLES,
    build_battleship_trace,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs


@pytest.fixture(name="env")
def _env() -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


def _episode(env: BattleshipPOMDP, seed: int = 21) -> List[StepData]:
    """Run a fixed probe sequence and record it, belief included.

    The belief attached to each step is the real
    :class:`BattleshipBelief`, not a stand-in: the exporter's exact marginal is
    the field most worth checking, and a mock belief would leave it untested.
    """
    np.random.seed(seed)
    belief = BattleshipBelief.from_environment(env, n_particles=24)
    state = belief.sample()
    history: List[StepData] = []
    for action in (0, 6, 12, 18, 24, 7):
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=env.step_info(state, action, next_state),
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
        state = next_state
    history.append(
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
    return history


def test_trace_round_trips_through_json(env: BattleshipPOMDP, tmp_path: Path):
    """A trace written and read back is the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A trace built from a recorded Battleship episode.
    When: It is written to disk and read back.
    Then: Every field matches, and the payload kind is the one the scene
        module registers itself under.
    """
    original = build_battleship_trace(env, _episode(env), episode_index=2, policy_name="PFT_DPW")
    restored = EpisodeTrace.read(original.write(tmp_path / "trace_2.json"))

    assert restored == original
    assert restored.payload_kind == BATTLESHIP_PAYLOAD_KIND
    assert restored.policy == "PFT_DPW"
    assert restored.episode_index == 2


def test_payload_world_block_comes_from_the_environment(env: BattleshipPOMDP):
    """The world a viewer builds is the world the episode ran on.

    Purpose: A viewer that read class defaults instead would draw the wrong
    board for any configured run, and nothing on the page would say so.

    Given: An environment with the suite's pinned configuration.
    When: A trace is built from one of its episodes.
    Then: Every field of the world block equals the environment's own value.
    """
    payload = build_battleship_trace(env, _episode(env), episode_index=0).payload
    world = payload["world"]

    assert world["board_size"] == env.board_size
    assert tuple(world["ship_lengths"]) == env.ship_lengths
    assert world["allow_adjacent_ships"] == env.allow_adjacent_ships
    assert world["num_cells"] == env.num_cells
    assert world["num_ship_cells"] == env.num_ship_cells
    assert world["hit_reward"] == pytest.approx(env.hit_reward)
    assert world["miss_penalty"] == pytest.approx(env.miss_penalty)
    assert world["num_layouts"] == env.layouts.num_layouts


def test_payload_records_the_board_and_what_had_been_probed(env: BattleshipPOMDP):
    """Occupancy and probe flags are the episode's, step by step.

    Purpose: The occupancy is the truth layer and the probe flags are the
    evidence layer; a viewer keeps them apart, and can only do that if the
    trace does.

    Given: A six-probe episode with a terminal bookkeeping step.
    When: A trace is built from it.
    Then: The occupancy matches the recorded state, one probe flag array is
        written per step, and the flags grow by exactly the probed cell.
    """
    history = _episode(env)
    payload = build_battleship_trace(env, history, episode_index=0).payload

    assert payload["occupancy"] == [int(v) for v in env.occupancy(history[0].state)]
    assert len(payload["probed"]) == len(history)
    assert payload["probed"][0] == [0] * env.num_cells
    for index, step in enumerate(history[:-1]):
        before = set(np.flatnonzero(payload["probed"][index]).tolist())
        after = set(np.flatnonzero(payload["probed"][index + 1]).tolist())
        assert after - before <= {int(step.action)}
        assert before <= after


def test_belief_is_serialised_by_core(env: BattleshipPOMDP):
    """The belief in the payload is core's, not this exporter's.

    Purpose: Every environment's belief is written by the same core function,
    so a viewer draws particles once rather than once per environment.

    Given: An episode whose steps carry a BattleshipBelief.
    When: A trace is built.
    Then: Each belief is the particle payload core produces, naming the
        concrete class it came from.
    """
    history = _episode(env)
    beliefs = build_battleship_trace(env, history, episode_index=0).payload["beliefs"]

    assert len(beliefs) == len(history)
    for entry in beliefs:
        assert entry["kind"] == "particles"
        assert entry["belief_class"] == "BattleshipBelief"
        assert len(entry["particles"]) == entry["num_written"]


def test_marginals_are_the_beliefs_own_exact_posterior(env: BattleshipPOMDP):
    """The per-cell posterior is the belief's, computed over every layout.

    Purpose: The viewer draws a column only where doubt remains, and treats a
    cell it shows as certain as something the belief deduced. A marginal
    estimated from a few hundred particles can read as certain on the luck of
    the draw, so the exact one is what gets written.

    Given: An episode whose steps carry a BattleshipBelief.
    When: A trace is built.
    Then: Each step's marginal equals ``occupancy_marginal`` for that step's
        belief, the support size equals the consistent-layout count, and the
        source says the posterior is exact.
    """
    history = _episode(env)
    payload = build_battleship_trace(env, history, episode_index=0).payload

    assert payload["marginal_source"] == MARGINAL_SOURCE_EXACT
    for index, step in enumerate(history):
        expected = step.belief.occupancy_marginal(env)
        assert payload["occupancy_marginals"][index] == pytest.approx(expected.tolist())
        assert payload["support_sizes"][index] == int(step.belief.consistent_indices(env).size)

    # The support can only shrink: a probe rules layouts out and never back in.
    supports = payload["support_sizes"]
    assert all(later <= earlier for earlier, later in zip(supports, supports[1:]))


def test_a_belief_without_an_exact_marginal_is_reported_as_missing(env: BattleshipPOMDP):
    """A generic particle belief yields nulls, not an invented marginal.

    Purpose: Battleship can be run with an ordinary particle filter. The
    exporter must say it has no exact posterior for those steps, so the viewer
    projects core's particles and labels what it is showing, rather than
    presenting a sample as the posterior.

    Given: An episode whose steps carry a plain WeightedParticleBelief.
    When: A trace is built.
    Then: Every marginal and support size is None, and the source says the
        viewer must fall back to the particles.
    """
    history = _episode(env)
    generic = [
        StepData(
            state=step.state,
            action=step.action,
            next_state=step.next_state,
            observation=step.observation,
            reward=step.reward,
            belief=WeightedParticleBelief(
                particles=list(step.belief.particles),
                log_weights=np.log(
                    np.ones(len(step.belief.particles)) / len(step.belief.particles)
                ),
            ),
            info=step.info,
        )
        for step in history
    ]
    payload = build_battleship_trace(env, generic, episode_index=0).payload

    assert payload["marginal_source"] == MARGINAL_SOURCE_PARTICLES
    assert all(entry is None for entry in payload["occupancy_marginals"])
    assert all(entry is None for entry in payload["support_sizes"])


def test_cache_trace_writes_a_readable_file(env: BattleshipPOMDP, tmp_path: Path):
    """The environment writes its trace where the reporting site looks.

    Purpose: ``cache_trace`` is the path a real run takes; a
    ``build_episode_trace`` that works only when called by hand would never
    produce a file.

    Given: A recorded episode and an output directory.
    When: cache_trace is called for episode 4.
    Then: ``trace_4.json`` is written and reads back as a Battleship trace.
    """
    written = env.cache_trace(history=_episode(env), output_dir=tmp_path, episode_index=4)

    assert written == tmp_path / "trace_4.json"
    assert EpisodeTrace.read(written).payload_kind == BATTLESHIP_PAYLOAD_KIND


def test_an_empty_episode_is_refused(env: BattleshipPOMDP):
    """There is no trace for an episode that did not happen."""
    with pytest.raises(ValueError):
        build_battleship_trace(env, [], episode_index=0)


def test_the_environment_and_the_package_render_the_same_gif(env: BattleshipPOMDP, tmp_path: Path):
    """Moving the visualizer left one renderer, reached the same way.

    Purpose: The GIF's bytes are pinned by a golden hash that runs only inside
    the project's Docker image. What a local suite can still catch is the risk
    the move created: the environment's lazy import and the package export
    drifting apart, or the GIF quietly becoming a single still.

    Given: A recorded episode.
    When: It is rendered through ``cache_visualization`` and through the
        exported ``BattleshipVisualizer``.
    Then: Both files are byte-identical and carry more than one frame.
    """
    history = _episode(env)

    env.cache_visualization(history=history, output_dir=tmp_path, episode_index=0)
    through_environment = tmp_path / "battleship_board_0.gif"

    through_package = tmp_path / "direct.gif"
    BattleshipVisualizer(env).create_visualization(history, through_package)

    assert hashlib.sha256(through_environment.read_bytes()).hexdigest() == (
        hashlib.sha256(through_package.read_bytes()).hexdigest()
    )

    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(through_environment) as image:
        assert image.n_frames > 1
