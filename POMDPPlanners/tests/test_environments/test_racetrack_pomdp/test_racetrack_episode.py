# SPDX-License-Identifier: MIT

"""End-to-end episode tests for the racetrack world/model/belief triple.

These are the tests that exercise the arrangement the environment actually exists for:
the forward-only world advancing the true state, a separate planner-side model on
``policy.environment`` doing the predicting, and a belief filtering between them. Every
other test file checks one piece in isolation; this one checks that they compose.

The policy here is a fixed action cycle rather than a real planner. That is deliberate —
a search planner would make these tests slow and would turn a planner regression into a
failure of the environment suite.
"""

# pylint: disable=protected-access  # One test inserts traffic into the live session.

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (
    RacetrackMetric,
    RacetrackPOMDP,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    EGO_STATE_WIDTH,
    ObservationMode,
    state_agent_rows,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import geometry_from_world
from POMDPPlanners.simulations.episodes import run_episode

pytest.importorskip("highway_env")

_NUM_PARTICLES = 24
_NUM_STEPS = 12
_MAX_TRACKED_AGENTS = 4
# One grid cell plus the jitter: a stamped blob sits at its cell centre, so it can be up
# to 1.5 m from the vehicle that produced it before any pose jitter is added.
_POSITION_MATCH_TOLERANCE_M = 6.0
# The tracker can only be measured when an opponent is inside the ego's +/-18 m grid
# window, and on this track that is rare: the lap is ~350 m of centreline, the ego covers
# about 2 m per decision, and the racetrack spawns its traffic anywhere on the lap. Across
# seeds 0-13 the nearest opponent sits 35-53 m away for entire episodes. Rather than hunt
# for a lucky seed -- which couples the measurement to both the seed and the action
# sequence, and silently stops measuring anything the moment either changes -- this test
# places an opponent itself.
_OPPONENT_GAP_M = 10.0
_OPPONENT_SPEED_MPS = 4.0


def _preset(acceleration: float, steering: float) -> int:
    """Index of a control preset, looked up rather than written as a literal.

    The shipped table is a 3-by-9 grid of accelerations by steering angles, so every index
    moves whenever the steering resolution changes. Naming the command keeps these tests
    testing what they say they test.
    """
    return DEFAULT_ACTION_PRESETS.index((acceleration, steering))


_COAST_ONLY = _preset(0.0, 0.0)
# Coast, accelerate, coast, brake -- all straight ahead, so the ego stays on the lane long
# enough for the episode to record something.
_STRAIGHT_CYCLE = [_COAST_ONLY, _preset(1.0, 0.0), _COAST_ONLY, _preset(-1.0, 0.0)]


class _FixedCyclePolicy(Policy):
    """Policy that cycles through a fixed action list, ignoring the belief entirely."""

    def __init__(self, environment: Any, discount_factor: float, actions: List[int]) -> None:
        self._actions = actions
        self._index = 0
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name="FixedCyclePolicy",
        )

    def action(self, belief: Any) -> Tuple[List[Any], PolicyRunData]:
        del belief
        chosen = self._actions[self._index % len(self._actions)]
        self._index += 1
        return [chosen], PolicyRunData(info_variables=[])

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return []


def _seed_particles(model: RacetrackModelPOMDP, world_state: np.ndarray) -> np.ndarray:
    """Seed particles around the world's true start, jittered on the ego block."""
    width = EGO_STATE_WIDTH + _MAX_TRACKED_AGENTS * AGENT_SLOT_WIDTH
    del model
    rng = np.random.default_rng(0)
    particles = np.tile(np.asarray(world_state, dtype=float), (_NUM_PARTICLES, 1))
    particles[:, :EGO_STATE_WIDTH] += rng.normal(0.0, 0.05, size=(_NUM_PARTICLES, EGO_STATE_WIDTH))
    assert particles.shape == (_NUM_PARTICLES, width)
    return particles


def _mean_agent_rows(belief: Any) -> np.ndarray:
    """The belief's stamped agent block, averaged over particles, one row per slot."""
    particles = np.asarray(belief.particles, dtype=float)
    return (
        particles[:, EGO_STATE_WIDTH:].mean(axis=0).reshape(_MAX_TRACKED_AGENTS, AGENT_SLOT_WIDTH)
    )


def _pair_by_position(stamped: np.ndarray, truth: np.ndarray) -> List[float]:
    """Velocity errors for stamped slots that sit on top of a truly occupied one.

    Slot index cannot be trusted for the pairing: the world fills its slots with the
    nearest vehicles at any range, while the belief only ever sees the ones inside the
    grid window, so the two orderings diverge as soon as a vehicle sits outside it.
    Position is what both agree on, to within the grid's own 3 m quantisation.
    """
    errors: List[float] = []
    occupied = truth[truth[:, AGENT_PRESENT] > 0.5]
    if occupied.size == 0:
        return errors
    for row in stamped[stamped[:, AGENT_PRESENT] > 0.5]:
        offsets = occupied[:, AGENT_REL_X : AGENT_REL_X + 2] - row[AGENT_REL_X : AGENT_REL_X + 2]
        nearest = int(np.argmin(np.linalg.norm(offsets, axis=1)))
        if float(np.linalg.norm(offsets[nearest])) > _POSITION_MATCH_TOLERANCE_M:
            continue
        difference = occupied[nearest, AGENT_REL_VX : AGENT_REL_VX + 2] - (
            row[AGENT_REL_VX : AGENT_REL_VX + 2]
        )
        errors.append(float(np.linalg.norm(difference)))
    return errors


def _velocity_errors(history: Any) -> List[float]:
    """Every slot-step where a stamped opponent can be lined up against the true one.

    Step ``i``'s observation is only stamped onto the belief *recorded* at step ``i + 1``:
    the episode loop records the step before it updates the belief, so pairing a step's
    own belief with its own successor would compare a stamp against the frame after the
    one that produced it.
    """
    errors: List[float] = []
    steps = history.history
    for step, following in zip(steps, steps[1:]):
        if step.next_state is None or following.belief is None:
            continue
        truth = state_agent_rows(np.asarray(step.next_state, dtype=float), _MAX_TRACKED_AGENTS)
        errors.extend(_pair_by_position(_mean_agent_rows(following.belief), truth))
    return errors


def _place_opponent_ahead(world: RacetrackPOMDP, gap_m: float, speed_mps: float) -> np.ndarray:
    """Put one slower vehicle directly ahead of the ego, inside the grid window.

    Reaches into the live simulator on purpose. The alternative is to pick a seed on which
    the racetrack happens to spawn traffic nearby, which couples this measurement to both
    the seed and the action sequence and stops measuring anything the moment either
    changes -- exactly the silent failure this test exists to prevent.

    Returns:
        The world's live state after the insertion, to seed the belief from.
    """
    from highway_env.vehicle.kinematics import (  # pylint: disable=import-outside-toplevel
        Vehicle,
    )

    session = world._get_session()
    unwrapped = session._env.unwrapped
    ego = unwrapped.vehicle
    heading = np.array([np.cos(ego.heading), np.sin(ego.heading)], dtype=float)
    unwrapped.road.vehicles = [v for v in unwrapped.road.vehicles if v is ego]
    unwrapped.road.vehicles.append(
        Vehicle(
            unwrapped.road,
            ego.position + gap_m * heading,
            heading=float(ego.heading),
            speed=speed_mps,
        )
    )
    state = np.asarray(session._read_state(), dtype=float)
    world._live_state = state
    return state


def _drive_and_compare(
    world: RacetrackPOMDP,
    model: RacetrackModelPOMDP,
    belief: Belief,
    state: np.ndarray,
) -> List[float]:
    """Coast forward, filtering as we go, collecting stamped-vs-true velocity errors.

    The loop is written out rather than delegated to ``run_episode`` because the opponent
    has to be inserted after the world resets, and the runner owns that reset.
    """
    errors: List[float] = []
    for _ in range(_NUM_STEPS):
        next_state = world.sample_next_state(state, _COAST_ONLY)
        observation = world.sample_observation(next_state, _COAST_ONLY)
        belief = belief.update(
            action=_COAST_ONLY,
            observation=model.encode_observation(observation),
            pomdp=model,
            state=next_state,
        )
        truth = state_agent_rows(next_state, _MAX_TRACKED_AGENTS)
        errors.extend(_pair_by_position(_mean_agent_rows(belief), truth))
        state = next_state
        if world.is_terminal(state):
            break
    return errors


def _build_triple(
    mode: ObservationMode, seed: int = 0, actions: Optional[List[int]] = None
) -> Tuple[RacetrackPOMDP, RacetrackModelPOMDP, _FixedCyclePolicy]:
    world = RacetrackPOMDP(
        discount_factor=0.95,
        observation_mode=mode,
        max_tracked_agents=_MAX_TRACKED_AGENTS,
        seed=seed,
    )
    # Reset before the map is walked, and walk it from the world's own starting lane. The
    # world writes arclength as an offset from that lane, so a map built from any other
    # starting point would put its corners in the wrong place relative to the state the
    # belief is seeded with -- consistent-looking numbers, silently misaligned track.
    world.initial_state_dist().sample()
    track_geometry, _ = geometry_from_world(world)
    model = KnownTrackModel(
        discount_factor=0.95,
        track_geometry=track_geometry,
        observation_mode=mode,
        max_tracked_agents=_MAX_TRACKED_AGENTS,
    )
    policy = _FixedCyclePolicy(
        environment=model,
        discount_factor=0.95,
        actions=list(actions) if actions is not None else list(_STRAIGHT_CYCLE),
    )
    return world, model, policy


@pytest.fixture(autouse=True)
def _headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pygame from touching a display in CI."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


class TestEpisodeRunsEndToEnd:
    """The Definition-of-Done check: a full episode through EpisodeRunner."""

    @pytest.mark.parametrize("mode", [ObservationMode.POMDP, ObservationMode.MDP])
    def test_episode_completes_and_reports_every_declared_metric(
        self, mode: ObservationMode
    ) -> None:
        """Test that a full episode runs and scores in both arms of the matched pair.

        Purpose: Validates the arrangement the environment exists for — a forward-only
            world advancing the truth, a separate model on policy.environment doing the
            predicting, and a belief between them — and that the result is scoreable

        Given: A racetrack world, a matching generative model, a particle belief seeded
            from the world's true start, and a fixed-cycle policy holding the model
        When: run_episode drives the three of them for a bounded number of steps
        Then: The episode completes, every recorded transition carries all six declared
            measurement channels, and compute_metrics returns exactly the declared metric
            names in their declared order

        Test type: integration
        """
        world, model, policy = _build_triple(mode)
        world_state = world.initial_state_dist().sample()[0]
        particles = _seed_particles(model, world_state)
        log_weights = np.full(_NUM_PARTICLES, -np.log(_NUM_PARTICLES))
        belief = TrackedAgentsBelief(
            particles=particles,
            log_weights=log_weights,
            observation_mode=mode,
            max_tracked_agents=_MAX_TRACKED_AGENTS,
        )

        history = run_episode(
            environment=world,
            policy=policy,
            initial_belief=belief,
            num_steps=_NUM_STEPS,
            logger=None,
        )

        assert history.actual_num_steps >= 1
        declared = {spec.channel for spec in world.get_metric_specs()}
        transitions = [step for step in history.history if step.action is not None]
        assert transitions, "the episode recorded no transition steps"
        for step in transitions:
            assert step.info is not None
            assert declared.issubset(step.info.keys())

        metrics = world.compute_metrics([history])
        assert [metric.name for metric in metrics] == world.get_metric_names()
        assert world.get_metric_names() == [member.value for member in RacetrackMetric]

    def test_the_same_seed_reproduces_the_same_rewards(self) -> None:
        """Test that a seeded episode is reproducible.

        Purpose: Validates that the world's seeding reaches the simulator, without which
            an MDP-versus-POMDP comparison could not hold anything else fixed

        Given: Two identically seeded worlds and identical fixed-cycle policies
        When: Each runs an episode of the same length
        Then: The two reward sequences are identical

        Test type: integration
        """
        rewards = []
        for _ in range(2):
            world, model, policy = _build_triple(ObservationMode.POMDP)
            world_state = world.initial_state_dist().sample()[0]
            belief = WeightedParticleBelief(
                particles=list(_seed_particles(model, world_state)),
                log_weights=np.full(_NUM_PARTICLES, -np.log(_NUM_PARTICLES)),
            )
            history = run_episode(
                environment=world,
                policy=policy,
                initial_belief=belief,
                num_steps=_NUM_STEPS,
                logger=None,
            )
            rewards.append([step.reward for step in history.history if step.action is not None])
        assert rewards[0] == rewards[1]


class TestGridVelocityIsMeasuredNotAssumed:
    """Quantify what the occupancy grid actually tells the belief about velocity."""

    def test_tracker_velocity_error_against_ground_truth_is_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test and report the tracker's relative-velocity error versus the truth.

        Purpose: Measures, rather than assumes, how much the occupancy grid reveals about
            opponent motion. A vehicle marks exactly one 3 m cell and a decision step is
            0.2 s, so the smallest detectable shift reads as 15 m/s — above the track's
            own speed limit. This is the main reason the partially-observed arm should
            underperform, so the number belongs in the record instead of a comment

        Given: A live POMDP world stepped forward with a fixed action, on the seed that
            actually brings the opponent inside the +/-18 m window. That is not the default
            seed and the choice is not cosmetic: on seed 0 the opponent stays 35 m away for
            the whole episode, so the tracker is never exercised and a comparison written
            against that seed would pass while measuring nothing
        When: The belief's stamped agent velocities are compared against the world state's
            true relative velocities for slots that are genuinely occupied
        Then: The comparison runs on at least one such step and the observed error is
            asserted only to be finite, with the measured magnitude printed for the
            write-up rather than pinned to a threshold that would encode today's accident
            as a requirement

        Test type: integration
        """
        world, model, _ = _build_triple(ObservationMode.POMDP)
        world_state = world.initial_state_dist().sample()[0]
        world_state = _place_opponent_ahead(world, _OPPONENT_GAP_M, _OPPONENT_SPEED_MPS)
        belief = TrackedAgentsBelief(
            particles=_seed_particles(model, world_state),
            log_weights=np.full(_NUM_PARTICLES, -np.log(_NUM_PARTICLES)),
            observation_mode=ObservationMode.POMDP,
            max_tracked_agents=_MAX_TRACKED_AGENTS,
        )
        errors = _drive_and_compare(world, model, belief, world_state)

        assert errors, "no step carried both a stamped slot and a true occupied slot"
        assert np.all(np.isfinite(errors))
        with capsys.disabled():
            print(
                f"\ntracker relative-velocity error over {len(errors)} slot-steps: "
                f"mean {float(np.mean(errors)):.2f} m/s, max {float(np.max(errors)):.2f} m/s"
            )
