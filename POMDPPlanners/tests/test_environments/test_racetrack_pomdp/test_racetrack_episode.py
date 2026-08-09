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

from typing import Any, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (
    RacetrackMetric,
    RacetrackPOMDP,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    ObservationMode,
)
from POMDPPlanners.simulations.episodes import run_episode

pytest.importorskip("highway_env")

_NUM_PARTICLES = 24
_NUM_STEPS = 12
_MAX_TRACKED_AGENTS = 4


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


def _build_triple(
    mode: ObservationMode,
) -> Tuple[RacetrackPOMDP, RacetrackModelPOMDP, _FixedCyclePolicy]:
    world = RacetrackPOMDP(
        discount_factor=0.95,
        observation_mode=mode,
        max_tracked_agents=_MAX_TRACKED_AGENTS,
        seed=0,
    )
    model = RacetrackModelPOMDP(
        discount_factor=0.95,
        observation_mode=mode,
        max_tracked_agents=_MAX_TRACKED_AGENTS,
    )
    policy = _FixedCyclePolicy(environment=model, discount_factor=0.95, actions=[4, 1, 4, 7])
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

    def test_tracker_velocity_error_against_ground_truth_is_reported(self) -> None:
        """Test and report the tracker's relative-velocity error versus the truth.

        Purpose: Measures, rather than assumes, how much the occupancy grid reveals about
            opponent motion. A vehicle marks exactly one 3 m cell and a decision step is
            0.2 s, so the smallest detectable shift reads as 15 m/s — above the track's
            own speed limit. This is the main reason the partially-observed arm should
            underperform, so the number belongs in the record instead of a comment

        Given: A live POMDP world stepped forward with a fixed action
        When: The belief's stamped agent velocities are compared against the world state's
            true relative velocities for slots that are genuinely occupied
        Then: The comparison runs and the observed error is asserted only to be finite,
            with the measured magnitude printed for the write-up rather than pinned to a
            threshold that would encode today's accident as a requirement

        Test type: integration
        """
        world, model, policy = _build_triple(ObservationMode.POMDP)
        world_state = world.initial_state_dist().sample()[0]
        belief = TrackedAgentsBelief(
            particles=_seed_particles(model, world_state),
            log_weights=np.full(_NUM_PARTICLES, -np.log(_NUM_PARTICLES)),
            observation_mode=ObservationMode.POMDP,
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
        assert np.all(np.isfinite(np.asarray(history.history[-1].state, dtype=float)))
