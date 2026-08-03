# SPDX-License-Identifier: MIT

"""Tests for the VOPP closed-loop episode runner.

The suite covers constructor validation, the recorded-trajectory bookkeeping,
terminal-state handling, the ground-truth world-hook overrides, and an
end-to-end integration run against the real Continuous Light-Dark vectorized
model.
"""

import pytest
import torch
from torch import Tensor

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
    ContinuousLightDarkVectorizedModel,
)
from POMDPPlanners.planners.vectorized_planners import VOPPPlanner
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
    aggregate_step_info_metrics,
)
from POMDPPlanners.planners.vectorized_planners.vopp.vopp_episode_runner import (
    VOPPEpisodeResult,
    VOPPEpisodeRunner,
)


class DriftModel:
    """Deterministic 1-D model whose action 0 drifts the state toward a goal.

    States are 1-D positions. Action 0 adds ``+1`` per step; a state is terminal
    once it reaches ``goal``. Observations equal the next state. This makes an
    episode's length and terminal step fully predictable.
    """

    def __init__(self, *, device: torch.device, goal: float = 3.0) -> None:
        self._device = device
        self._goal = goal
        self.num_actions = 2

    @property
    def device(self) -> torch.device:
        return self._device

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        return states + (actions == 0).to(states.dtype).unsqueeze(1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return next_states.clone()

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions
        return next_states[:, 0].clone()

    def terminal_mask(self, states: Tensor) -> Tensor:
        return states[:, 0] >= self._goal

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions
        return -torch.abs(observations[:, 0] - next_states[:, 0])

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return torch.floor(observations[:, 0]).to(torch.int64)


def _drift_runner(
    device: torch.device, *, num_belief_particles: int = 64, max_steps: int = 10
) -> VOPPEpisodeRunner:
    model = DriftModel(device=device)
    planner = VOPPPlanner(
        model,
        num_actions=model.num_actions,
        num_particles=64,
        max_depth=4,
        num_planning_iterations=4,
    )
    return VOPPEpisodeRunner(
        planner, model, num_belief_particles=num_belief_particles, max_steps=max_steps
    )


@pytest.mark.parametrize("num_belief_particles, max_steps", [(0, 5), (16, 0)])
def test_runner_rejects_non_positive_configuration(num_belief_particles, max_steps):
    """Test that non-positive runner configuration is rejected.

    Purpose: Validates the constructor guards on particle count and step budget.

    Given: A valid planner/model and one non-positive configuration value
    When: A runner is constructed with that value
    Then: ``ValueError`` is raised

    Test type: unit
    """
    device = torch.device("cpu")
    model = DriftModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions)
    with pytest.raises(ValueError):
        VOPPEpisodeRunner(
            planner,
            model,
            num_belief_particles=num_belief_particles,
            max_steps=max_steps,
        )


def test_runner_rejects_non_single_initial_state():
    """Test that a non-single initial state is rejected.

    Purpose: Validates the initial-state shape guard.

    Given: A runner and a two-row initial-state tensor
    When: ``run_episode`` is called with it
    Then: ``ValueError`` is raised

    Test type: unit
    """
    runner = _drift_runner(torch.device("cpu"))
    with pytest.raises(ValueError):
        runner.run_episode(torch.zeros(2, 1))


def test_runner_records_consistent_trajectory_lengths():
    """Test that the recorded trajectory has internally consistent lengths.

    Purpose: Validates the per-step bookkeeping of the episode result.

    Given: A deterministic drift model and a single-state root belief
    When: An episode is run
    Then: There is one more state than actions, and beliefs / rewards /
        plan-times / root-visit-counts all match the number of actions

    Test type: integration
    """
    torch.manual_seed(0)
    runner = _drift_runner(torch.device("cpu"))
    result = runner.run_episode(torch.zeros(1, 1))
    steps = result.num_steps
    assert len(result.states) == steps + 1
    assert len(result.beliefs) == steps
    assert len(result.rewards) == steps
    assert len(result.plan_times) == steps
    assert len(result.root_visit_counts) == steps


def test_runner_reaches_terminal_state():
    """Test that reaching a terminal world state ends the episode as a goal.

    Purpose: Validates terminal detection and the ``reached_terminal_state`` flag,
        independent of planner optimality.

    Given: A world transition that deterministically advances the true state by
        ``+1`` each step, reaching the goal at 3.0 on the third step
    When: An episode is run from the origin with a generous step budget
    Then: The episode reaches the goal in exactly three steps and the final
        recorded state is at the goal

    Test type: integration
    """
    device = torch.device("cpu")
    model = DriftModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)

    def world_transition(states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return states + 1.0

    runner = VOPPEpisodeRunner(
        planner, model, num_belief_particles=32, max_steps=10, world_transition=world_transition
    )
    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))
    assert result.reached_terminal_state
    assert result.num_steps == 3
    assert float(result.states[-1][0]) == pytest.approx(3.0)


def test_runner_honors_world_hooks():
    """Test that injected world hooks override the ground-truth dynamics.

    Purpose: Validates that ``world_transition`` / ``world_observation`` replace
        the model's default dynamics for the ground-truth rollout.

    Given: A world transition that always advances the true state by ``+5``
    When: An episode is run for a single step
    Then: The realised next true state reflects the injected ``+5`` transition
        rather than the model's ``+1`` drift

    Test type: unit
    """
    device = torch.device("cpu")
    model = DriftModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)

    def world_transition(states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return states + 5.0

    runner = VOPPEpisodeRunner(
        planner,
        model,
        num_belief_particles=32,
        max_steps=1,
        world_transition=world_transition,
    )
    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))
    assert float(result.states[-1][0]) == pytest.approx(5.0)


class DegenerateObservationModel(DriftModel):
    """Drift model whose observation likelihood is ``-inf`` for every particle.

    Reproduces the total-degeneracy case where no particle explains the
    observation, which must not crash the SIR belief filter.
    """

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions, observations
        return torch.full((next_states.shape[0],), float("-inf"), device=next_states.device)


def test_filter_belief_survives_all_zero_likelihood():
    """Test that the SIR filter falls back to uniform on total degeneracy.

    Purpose: Validates that an observation with zero likelihood under every
        particle resamples uniformly instead of producing NaN weights.

    Given: A model whose observation log-probs are all ``-inf``
    When: An episode is run for several steps
    Then: The episode completes without error and every belief keeps the
        configured particle count and stays finite

    Test type: unit
    """
    device = torch.device("cpu")
    model = DegenerateObservationModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)
    runner = VOPPEpisodeRunner(planner, model, num_belief_particles=32, max_steps=4)
    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))
    assert result.num_steps >= 1
    assert all(bool(torch.isfinite(belief).all()) for belief in result.beliefs)
    assert all(belief.shape == (32, 1) for belief in result.beliefs)


def test_runner_light_dark_end_to_end_reaches_goal():
    """Test an end-to-end VOPP episode on the real light-dark vectorized model.

    Purpose: Validates that the runner drives a full closed-loop episode on the
        real Continuous Light-Dark vectorized model and reaches the goal.

    Given: The light-dark vectorized model and a start belief at ``(0, 5)``
    When: A VOPP episode is run with a modest planning budget
    Then: The episode reaches the goal and every belief carries the configured
        number of particles

    Test type: integration
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    model = ContinuousLightDarkVectorizedModel(env, device=device)
    planner = VOPPPlanner(
        model,
        num_actions=model.num_actions,
        num_particles=256,
        max_depth=10,
        num_planning_iterations=16,
    )
    runner = VOPPEpisodeRunner(planner, model, num_belief_particles=256, max_steps=40)
    result = runner.run_episode(torch.tensor([[0.0, 5.0]]))
    assert result.reached_terminal_state
    assert all(belief.shape == (256, 2) for belief in result.beliefs)


class _NonTerminatingModel(DriftModel):
    """DriftModel whose terminal_mask is constantly false.

    Mirrors ``IsaacLabVectorizedModel``, whose ``terminal_mask`` is hardcoded to
    all-False: without a world-side hook such a model can never end an episode.
    """

    def terminal_mask(self, states: Tensor) -> Tensor:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)


def test_runner_records_world_step_info_per_step():
    """Test that the world's per-step measurements are recorded.

    Purpose: Validates that VOPP episodes carry the same per-step channels as the
        standard episode loop, so one metric aggregator serves both runners

    Given: A world_step_info hook reporting an incrementing impact channel
    When: An episode of three steps is run
    Then: One mapping per executed action is recorded, in order

    Test type: unit
    """
    device = torch.device("cpu")
    model = _NonTerminatingModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)
    calls = {"n": 0}

    def world_step_info() -> dict:
        calls["n"] += 1
        return {"impact": float(calls["n"]), "success": 0.0}

    runner = VOPPEpisodeRunner(
        planner, model, num_belief_particles=32, max_steps=3, world_step_info=world_step_info
    )
    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))

    assert result.num_steps == 3
    assert [info["impact"] for info in result.step_infos] == [1.0, 2.0, 3.0]


def test_step_infos_empty_without_the_hook():
    """Test that step_infos stays empty when no hook is supplied.

    Purpose: Validates that the new field is opt-in and costs nothing for
        existing callers

    Given: A runner constructed without world_step_info
    When: An episode is run
    Then: No step infos are recorded

    Test type: unit
    """
    device = torch.device("cpu")
    model = DriftModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)
    runner = VOPPEpisodeRunner(planner, model, num_belief_particles=32, max_steps=2)

    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))

    assert not result.step_infos


def test_world_terminal_hook_ends_the_episode():
    """Test that the world can end an episode a non-terminating model cannot.

    Purpose: Validates the path that makes task completion measurable at all
        against a surrogate model whose terminal_mask is constantly false

    Given: A model that never reports terminal and a world_terminal hook firing
        on the second step
    When: An episode with max_steps=10 is run
    Then: The episode stops after two steps and reports having reached the goal

    Test type: unit
    """
    device = torch.device("cpu")
    model = _NonTerminatingModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)
    calls = {"n": 0}

    def world_terminal() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    runner = VOPPEpisodeRunner(
        planner, model, num_belief_particles=32, max_steps=10, world_terminal=world_terminal
    )
    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))

    assert result.num_steps == 2
    assert result.reached_terminal_state is True


def test_without_world_terminal_a_non_terminating_model_runs_to_max_steps():
    """Test the failure mode the world_terminal hook exists to fix.

    Purpose: Documents that a model whose terminal_mask is always false never
        ends an episode early nor reports reaching the goal, which is why task
        completion needs a world-side signal

    Given: A model that never reports terminal and no world_terminal hook
    When: An episode with max_steps=4 is run
    Then: All four steps execute and reached_terminal_state stays False

    Test type: unit
    """
    device = torch.device("cpu")
    model = _NonTerminatingModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)
    runner = VOPPEpisodeRunner(planner, model, num_belief_particles=32, max_steps=4)

    torch.manual_seed(0)
    result = runner.run_episode(torch.zeros(1, 1))

    assert result.num_steps == 4
    assert result.reached_terminal_state is False


def test_vopp_step_infos_feed_the_shared_metric_aggregator():
    """Test that VOPP results flow into the shared metric aggregation.

    Purpose: Validates the design claim that the VOPP runner and the standard
        episode loop produce the same metric shapes despite different result types

    Given: Two VOPP episodes reporting success on one of them
    When: Their step_infos are aggregated with a completion-rate spec
    Then: task_completion_rate is 0.5

    Test type: integration
    """
    device = torch.device("cpu")
    model = _NonTerminatingModel(device=device)
    planner = VOPPPlanner(model, num_actions=model.num_actions, num_planning_iterations=2)

    def make_result(success: float) -> VOPPEpisodeResult:
        runner = VOPPEpisodeRunner(
            planner,
            model,
            num_belief_particles=32,
            max_steps=2,
            world_step_info=lambda: {"success": success},
        )
        torch.manual_seed(0)
        return runner.run_episode(torch.zeros(1, 1))

    episodes = [make_result(1.0).step_infos, make_result(0.0).step_infos]
    spec = StepInfoMetric(
        name="task_completion_rate",
        channel="success",
        per_episode=EpisodeReduction.ANY,
    )

    metrics = aggregate_step_info_metrics(episodes, [spec])

    assert metrics[0].name == "task_completion_rate"
    assert metrics[0].value == pytest.approx(0.5)


def test_reached_goal_alias_reads_and_writes_terminal_state():
    """Test the deprecated reached_goal alias in both directions.

    Purpose: Validates that renaming the flag to reached_terminal_state does not
        break attribute reads or writes for existing callers, while documenting
        that the flag means generic termination rather than success

    Given: A fresh VOPPEpisodeResult
    When: The alias is read, written, and the canonical field is written
    Then: Both names stay in sync in both directions

    Test type: unit
    """
    result = VOPPEpisodeResult()

    assert result.reached_goal is False

    result.reached_goal = True
    assert result.reached_terminal_state is True

    result.reached_terminal_state = False
    assert result.reached_goal is False
