# SPDX-License-Identifier: MIT

"""Tests for EpisodeRunner (POMDPPlanners/simulations/episodes.py).

These currently focus on the two-environment split (world vs planner model).
The EpisodeRunner samples the executed trajectory (reward, next state,
observation, terminal, initial state) from the ground-truth *world* environment,
while the belief filter runs on the planner's own generative *model*
(``policy.environment``). These tests use two distinct TigerPOMDP instances —
one spied world, one spied model — plus a fixed-action policy, so the routing of
each call can be asserted directly. A single environment (world is the model)
must preserve the classic behavior.
"""

# pylint: disable=protected-access  # Tests inspect spy call counters

from typing import Any, Dict, List, Optional
from pathlib import Path

import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.simulations.episodes import EpisodeRunner, run_episode


class SpyTiger(TigerPOMDP):
    """A TigerPOMDP that records how many times each routed method is called."""

    def __init__(self, discount_factor: float = 0.95) -> None:
        super().__init__(discount_factor=discount_factor)
        self._calls: Dict[str, int] = {
            "reward": 0,
            "sample_next_state": 0,
            "sample_next_state_batch": 0,
            "sample_observation": 0,
            "is_terminal": 0,
            "initial_state_dist": 0,
        }

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        self._calls["reward"] += 1
        return super().reward(state, action, next_state)

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        self._calls["sample_next_state"] += 1
        return super().sample_next_state(state, action, n_samples)

    def sample_next_state_batch(self, states: Any, action: Any) -> Any:
        self._calls["sample_next_state_batch"] += 1
        return super().sample_next_state_batch(states, action)

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        self._calls["sample_observation"] += 1
        return super().sample_observation(next_state, action, n_samples)

    def is_terminal(self, state: Any) -> bool:
        self._calls["is_terminal"] += 1
        return super().is_terminal(state)

    def initial_state_dist(self) -> Any:
        self._calls["initial_state_dist"] += 1
        return super().initial_state_dist()


class TerminatingTiger(SpyTiger):
    """A spied TigerPOMDP whose terminal condition is scripted by step count.

    ``is_terminal`` returns True once ``terminate_after`` real transitions have
    been taken (counted via ``sample_next_state``), giving deterministic control
    over when an episode ends. ``terminate_after=0`` makes the initial state
    terminal; a large value makes the episode run to its step budget.
    """

    def __init__(self, terminate_after: int, discount_factor: float = 0.95) -> None:
        super().__init__(discount_factor=discount_factor)
        self._terminate_after = terminate_after
        self._steps_taken = 0

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        self._steps_taken += 1
        return super().sample_next_state(state, action, n_samples)

    def is_terminal(self, state: Any) -> bool:
        super().is_terminal(state)  # keep the spy counter consistent
        return self._steps_taken >= self._terminate_after


class FixedActionPolicy(Policy):
    """A trivial discrete policy that returns ``actions_per_call`` listen actions."""

    def __init__(
        self,
        environment: Environment,
        discount_factor: float,
        name: str = "FixedActionPolicy",
        actions_per_call: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False,
    ) -> None:
        self.actions_per_call = actions_per_call
        super().__init__(environment, discount_factor, name, log_path=log_path, debug=debug)

    def action(self, belief: Belief):
        del belief
        return (["listen"] * self.actions_per_call, PolicyRunData(info_variables=[]))

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return []


def _run_two_env_episode(num_steps: int = 3):
    """Run one episode with a distinct spied world and spied model; return spies+history."""
    model = SpyTiger(discount_factor=0.95)
    world = SpyTiger(discount_factor=0.95)
    policy = FixedActionPolicy(environment=model, discount_factor=0.95)
    belief = get_initial_belief(model, n_particles=5)

    # Snapshot after belief construction but before the runner, since the
    # ground-truth initial state is sampled inside EpisodeRunner.__init__.
    world_before = dict(world._calls)
    model_before = dict(model._calls)
    runner = EpisodeRunner(
        environment=world, policy=policy, initial_belief=belief, num_steps=num_steps, logger=None
    )
    history = runner.run()

    world_delta = {k: v - world_before[k] for k, v in world._calls.items()}
    model_delta = {k: v - model_before[k] for k, v in model._calls.items()}
    return world_delta, model_delta, history


def test_two_env_reward_and_transition_come_from_world() -> None:
    """Reward, next-state and observation are sampled from the world environment.

    Purpose: Validates the executed trajectory is drawn from the world, not the model.

    Given: A distinct spied world and spied model with a fixed-action policy
    When: An episode is run through EpisodeRunner
    Then: The world's reward/sample_next_state/sample_observation are exercised
        while the model's reward is never called

    Test type: integration
    """
    world_delta, model_delta, _ = _run_two_env_episode()

    assert world_delta["reward"] > 0
    assert world_delta["sample_next_state"] > 0
    assert world_delta["sample_observation"] > 0
    # The model is a generative model for the belief, never the reward source.
    assert model_delta["reward"] == 0


def test_two_env_initial_state_drawn_from_world() -> None:
    """The ground-truth initial state comes from the world's initial distribution.

    Purpose: Validates initial-state sourcing switches to the world when world != model.

    Given: A distinct spied world and spied model
    When: An episode is run
    Then: The world's initial_state_dist is queried during the run

    Test type: integration
    """
    world_delta, _, _ = _run_two_env_episode()
    assert world_delta["initial_state_dist"] >= 1


def test_two_env_belief_update_runs_on_model() -> None:
    """Belief particle propagation runs on the planner's model environment.

    Purpose: Validates the belief update uses model, not world.

    Given: A fixed-action policy (no planning) so model sampling can only come
        from the belief update
    When: An episode is run
    Then: The model's sample_next_state is exercised (belief propagation) and a
        non-empty history is produced

    Test type: integration
    """
    _, model_delta, history = _run_two_env_episode()
    # Tiger's WeightedParticleBelief update propagates particles via the
    # vectorized batch entry point on the model environment.
    assert model_delta["sample_next_state_batch"] > 0
    assert len(history.history) > 0


class EncodingSpyModel(SpyTiger):
    """A model whose ``encode_observation`` tags observations.

    Tagging lets a test tell apart the raw observation the world emitted (recorded
    in history) from the encoded observation the belief update consumes. The
    belief-update likelihood entry point (``observation_log_probability_per_state``)
    records the observation it was handed, then strips the tag before delegating so
    the underlying Tiger scoring still works.
    """

    def __init__(self, discount_factor: float = 0.95) -> None:
        super().__init__(discount_factor=discount_factor)
        self.encoded_inputs: List[Any] = []
        self.belief_observations: List[Any] = []

    def encode_observation(self, observation: Any) -> Any:
        self.encoded_inputs.append(observation)
        return ("encoded", observation)

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> Any:
        self.belief_observations.append(observation)
        raw = (
            observation[1]
            if isinstance(observation, tuple) and observation[0] == "encoded"
            else observation
        )
        assert isinstance(raw, str)  # decoded Tiger observation; narrows for the typed super call
        return super().observation_log_probability_per_state(next_states, action, raw)


def test_two_env_history_records_raw_and_belief_receives_encoded() -> None:
    """History keeps the raw observation while the belief update consumes the encoded one.

    Purpose: Validates the encode_observation seam routes raw to history and encoded to belief

    Given: A world emitting raw Tiger observations and a model whose encode_observation
        tags each observation
    When: An episode is run through EpisodeRunner
    Then: Each recorded history observation is the raw (untagged) world observation,
        encode_observation saw those exact raw observations, and the belief update was
        handed the encoded (tagged) form corresponding to each

    Test type: integration
    """
    model = EncodingSpyModel(discount_factor=0.95)
    world = SpyTiger(discount_factor=0.95)
    policy = FixedActionPolicy(environment=model, discount_factor=0.95)
    belief = get_initial_belief(model, n_particles=5)

    history = EpisodeRunner(
        environment=world, policy=policy, initial_belief=belief, num_steps=3, logger=None
    ).run()

    steps = [step for step in history.history if step.observation is not None]
    assert steps  # sanity: the episode produced non-terminal steps
    # History keeps the raw world observation, never the encoded tag.
    for step in steps:
        assert not (isinstance(step.observation, tuple) and step.observation[0] == "encoded")
    # encode_observation was applied to exactly those raw observations.
    assert model.encoded_inputs == [step.observation for step in steps]
    # The belief update was fed the encoded form corresponding to each raw observation.
    assert model.belief_observations == [("encoded", step.observation) for step in steps]


def test_two_env_discount_mismatch_raises() -> None:
    """A world/model discount-factor mismatch is rejected at construction.

    Purpose: Validates the discount-consistency guard for the two-env case.

    Given: A world discounted at 0.9 and a policy model discounted at 0.95
    When: An EpisodeRunner is constructed
    Then: ValueError is raised naming the discount mismatch

    Test type: unit
    """
    model = SpyTiger(discount_factor=0.95)
    world = SpyTiger(discount_factor=0.9)
    policy = FixedActionPolicy(environment=model, discount_factor=0.95)
    belief = get_initial_belief(model, n_particles=3)

    with pytest.raises(ValueError, match="discount_factor"):
        EpisodeRunner(
            environment=world, policy=policy, initial_belief=belief, num_steps=2, logger=None
        )


def test_single_env_backcompat_initial_state_from_belief() -> None:
    """With one environment, initial state comes from the belief (classic behavior).

    Purpose: Validates zero behavior change when world is the model.

    Given: A single spied env used as both the world and the policy's model
    When: An episode is run
    Then: The env's initial_state_dist is NOT called by the runner (the belief
        sample seeds the true state) while reward/transition still route to it

    Test type: integration
    """
    env = SpyTiger(discount_factor=0.95)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)

    runner = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=3, logger=None
    )
    before = dict(env._calls)
    runner.run()
    delta = {k: v - before[k] for k, v in env._calls.items()}

    assert delta["initial_state_dist"] == 0
    assert delta["reward"] > 0
    assert delta["sample_next_state"] > 0


@pytest.fixture(name="valid_kwargs")
def _valid_kwargs() -> Dict[str, Any]:
    """A set of valid EpisodeRunner constructor arguments to mutate per case."""
    env = TigerPOMDP(discount_factor=0.95)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=3)
    return {"environment": env, "policy": policy, "initial_belief": belief, "num_steps": 3}


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("environment", None, ValueError),
        ("policy", None, ValueError),
        ("initial_belief", None, ValueError),
        ("num_steps", None, ValueError),
        ("environment", object(), TypeError),
        ("policy", "not_a_policy", TypeError),
        ("initial_belief", 3, TypeError),
        ("num_steps", "3", TypeError),
        ("num_steps", 0, ValueError),
        ("num_steps", -1, ValueError),
    ],
)
def test_episode_runner_rejects_invalid_inputs(
    valid_kwargs: Dict[str, Any], field: str, value: Any, expected: type
) -> None:
    """Invalid constructor inputs are rejected with the documented error type.

    Purpose: Validates _validate_episode_inputs covers None/type/positivity checks.

    Given: An otherwise-valid set of EpisodeRunner arguments
    When: One argument is replaced with a None, wrong-typed, or non-positive value
    Then: The matching ValueError/TypeError is raised at construction

    Test type: unit
    """
    kwargs = dict(valid_kwargs)
    kwargs[field] = value
    with pytest.raises(expected):
        EpisodeRunner(**kwargs)


def test_episode_stops_at_terminal_and_records_terminal_step() -> None:
    """Reaching a terminal world state ends the episode with a terminal step.

    Purpose: Validates early termination bookkeeping (_should_continue/_add_terminal_step).

    Given: A world scripted to become terminal after two transitions, budget 10
    When: An episode is run
    Then: reach_terminal_state is True, exactly two normal steps are taken, and a
        trailing terminal step with all-None fields is appended

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=2)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)

    history = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=10, logger=None
    ).run()

    assert history.reach_terminal_state is True
    assert history.actual_num_steps == 2
    normal_steps = [step for step in history.history if step.action is not None]
    assert len(normal_steps) == 2
    terminal = history.history[-1]
    assert terminal.action is None
    assert terminal.next_state is None
    assert terminal.observation is None
    assert terminal.reward is None


def test_episode_terminal_at_start_records_only_terminal_step() -> None:
    """An already-terminal initial state ends the episode immediately.

    Purpose: Validates the step-0 terminal edge case.

    Given: A world whose initial state is already terminal
    When: An episode is run
    Then: No policy action is executed and the history is a single terminal step

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=0)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)

    history = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=5, logger=None
    ).run()

    assert history.reach_terminal_state is True
    assert history.actual_num_steps == 0
    assert len(history.history) == 1
    assert history.history[0].action is None


def test_episode_runs_full_length_when_never_terminal() -> None:
    """A non-terminal world runs exactly num_steps with no terminal step.

    Purpose: Validates the step-budget stop path.

    Given: A world that never terminates and a budget of four steps
    When: An episode is run
    Then: Exactly four normal steps are recorded, reach_terminal_state is False,
        and no trailing terminal step is appended

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=999)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)

    history = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=4, logger=None
    ).run()

    assert history.reach_terminal_state is False
    assert history.actual_num_steps == 4
    assert len(history.history) == 4
    assert all(step.action is not None for step in history.history)


def test_recorded_steps_chain_pretransition_state_to_next_state() -> None:
    """Each StepData records the pre-transition state, chaining across steps.

    Purpose: Validates _record_step captures state before the transition is applied.

    Given: A non-terminal world run for four steps
    When: The recorded history is inspected
    Then: Every step's state equals the previous step's next_state, and every
        normal step has populated next_state/observation/reward fields

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=999)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)

    history = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=4, logger=None
    ).run()

    steps = [step for step in history.history if step.action is not None]
    assert len(steps) == 4
    for previous, current in zip(steps, steps[1:]):
        assert current.state == previous.next_state
    for step in steps:
        assert step.next_state is not None
        assert step.observation is not None
        assert step.reward is not None


def test_multi_action_policy_step_breaks_at_num_steps() -> None:
    """A policy emitting several actions per call still stops at the step budget.

    Purpose: Validates _execute_policy_step iterates actions and breaks at num_steps.

    Given: A policy returning two actions per selection and a budget of three
    When: An episode is run
    Then: Exactly three actions execute (the fourth is cut off mid-selection) over
        two policy selections

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=999)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95, actions_per_call=2)
    belief = get_initial_belief(env, n_particles=5)

    runner = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=3, logger=None
    )
    history = runner.run()

    assert history.actual_num_steps == 3
    normal_steps = [step for step in history.history if step.action is not None]
    assert len(normal_steps) == 3
    assert len(runner.policy_run_data) == 2


def test_initial_belief_is_deep_copied() -> None:
    """The runner copies the initial belief and never mutates the caller's object.

    Purpose: Validates the copy.deepcopy of initial_belief.

    Given: A caller-owned initial belief
    When: An episode is run
    Then: The runner's belief is a distinct object and the caller's belief
        particles are unchanged

    Test type: unit
    """
    env = TerminatingTiger(terminate_after=999)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=5)
    original_particles = list(belief.particles)

    runner = EpisodeRunner(
        environment=env, policy=policy, initial_belief=belief, num_steps=3, logger=None
    )
    assert runner.belief is not belief
    runner.run()

    assert list(belief.particles) == original_particles


def test_run_episode_wrapper_returns_history() -> None:
    """run_episode constructs a runner and returns its History.

    Purpose: Validates the public run_episode entry point.

    Given: A non-terminal world, fixed policy and initial belief
    When: run_episode is called with a three-step budget
    Then: A History is returned with the expected number of executed steps

    Test type: integration
    """
    env = TerminatingTiger(terminate_after=999)
    policy = FixedActionPolicy(environment=env, discount_factor=0.95)
    belief = get_initial_belief(env, n_particles=3)

    history = run_episode(
        environment=env, policy=policy, initial_belief=belief, num_steps=3, logger=None
    )

    assert isinstance(history.history, list)
    assert history.actual_num_steps == 3
