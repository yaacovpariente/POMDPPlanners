# SPDX-License-Identifier: MIT

"""Tests for the Vectorized Online POMDP Planner (VOPP / PORPP).

The suite covers construction and validation, the forward-search /
preference-backup control flow driven by a fully controllable mock generative
model, and a smoke / geometry integration test against the real Continuous
Light-Dark vectorized model.
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


class MockGenerativeModel:
    """Deterministic, fully controllable vectorized generative model.

    States are 1-D scalars left unchanged by every transition, so all episodes
    from a belief share one observation key and the search grows a compact,
    predictable path. Immediate reward is a per-action constant read from
    ``reward_table``, letting a test pin which action must win. Termination is
    toggled by ``always_terminal``.
    """

    def __init__(
        self,
        reward_table: Tensor,
        *,
        device: torch.device,
        always_terminal: bool = False,
    ) -> None:
        self._device = device
        self._reward_table = reward_table.to(device)
        self.num_actions = int(reward_table.shape[0])
        self._always_terminal = always_terminal

    @property
    def device(self) -> torch.device:
        return self._device

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return states.clone()

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return next_states.clone()

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, next_states
        return self._reward_table[actions]

    def terminal_mask(self, states: Tensor) -> Tensor:
        return torch.full(
            (states.shape[0],), self._always_terminal, dtype=torch.bool, device=self._device
        )

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions, observations
        return torch.zeros(next_states.shape[0], device=self._device)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return torch.floor(observations[:, 0] * 10.0).to(torch.int64)


def _mock_planner(reward_table: Tensor, **kwargs) -> VOPPPlanner:
    device = torch.device("cpu")
    model = MockGenerativeModel(reward_table, device=device)
    defaults = {
        "num_particles": 256,
        "max_depth": 4,
        "num_planning_iterations": 6,
        "discount_factor": 0.95,
    }
    defaults.update(kwargs)
    return VOPPPlanner(model, num_actions=model.num_actions, **defaults)


def _light_dark_planner(**kwargs) -> VOPPPlanner:
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))
    defaults = {
        "num_particles": 512,
        "max_depth": 6,
        "num_planning_iterations": 12,
        "discount_factor": 0.95,
    }
    defaults.update(kwargs)
    return VOPPPlanner(model, num_actions=model.num_actions, **defaults)


def test_vopp_registers_preference_and_value_fields():
    """Test that construction registers correctly shaped preference/value fields.

    Purpose: Validates that the planner registers a per-action preference field
        and a scalar value field on its belief tree.

    Given: A VOPP planner built for a three-action mock model
    When: The registered belief fields are inspected on the root belief
    Then: ``preferences`` has one column per action (initialised to zero) and
        ``value`` is a scalar field

    Test type: unit
    """
    planner = _mock_planner(torch.tensor([1.0, 0.0, 0.0]))
    preferences = planner.tree.belief_field("preferences")
    value = planner.tree.belief_field("value")
    assert preferences.shape == (1, 3)
    assert value.shape == (1,)
    assert bool((preferences == 0.0).all())


def test_vopp_plan_returns_valid_action_index():
    """Test that planning returns an in-range integer action index.

    Purpose: Validates the greedy root action returned by :meth:`plan`.

    Given: A VOPP planner for a three-action mock model
    When: ``plan`` is called on a particle-set root belief
    Then: The returned action is an ``int`` in ``[0, num_actions)``

    Test type: unit
    """
    torch.manual_seed(0)
    planner = _mock_planner(torch.tensor([1.0, 0.0, 0.0]))
    root = torch.zeros(64, 1)
    action = planner.plan(root)
    assert isinstance(action, int)
    assert 0 <= action < 3


def test_vopp_plan_is_deterministic_under_fixed_seed():
    """Test that planning is reproducible when the RNG seed is fixed.

    Purpose: Validates that a fixed torch seed yields identical planning output.

    Given: The same planner configuration and root belief
    When: ``plan`` is run twice, each preceded by the same manual seed
    Then: Both runs return the same action and identical root preferences

    Test type: unit
    """
    root = torch.zeros(64, 1)
    torch.manual_seed(7)
    planner_a = _mock_planner(torch.tensor([0.0, 1.0, 0.0]))
    action_a = planner_a.plan(root)
    prefs_a = planner_a.tree.belief_field("preferences")[0].clone()
    torch.manual_seed(7)
    planner_b = _mock_planner(torch.tensor([0.0, 1.0, 0.0]))
    action_b = planner_b.plan(root)
    prefs_b = planner_b.tree.belief_field("preferences")[0].clone()
    assert action_a == action_b
    assert torch.allclose(prefs_a, prefs_b)


def test_vopp_selects_dominant_reward_action():
    """Test that the planner selects the action with dominant immediate reward.

    Purpose: Validates that preference backups push the greedy root action to
        the clearly best-rewarded action.

    Given: A mock model where only action 1 yields positive reward
    When: The planner runs its full budget of forward-search / backup passes
    Then: The greedy root action is action 1 and it has the highest preference

    Test type: unit
    """
    torch.manual_seed(1)
    planner = _mock_planner(torch.tensor([0.0, 1.0, 0.0]), num_planning_iterations=10)
    action = planner.plan(torch.zeros(256, 1))
    preferences = planner.tree.belief_field("preferences")[0]
    assert action == 1
    assert int(torch.argmax(preferences).item()) == 1


def test_vopp_forward_search_expands_tree_and_accumulates_statistics():
    """Test that the forward search grows the tree and accumulates action stats.

    Purpose: Validates that planning expands beyond the root and records visit
        counts and reward sums on action nodes.

    Given: A non-terminating mock model and a modest planning budget
    When: ``plan`` completes
    Then: The tree contains action and non-root belief nodes, and total action
        visits are positive

    Test type: integration
    """
    torch.manual_seed(2)
    planner = _mock_planner(torch.tensor([1.0, 0.0, 0.0]))
    planner.plan(torch.zeros(128, 1))
    tree = planner.tree
    num_actions = tree.num_action_nodes
    assert num_actions > 0
    assert tree.num_belief_nodes > 1
    assert int(tree.action_visit_count[:num_actions].sum().item()) > 0


def test_vopp_terminal_transitions_stop_expansion():
    """Test that fully terminal transitions prevent successor belief creation.

    Purpose: Validates the terminal-filtering step of the forward search.

    Given: A mock model whose transitions are always terminal
    When: ``plan`` runs
    Then: No belief node beyond the root is created, yet the root's action
        nodes still recorded their immediate rewards

    Test type: unit
    """
    torch.manual_seed(3)
    device = torch.device("cpu")
    model = MockGenerativeModel(torch.tensor([1.0, 0.0]), device=device, always_terminal=True)
    planner = VOPPPlanner(
        model, num_actions=2, num_particles=64, max_depth=4, num_planning_iterations=3
    )
    planner.plan(torch.zeros(64, 1))
    tree = planner.tree
    assert tree.num_belief_nodes == 1
    assert tree.num_action_nodes > 0
    assert int(tree.action_visit_count[: tree.num_action_nodes].sum().item()) > 0


@pytest.mark.parametrize(
    "temperature, num_particles, max_depth, num_planning_iterations",
    [
        (0.0, 8, 4, 4),
        (1.0, 0, 4, 4),
        (1.0, 8, 0, 4),
        (1.0, 8, 4, 0),
    ],
)
def test_vopp_invalid_hyperparameters_raise(
    temperature, num_particles, max_depth, num_planning_iterations
):
    """Test that non-positive hyperparameters are rejected at construction.

    Purpose: Validates constructor guards on temperature, particle count,
        depth, and iteration budget.

    Given: A mock model and one non-positive hyperparameter
    When: A planner is constructed with that value
    Then: ``ValueError`` is raised

    Test type: unit
    """
    model = MockGenerativeModel(torch.tensor([1.0, 0.0]), device=torch.device("cpu"))
    with pytest.raises(ValueError):
        VOPPPlanner(
            model,
            num_actions=2,
            temperature=temperature,
            num_particles=num_particles,
            max_depth=max_depth,
            num_planning_iterations=num_planning_iterations,
        )


def test_vopp_invalid_num_actions_raises():
    """Test that a non-positive action count is rejected at construction.

    Purpose: Validates the ``num_actions`` guard.

    Given: A mock model
    When: A planner is constructed with ``num_actions=0``
    Then: ``ValueError`` is raised

    Test type: unit
    """
    model = MockGenerativeModel(torch.tensor([1.0, 0.0]), device=torch.device("cpu"))
    with pytest.raises(ValueError):
        VOPPPlanner(model, num_actions=0)


def test_vopp_rejects_non_2d_root_particles():
    """Test that ``plan`` rejects a root belief that is not 2-D.

    Purpose: Validates the root-particle shape guard.

    Given: A VOPP planner and a 1-D root-particle tensor
    When: ``plan`` is called with it
    Then: ``ValueError`` is raised

    Test type: unit
    """
    planner = _mock_planner(torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        planner.plan(torch.zeros(8))


def test_vopp_light_dark_moves_toward_goal():
    """Test that VOPP prefers the action pointing toward the goal.

    Purpose: Validates end-to-end planning on the real Continuous Light-Dark
        vectorized model produces a sensible greedy action.

    Given: A root belief concentrated to the left of the goal (goal at x=10)
    When: The planner plans a step
    Then: The greedy action is the +x move (action index 2) that reduces the
        distance to the goal

    Test type: integration
    """
    torch.manual_seed(0)
    planner = _light_dark_planner()
    root_particles = torch.tensor([[2.0, 5.0]]).repeat(512, 1)
    action = planner.plan(root_particles)
    assert action == 2


def test_vopp_tree_metrics_report_expected_names_and_values():
    """Test that tree metrics mirror the MCTS metric set after planning.

    Purpose: Validates that :meth:`VOPPPlanner.tree_metrics` reports the same
        analysis metrics the MCTS planners emit, with sensible values.

    Given: A non-terminating mock model planned for a fixed budget
    When: ``tree_metrics`` is queried after ``plan``
    Then: The standard metric names are present, the root is not a leaf, the
        root visit count equals particles times iterations, and the number of
        root actions does not exceed the action set size

    Test type: integration
    """
    torch.manual_seed(4)
    planner = _mock_planner(
        torch.tensor([1.0, 0.0, 0.0]), num_particles=128, num_planning_iterations=5
    )
    planner.plan(torch.zeros(128, 1))
    metrics = {variable.name: variable.value for variable in planner.tree_metrics()}
    for name in (
        "min_actions_visit_count",
        "max_actions_visit_count",
        "actions_visit_count_entropy",
        "n_actions_from_root",
        "root_visit_count",
        "tree_max_depth",
        "is_leaf",
    ):
        assert name in metrics
    assert metrics["is_leaf"] == 0
    assert metrics["root_visit_count"] == 128 * 5
    assert 0 < metrics["n_actions_from_root"] <= 3
    assert metrics["tree_max_depth"] >= 1


def test_vopp_tree_metrics_report_leaf_before_planning():
    """Test that an unexpanded tree reports the leaf metric subset.

    Purpose: Validates the leaf branch of the vectorized metric computation.

    Given: A freshly constructed planner whose tree holds only the root
    When: ``tree_metrics`` is queried before any planning
    Then: ``is_leaf`` is 1 and the visit statistics are zero

    Test type: unit
    """
    planner = _mock_planner(torch.tensor([1.0, 0.0, 0.0]))
    metrics = {variable.name: variable.value for variable in planner.tree_metrics()}
    assert metrics["is_leaf"] == 1
    assert metrics["min_actions_visit_count"] == 0
    assert metrics["max_actions_visit_count"] == 0


def test_vopp_module_docstring_example():
    """Test the runnable example from the module docstring.

    Purpose: Validates that the documented usage example executes and returns a
        valid action.

    Given: The exact setup from the :mod:`vopp` module docstring
    When: ``plan`` is called
    Then: A valid in-range action index is returned

    Test type: example
    """
    torch.manual_seed(0)
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))
    planner = VOPPPlanner(
        model,
        num_actions=model.num_actions,
        num_particles=256,
        max_depth=5,
        num_planning_iterations=8,
        discount_factor=0.95,
    )
    root_particles = torch.tensor([[2.0, 2.0]]).repeat(256, 1)
    action = planner.plan(root_particles)
    assert 0 <= action < model.num_actions
