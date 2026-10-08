# SPDX-License-Identifier: MIT

"""Parity and behaviour tests for the vectorized analytic manipulator model.

The torch kernels duplicate numeric code that also exists in numpy, so most of these tests pin the
two to each other: a drift between them would not raise anywhere, it would simply mean the planner
searched a different model from the one the tests describe.

The action-separation tests are the regression guard for the measured failure — under the fitted
linear surrogate every action scored alike and the planner emitted one index for a whole episode.
"""

from typing import Any

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    IsaacChannelSchema,
    ManipulatorIsaacModel,
    franka_panda_chain,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_vectorized_model import (  # noqa: E501  pylint: disable=line-too-long
    ManipulatorVectorizedModel,
)

JOINTS = 7
COMMAND_WIDTH = 7
SCHEMA = IsaacChannelSchema(
    (
        ("joint_pos", JOINTS),
        ("joint_vel", JOINTS),
        ("command", COMMAND_WIDTH),
        ("last_action", JOINTS),
    )
)
STATE_DIM = SCHEMA.total_dim
ISAAC_DEFAULT_POSE = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
PRESETS = [
    np.zeros(JOINTS),
    np.full(JOINTS, 0.4),
    np.full(JOINTS, -0.4),
    np.linspace(-0.5, 0.5, JOINTS),
]
CPU = torch.device("cpu")


def _scalar_model(**overrides: Any) -> ManipulatorIsaacModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": PRESETS,
        "discount_factor": 0.99,
        "step_dt": 0.1,
        "tracking_gain": 0.4,
        "chain": franka_panda_chain(),
        "default_joint_positions": ISAAC_DEFAULT_POSE,
        "action_scale": 0.5,
        "position_noise_std": 1e-9,
        "velocity_noise_std": 1e-9,
        "action_noise_std": 1e-9,
    }
    settings.update(overrides)
    return ManipulatorIsaacModel(**settings)


def _vectorized(**overrides: Any) -> ManipulatorVectorizedModel:
    return ManipulatorVectorizedModel(_scalar_model(**overrides), device=CPU, dtype=torch.float64)


def _random_states(count: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    states = np.zeros((count, STATE_DIM))
    states[:, SCHEMA.slice_of("joint_pos")] = rng.uniform(-0.6, 0.6, size=(count, JOINTS))
    states[:, SCHEMA.slice_of("joint_vel")] = rng.uniform(-1.0, 1.0, size=(count, JOINTS))
    states[:, SCHEMA.slice_of("command")] = rng.uniform(-0.6, 0.6, size=(count, COMMAND_WIDTH))
    states[:, SCHEMA.slice_of("last_action")] = rng.uniform(-1.0, 1.0, size=(count, JOINTS))
    return torch.as_tensor(states, dtype=torch.float64, device=CPU)


# ── Parity with the scalar model ────────────────────────────────────────


def test_torch_kinematics_matches_the_numpy_chain() -> None:
    """The FK is duplicated in two languages; a drift would move the goal without any error.

    Purpose: Pins the torch forward kinematics to the numpy ModifiedDHChain

    Given: Sixteen random joint configurations
    When: Both implementations place the hand
    Then: The positions agree to within floating-point tolerance

    Test type: unit
    """
    rng = np.random.default_rng(1)
    angles = rng.uniform(-2.0, 2.0, size=(16, JOINTS))
    expected = franka_panda_chain().end_effector_position(angles)
    actual = _vectorized().end_effector_positions(
        torch.as_tensor(angles, dtype=torch.float64, device=CPU)
    )
    assert actual.numpy() == pytest.approx(expected, abs=1e-9)


def test_the_transition_mean_matches_the_scalar_joint_lag() -> None:
    """A different lag on the GPU means the planner searched a model nobody tested.

    Purpose: Pins the torch transition to the scalar JointLagTransition

    Given: Eight random states, one per action preset, and negligible process noise
    When: Both models take one step
    Then: The full next-state vectors agree

    Test type: unit
    """
    torch.manual_seed(0)
    np.random.seed(0)
    scalar = _scalar_model()
    model = ManipulatorVectorizedModel(scalar, device=CPU, dtype=torch.float64)
    states = _random_states(len(PRESETS))
    actions = torch.arange(len(PRESETS))
    batched = model.sample_next_states(states, actions).numpy()
    for index, preset in enumerate(PRESETS):
        expected = scalar.sample_next_state(states[index].numpy(), preset)
        assert batched[index] == pytest.approx(expected, abs=1e-6)


def test_the_reward_matches_the_scalar_reach_objective() -> None:
    """The reward is the whole point of the model; a mismatch optimizes the wrong thing.

    Purpose: Pins the torch reward kernel to the scalar ReachRewardModel

    Given: Twelve random states
    When: Both models score them
    Then: The rewards agree

    Test type: unit
    """
    scalar = _scalar_model()
    model = ManipulatorVectorizedModel(scalar, device=CPU, dtype=torch.float64)
    states = _random_states(12, seed=2)
    actions = torch.zeros(12, dtype=torch.int64)
    batched = model.rewards(states, actions, states).numpy()
    for index in range(12):
        row = states[index].numpy()
        assert batched[index] == pytest.approx(scalar.reward(row, PRESETS[0], row), abs=1e-9)


def test_the_reported_distance_matches_the_scalar_one() -> None:
    """The success threshold is a distance, so the model's notion of it must be the same one.

    Purpose: Pins goal_distances to ReachRewardModel.end_effector_distance

    Given: Six random states
    When: Both models measure hand-to-goal distance
    Then: The distances agree

    Test type: unit
    """
    scalar = _scalar_model()
    model = ManipulatorVectorizedModel(scalar, device=CPU, dtype=torch.float64)
    states = _random_states(6, seed=3)
    distances = model.goal_distances(states).numpy()
    reach_reward = scalar.reach_reward
    assert reach_reward is not None
    for index in range(6):
        expected = reach_reward.end_effector_distance(states[index].numpy())
        assert distances[index] == pytest.approx(expected, abs=1e-9)


# ── Behaviour of the kernels ────────────────────────────────────────────


def test_two_actions_place_the_hand_in_two_different_places() -> None:
    """This is the measured failure: a model that cannot separate actions cannot be planned in.

    Purpose: Validates that distinct action presets yield distinct predicted hand positions

    Given: One state repeated once per action preset
    When: One step is taken under each preset
    Then: The hand positions are pairwise distinct by centimetres, not float noise

    Test type: unit
    """
    torch.manual_seed(0)
    model = _vectorized()
    states = _random_states(1, seed=4).repeat(len(PRESETS), 1)
    successors = model.sample_next_states(states, torch.arange(len(PRESETS)))
    joints = successors[:, SCHEMA.slice_of("joint_pos")] + torch.as_tensor(
        ISAAC_DEFAULT_POSE, dtype=torch.float64
    )
    hands = model.end_effector_positions(joints)
    pairwise = torch.cdist(hands, hands)
    off_diagonal = pairwise[~torch.eye(len(PRESETS), dtype=torch.bool)]
    assert float(off_diagonal.min()) > 0.01


def test_the_reward_separates_the_action_presets() -> None:
    """A flat reward across actions is exactly what made the planner hold one index for 40 steps.

    Purpose: Validates that the analytic reward ranks the action presets distinctly

    Given: One state repeated once per action preset
    When: Each preset is stepped and the successor scored
    Then: The rewards are all different

    Test type: unit
    """
    torch.manual_seed(0)
    model = _vectorized()
    states = _random_states(1, seed=5).repeat(len(PRESETS), 1)
    actions = torch.arange(len(PRESETS))
    rewards = model.rewards(states, actions, model.sample_next_states(states, actions))
    assert len(set(round(float(value), 6) for value in rewards)) == len(PRESETS)


def test_the_command_block_is_carried_through_the_transition() -> None:
    """A goal the dynamics can move is a goal the planner can chase by moving it.

    Purpose: Validates that the transition leaves the commanded pose untouched

    Given: Four random states with distinct command blocks
    When: One step is taken
    Then: The command blocks are unchanged

    Test type: unit
    """
    torch.manual_seed(0)
    model = _vectorized()
    states = _random_states(4, seed=6)
    successors = model.sample_next_states(states, torch.zeros(4, dtype=torch.int64))
    command = SCHEMA.slice_of("command")
    assert successors[:, command].numpy() == pytest.approx(states[:, command].numpy())


def test_the_recorded_action_block_holds_the_action_just_applied() -> None:
    """The world's observation carries the last action, so a mismatch misweights every particle.

    Purpose: Validates the recorded-action block of the vectorized transition

    Given: One state per action preset
    When: One step is taken under each
    Then: The trailing block equals that preset

    Test type: unit
    """
    torch.manual_seed(0)
    model = _vectorized()
    states = _random_states(len(PRESETS), seed=7)
    successors = model.sample_next_states(states, torch.arange(len(PRESETS)))
    recorded = successors[:, SCHEMA.slice_of("last_action")].numpy()
    assert recorded == pytest.approx(np.asarray(PRESETS), abs=1e-6)


def test_no_state_is_terminal_in_the_model() -> None:
    """The live world owns termination; a model guessing at it would prune reachable states.

    Purpose: Validates the always-false terminal kernel

    Given: A batch of random states
    When: The terminal mask is computed
    Then: Every entry is False

    Test type: unit
    """
    mask = _vectorized().terminal_mask(_random_states(5, seed=8))
    assert mask.dtype == torch.bool
    assert not bool(mask.any())


def test_the_observation_likelihood_peaks_at_the_true_state() -> None:
    """A likelihood that did not peak on the truth would drive the belief away from it.

    Purpose: Validates the additive-Gaussian observation log-likelihood

    Given: A batch of states, an exact observation of each, and a displaced observation
    When: Both are scored
    Then: The exact observation scores higher, and matches a hand-computed Gaussian

    Test type: unit
    """
    model = _vectorized()
    states = _random_states(3, seed=9)
    actions = torch.zeros(3, dtype=torch.int64)
    exact = model.observation_log_probs(states, actions, states)
    displaced = model.observation_log_probs(states, actions, states + 0.5)
    assert bool((exact > displaced).all())
    expected_peak = -STATE_DIM * np.log(0.1) - 0.5 * STATE_DIM * np.log(2.0 * np.pi)
    assert exact.numpy() == pytest.approx(np.full(3, expected_peak), abs=1e-9)


def test_identical_observations_hash_to_one_key_and_distant_ones_do_not() -> None:
    """Tree keys are what merge branches; collapsing everything would flatten the search.

    Purpose: Validates the observation-key quantization

    Given: One observation, its exact copy, and one a full grid cell away
    When: Keys are computed with a fine resolution
    Then: The copy shares the key and the displaced one does not

    Test type: unit
    """
    model = _vectorized(step_dt=0.1)
    base = _random_states(1, seed=10)
    keys = model.observation_keys(torch.cat([base, base, base + 1.0]))
    assert int(keys[0]) == int(keys[1])
    assert int(keys[0]) != int(keys[2])


def test_action_keys_are_the_action_indices() -> None:
    """The tree addresses actions by index, so the key has to be the index itself.

    Purpose: Validates the action-key kernel

    Given: A tensor of action indices
    When: Keys are computed
    Then: They equal the indices, as int64

    Test type: unit
    """
    actions = torch.tensor([0, 3, 1, 2])
    keys = _vectorized().action_keys(actions)
    assert keys.dtype == torch.int64
    assert keys.tolist() == actions.tolist()


def test_the_franka_reach_layout_stays_in_parity() -> None:
    """The 9-observed / 7-commanded layout is the one the run uses, so parity has to hold there.

    Purpose: Pins the torch kernels to the scalar model on the real Franka reach layout

    Given: A 9 + 9 + 7 + 7 schema with explicit arm and actuated index maps
    When: Both models step and score the same states
    Then: Successors and rewards agree, and the two gripper joints are left alone

    Test type: unit
    """
    torch.manual_seed(0)
    np.random.seed(0)
    schema = IsaacChannelSchema(
        (("joint_pos", 9), ("joint_vel", 9), ("command", COMMAND_WIDTH), ("last_action", JOINTS))
    )
    scalar = _scalar_model(
        state_schema=schema, arm_joint_indices=range(JOINTS), actuated_indices=range(JOINTS)
    )
    model = ManipulatorVectorizedModel(scalar, device=CPU, dtype=torch.float64)
    rng = np.random.default_rng(11)
    rows = rng.uniform(-0.4, 0.4, size=(len(PRESETS), schema.total_dim))
    states = torch.as_tensor(rows, dtype=torch.float64, device=CPU)
    successors = model.sample_next_states(states, torch.arange(len(PRESETS)))
    for index, preset in enumerate(PRESETS):
        expected = scalar.sample_next_state(rows[index], preset)
        assert successors[index].numpy() == pytest.approx(expected, abs=1e-6)
    rewards = model.rewards(states, torch.arange(len(PRESETS)), successors).numpy()
    for index, preset in enumerate(PRESETS):
        row = successors[index].numpy()
        assert rewards[index] == pytest.approx(scalar.reward(rows[index], preset, row), abs=1e-9)
    gripper = successors[:, 7:9].numpy()
    assert gripper == pytest.approx(rows[:, 7:9], abs=1e-6)


# ── Construction guards ─────────────────────────────────────────────────


def test_a_model_without_the_analytic_reward_is_rejected() -> None:
    """There is nothing to vectorize if the objective is not the one these kernels implement.

    Purpose: Validates the reward-model guard in the constructor

    Given: A scalar model built with a reward model that is not a ReachRewardModel
    When: The vectorized model is constructed
    Then: ValueError explains that there is nothing to vectorize

    Test type: unit
    """

    class _ConstantReward:
        def reward(self, state: Any, action: Any, next_state: Any) -> float:
            del state, action, next_state
            return 0.0

    with pytest.raises(ValueError, match="nothing to vectorize"):
        ManipulatorVectorizedModel(_scalar_model(reward_model=_ConstantReward()), device=CPU)


def test_action_presets_narrower_than_the_arm_are_rejected() -> None:
    """A short preset would be broadcast against the joints and command a different arm.

    Purpose: Validates the action-preset width guard

    Given: A scalar model whose presets are three wide for a seven-joint chain
    When: The vectorized model is constructed
    Then: ValueError names both widths

    Test type: unit
    """
    with pytest.raises(ValueError, match="but the arm has 7 commanded joints"):
        ManipulatorVectorizedModel(_scalar_model(action_presets=[np.zeros(3)]), device=CPU)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"observation_noise_std": 0.0}, "observation_noise_std"),
        ({"observation_resolution": 0.0}, "observation_resolution"),
    ],
)
def test_non_positive_observation_parameters_are_rejected(overrides: dict, message: str) -> None:
    """A zero noise std makes every particle weight -inf; a zero resolution divides by zero.

    Purpose: Validates the numeric guards in the constructor

    Given: A zero observation noise std or grid resolution
    When: The vectorized model is constructed
    Then: ValueError names the parameter

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        ManipulatorVectorizedModel(_scalar_model(), device=CPU, **overrides)
