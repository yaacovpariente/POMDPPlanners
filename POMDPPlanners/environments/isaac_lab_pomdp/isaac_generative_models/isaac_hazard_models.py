# SPDX-License-Identifier: MIT

# pylint: disable=too-many-lines  # Two full model-is-world environments plus their twins.

"""Hazard/severity variants of the Isaac navigation and reach models, model-is-world.

The base :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model.NavigationIsaacModel`
and :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model.ManipulatorIsaacModel`
are planner-side generative models with no danger in them, so no risk measure has anything to
grade. The two models here add a *latent* hazard — a disc whose type (bad/benign) or presence is a
persistent hidden property fixed at episode start — plus a signal channel that is informative only
near the hazard. The severity of a hit scales with the speed of the hit, which is what separates a
risk-averse planner (slows down near an unresolved hazard) from an expectation planner (drives
through at full speed). Both models run standalone: they carry an initial-state distribution and
are their own world, no Isaac Sim attached.

Hazards live in the robot's own frame, and the transition moves them
--------------------------------------------------------------------
The navigation state carries no absolute pose — ``pose_command`` is the goal expressed in the base
frame, and the base itself is always the origin of that frame. A hazard disc therefore cannot be
stored in world coordinates; nothing in the state could ever locate it. Instead each hazard centre
is a state block expressed in the same base frame as the goal, initialised in the frame the episode
starts in, and *updated by the transition* with exactly the SE(2) inverse step the goal gets::

    c' = R(-dyaw) @ (c - d)

where ``(d, dyaw)`` is the base's own displacement. The centres are "carried" in the sense that no
action commands them, but unlike a latent type they must ride the frame change every step — a
centre copied through unchanged would silently detach the hazard from the floor and re-attach it to
the robot. That is why ``hazard_xy`` is part of the *driven* block of
:class:`HazardRelativeTransition` while ``hazard_type`` stays outside it.

Severity is clipped at ``v_max``
--------------------------------
The contact penalty is ``-collision_penalty * (min(speed, v_max) / v_max)**2``. The clip is not in
the task brief's formula but is forced by honesty about ``reward_range``: the speed entries carry
Gaussian process noise and are therefore unbounded, so an unclipped quadratic penalty has no finite
lower bound and any declared range would be a lie. Clipping at ``v_max`` says "beyond the top
commanded speed a hit is maximally bad", keeps the declared minimum exact, and changes nothing for
the noise-free speeds the action presets actually command.

No visualizer, deliberately
---------------------------
These are 14+-dimensional factored states paired with dict observations; a particle cloud over the
full state is not a picture of anything (see the env-implementation guidance on high-dimensional
belief views). The quantity a study needs to watch — the per-hazard type posterior — is exactly
what :meth:`RelativeHazardSignalObservationModel.posterior_after_signal` returns as a number, so
the belief diagnostic is numeric rather than drawn. No golden-visualization entry exists for these
models for that reason.

Classes:
    HazardRelativeTransition: Goal-relative transition that also rides hazard centres.
    RelativeHazardSignalObservationModel: Latent-type signal for hazards stored in the base frame.
    EndEffectorPresenceSignalObservationModel: Presence signal gated on end-effector proximity.
    HazardNavigationRewardModel: Navigation objective plus speed-scaled bad-contact penalty.
    HazardNavigationIsaacModel: Navigation model with latent-type hazard discs, model-is-world.
    HazardReachRewardModel: Reach objective plus speed-scaled obstacle-contact penalty.
    HazardReachIsaacModel: Franka reach model with one latent-presence obstacle, model-is-world.
    ConstrainedHazardNavigationIsaacModel: Twin with the penalty moved to a constraint channel.
    ConstrainedHazardReachIsaacModel: Twin with the penalty moved to a constraint channel.
"""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import ConstrainedEnvironment
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction, StepInfoMetric
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model import (
    ManipulatorIsaacModel,
    ModifiedDHChain,
    ReachRewardModel,
    franka_panda_chain,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_model import (
    BASE_VELOCITY_WIDTH,
    POSE_COMMAND_WIDTH,
    VELOCITY_COMMAND_WIDTH,
    GoalRelativeTransition,
    NavigationIsaacModel,
    NavigationRewardModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model import (
    wrap_angle,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.hazard_signal_models import (  # noqa: E501  pylint: disable=line-too-long
    UNINFORMATIVE_ACCURACY,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.proprioception_models import (  # noqa: E501  pylint: disable=line-too-long
    GaussianChannelObservationModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.registry import (
    register_observation_model,
)

#: State block holding the hazard centres, ``(x, y)`` per hazard, in the current base frame.
HAZARD_XY_CHANNEL = "hazard_xy"

#: State block holding the latent per-hazard type, 1.0 = bad, 0.0 = benign.
HAZARD_TYPE_CHANNEL = "hazard_type"

#: State block holding the latent obstacle presence on the reach model, 1.0 = present.
OBSTACLE_PRESENCE_CHANNEL = "obstacle_present"

#: One-slot state block set to 1.0 once a terminal bad contact happened; ``is_terminal`` reads it.
#: A slot rather than a recomputed predicate, so process noise that later drifts a hazard centre
#: off the origin cannot un-terminate an episode that already ended.
EPISODE_DONE_CHANNEL = "episode_done"

#: Observation channel the hazard signal models produce.
HAZARD_SIGNAL_CHANNEL = "hazard_signal"

#: Default navigation action presets, ``(v_x, v_y, omega_z)`` per action: fast ahead, slow ahead,
#: arc left, arc right, stop. Two forward speeds is the risk lever — a planner that resolves the
#: hazard type can commit to the fast one.
DEFAULT_NAVIGATION_ACTION_PRESETS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.5, 0.0, 0.6),
    (0.5, 0.0, -0.6),
    (0.0, 0.0, 0.0),
)

#: Default reach action presets: uniform joint-target offsets. Under the first-order lag each
#: preset settles the arm at a distinct end-effector position, so the argmax has something to move.
DEFAULT_REACH_ACTION_PRESETS: Tuple[Tuple[float, ...], ...] = (
    (0.0,) * 7,
    (0.4,) * 7,
    (-0.4,) * 7,
    (0.2,) * 7,
    (-0.2,) * 7,
)

#: The Franka arm's default joint pose in IsaacLab's reach task, chain order, radians.
FRANKA_DEFAULT_ARM_POSE: Tuple[float, ...] = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)

#: Number of joints the Franka reach observation reports (7 arm + 2 finger).
FRANKA_OBSERVED_JOINTS = 9

#: Number of joints the Franka reach action commands.
FRANKA_COMMANDED_JOINTS = 7

#: Width of the reach task's command block: position (3) + quaternion (4).
REACH_COMMAND_WIDTH = 7

_TERMINAL_THRESHOLD = 0.5


def _coerce_action_preset(preset: Any) -> np.ndarray:
    """Accept a plain vector or the serializer's ndarray marker.

    ``Environment.from_dict`` deserializes each parameter by its annotation but does not recurse
    into a plain list, so a round-tripped ``action_presets`` list arrives as
    ``{"__type__": "ndarray", ...}`` marker dicts. Unwrapping here is what keeps
    ``to_dict`` / ``from_dict`` a true round trip.
    """
    if isinstance(preset, Mapping) and preset.get("__type__") == "ndarray":
        return np.asarray(preset["value"], dtype=float).reshape(-1)
    return np.asarray(preset, dtype=float).reshape(-1)


def _clipped_severity(speed: Union[float, np.ndarray], speed_max: float) -> Any:
    """Quadratic severity in ``[0, 1]``: ``(min(speed, speed_max) / speed_max) ** 2``.

    See the module docstring for why the clip exists — it is what keeps the declared reward
    range finite while the speed channels carry unbounded Gaussian noise.
    """
    return (np.minimum(speed, speed_max) / speed_max) ** 2


class HazardRelativeTransition(GoalRelativeTransition):
    """Goal-relative transition whose driven block also carries hazard centres.

    The driven block is ``[base_lin_vel (3), pose_command (4), hazard_xy (2 * H)]``. The first
    seven entries move exactly as in :class:`GoalRelativeTransition`; each hazard centre gets the
    same SE(2) inverse update as the goal position, ``c' = R(-dyaw) @ (c - d)``, because both are
    fixed points of the floor expressed in a frame the robot drags along. A hazard has no heading,
    so unlike the goal it contributes no wrapped entry.

    Attributes:
        num_hazards: Number of hazard centres ``H`` in the driven block.
        hazard_position_noise_std: Std of the per-entry noise on the hazard centres, in metres.
        dim: Width of the driven block, ``7 + 2 * H``.

    Example:
        >>> import numpy as np
        >>> transition = HazardRelativeTransition(
        ...     num_hazards=1, step_dt=0.2, velocity_noise_std=1e-9,
        ...     position_noise_std=1e-9, heading_noise_std=1e-9)
        >>> state = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0])
        >>> ahead = transition.sample_next_state(state, np.array([1.0, 0.0, 0.0]))
        >>> round(float(ahead[7]), 3)  # driving forward brings the hazard closer too
        1.8
    """

    #: Index of the goal heading inside the driven block — the only angular entry.
    _HEADING_INDEX = BASE_VELOCITY_WIDTH + POSE_COMMAND_WIDTH - 1

    def __init__(
        self,
        num_hazards: int,
        step_dt: float,
        linear_scale: float = 1.0,
        angular_scale: float = 1.0,
        velocity_noise_std: float = 0.1,
        position_noise_std: float = 0.05,
        heading_noise_std: float = 0.05,
        hazard_position_noise_std: Optional[float] = None,
    ) -> None:
        """Initialize the hazard-carrying goal-relative transition.

        Args:
            num_hazards: Number of hazard centres in the driven block; must be positive.
            step_dt: Control-step duration in seconds.
            linear_scale: Fraction of the commanded linear velocity actually achieved.
            angular_scale: Fraction of the commanded yaw rate actually achieved.
            velocity_noise_std: Std of the noise on the tracked base velocity, in m/s.
            position_noise_std: Std of the noise on the base-frame goal position, in metres.
            heading_noise_std: Std of the noise on the heading error, in radians.
            hazard_position_noise_std: Std of the noise on the hazard centres, in metres.
                ``None`` (the default) reuses ``position_noise_std`` — the hazard centre and the
                goal position are corrupted by the same ego-motion error, so one scale is the
                natural default. Must be strictly positive; a zero would make the transition
                density degenerate.

        Raises:
            ValueError: If ``num_hazards`` is not positive or a noise std is not strictly
                positive.
        """
        super().__init__(
            step_dt=step_dt,
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            velocity_noise_std=velocity_noise_std,
            position_noise_std=position_noise_std,
            heading_noise_std=heading_noise_std,
        )
        if num_hazards <= 0:
            raise ValueError(f"num_hazards must be positive, got {num_hazards}")
        resolved = position_noise_std if hazard_position_noise_std is None else float(
            hazard_position_noise_std
        )
        if resolved <= 0.0:
            raise ValueError(
                f"hazard_position_noise_std must be strictly positive, got {resolved}; a zero "
                "makes the transition density degenerate"
            )
        self.num_hazards = int(num_hazards)
        self.hazard_position_noise_std = float(resolved)
        self.dim = BASE_VELOCITY_WIDTH + POSE_COMMAND_WIDTH + 2 * self.num_hazards
        self._std = np.concatenate(
            [self._std, np.full(2 * self.num_hazards, self.hazard_position_noise_std)]
        )

    def mean_next_rows(self, blocks: Any, action: Any) -> np.ndarray:
        """Noise-free next driven block for ``N`` rows under one shared action, vectorized.

        The particle-filter belief update calls the batched sampler once per decision with a
        single action, so this path has no per-row Python loop.

        Args:
            blocks: ``(N, dim)`` driven blocks (or one ``(dim,)`` block).
            action: The single ``(v_x, v_y, omega_z)`` command applied to every row.

        Returns:
            The ``(N, dim)`` noise-free next blocks.
        """
        rows = np.atleast_2d(np.asarray(blocks, dtype=float))[:, : self.dim]
        command = np.asarray(action, dtype=float).reshape(-1)[:VELOCITY_COMMAND_WIDTH]
        delta = self.body_step(command)
        cos_d, sin_d = float(np.cos(delta[2])), float(np.sin(delta[2]))
        goal = rows[:, BASE_VELOCITY_WIDTH : BASE_VELOCITY_WIDTH + POSE_COMMAND_WIDTH]
        relative = goal[:, :2] - delta[:2]
        out = np.empty((rows.shape[0], self.dim), dtype=float)
        out[:, 0] = command[0] * self.linear_scale
        out[:, 1] = command[1] * self.linear_scale
        out[:, 2] = 0.0
        out[:, 3] = cos_d * relative[:, 0] + sin_d * relative[:, 1]
        out[:, 4] = -sin_d * relative[:, 0] + cos_d * relative[:, 1]
        out[:, 5] = goal[:, 2]
        out[:, 6] = wrap_angle(goal[:, 3] - delta[2])
        centers = rows[:, self._HEADING_INDEX + 1 :].reshape(rows.shape[0], self.num_hazards, 2)
        center_relative = centers - delta[:2]
        rotated_x = cos_d * center_relative[..., 0] + sin_d * center_relative[..., 1]
        rotated_y = -sin_d * center_relative[..., 0] + cos_d * center_relative[..., 1]
        out[:, self._HEADING_INDEX + 1 :] = np.stack([rotated_x, rotated_y], axis=-1).reshape(
            rows.shape[0], -1
        )
        return out

    def predict_next(self, states: Any, actions: Any) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(states, dtype=float))[:, : self.dim]
        commands = np.atleast_2d(np.asarray(actions, dtype=float))
        if commands.shape[0] == 1:
            return self.mean_next_rows(rows, commands[0])
        return np.vstack(
            [
                self.mean_next_rows(row[np.newaxis, :], command)
                for row, command in zip(rows, commands)
            ]
        )

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        # Reimplemented (not inherited) because the parent wraps the *last* entry as the heading;
        # here the last entries are hazard coordinates and the heading sits at _HEADING_INDEX.
        mean = self._mean(state, action)
        samples = mean[np.newaxis, :] + np.random.normal(0.0, self._std, size=(n_samples, self.dim))
        samples[:, self._HEADING_INDEX] = wrap_angle(samples[:, self._HEADING_INDEX])
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        mean = self._mean(state, action)
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))[:, : self.dim]
        residual = candidates - mean[np.newaxis, :]
        residual[:, self._HEADING_INDEX] = wrap_angle(residual[:, self._HEADING_INDEX])
        normalizer = float(np.sum(np.log(self._std)) + 0.5 * self.dim * np.log(2.0 * np.pi))
        return -0.5 * np.sum((residual / self._std[np.newaxis, :]) ** 2, axis=-1) - normalizer


class _BinarySignalObservationModel(IsaacObservationModel):
    """Shared math of a per-slot binary signal whose accuracy depends on the state.

    Subclasses define :meth:`occupancy` (per-slot 0/1 "is the sensor in range") and the type
    block read; everything else — the flat-outside / separating-inside likelihood — is the same
    construction :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.hazard_signal_models.LatentTypeSignalObservationModel`
    uses. Randomness is drawn from the global numpy generator rather than a held
    ``np.random.Generator`` so that ``np.random.seed`` — the seeding contract every other
    environment here follows — reproduces the signals, and so that no generator object leaks into
    ``config_id``.
    """

    supports_density = True

    def __init__(self, channel: str, type_channel: str, accuracy_inside: float) -> None:
        if not UNINFORMATIVE_ACCURACY < accuracy_inside <= 1.0:
            raise ValueError(
                "accuracy_inside must be in (0.5, 1] for the in-range signal to be informative, "
                f"got {accuracy_inside}"
            )
        self.channel = channel
        self.type_channel = type_channel
        self.accuracy_inside = float(accuracy_inside)

    def occupancy(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-slot in-range indicator at the state, shape matching the type block."""
        raise NotImplementedError

    def accuracy_at(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Per-slot signal accuracy at the state: ``accuracy_inside`` in range, 0.5 outside."""
        inside = self.occupancy(clean_state)
        return UNINFORMATIVE_ACCURACY + inside * (self.accuracy_inside - UNINFORMATIVE_ACCURACY)

    def _types(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(clean_state[self.type_channel], dtype=float).reshape(-1)

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        types = self._types(clean_state)
        truthful = np.random.random(types.size) < self.accuracy_at(clean_state)
        return np.where(truthful, types, 1.0 - types)

    def log_probability(
        self, clean_state: Mapping[str, np.ndarray], channel_observation: Any
    ) -> float:
        types = self._types(clean_state)
        signals = np.asarray(channel_observation, dtype=float).reshape(-1)
        if signals.shape != types.shape:
            return float("-inf")
        accuracy = self.accuracy_at(clean_state)
        agrees = np.isclose(types, signals)
        return float(np.log(np.where(agrees, accuracy, 1.0 - accuracy)).sum())

    def posterior_after_signal(
        self,
        prior: ArrayLike,
        clean_state: Mapping[str, np.ndarray],
        signals: ArrayLike,
    ) -> np.ndarray:
        """Bayes-update the per-slot high-type probability on one signal, for diagnostics."""
        accuracy = self.accuracy_at(clean_state)
        signal = np.asarray(signals, dtype=float).reshape(-1)
        prior_high = np.asarray(prior, dtype=float).reshape(-1)
        likelihood_high = np.where(signal > 0.5, accuracy, 1.0 - accuracy)
        likelihood_low = np.where(signal > 0.5, 1.0 - accuracy, accuracy)
        joint_high = prior_high * likelihood_high
        evidence = joint_high + (1.0 - prior_high) * likelihood_low
        return np.divide(joint_high, evidence, out=prior_high.copy(), where=evidence > 0.0)


@register_observation_model("relative_hazard_signal")
class RelativeHazardSignalObservationModel(_BinarySignalObservationModel):
    """Latent-type signal for hazards whose centres are state blocks in the base frame.

    The sibling
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.hazard_signal_models.LatentTypeSignalObservationModel`
    holds fixed world-frame zone geometry and reads the robot position from the state. Here the
    geometry is inverted: the robot is the origin of its own frame and the hazard centres are the
    moving state blocks, so being "in range" of hazard *j* means its centre lies within
    ``signal_radius`` of the origin.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The type block and the centres block, in that order.
        signal_radius: Range within which the signal reports the true type, in metres.
        accuracy_inside: Probability the in-range signal reports the true type.

    Example:
        >>> import numpy as np
        >>> model = RelativeHazardSignalObservationModel(signal_radius=1.0, accuracy_inside=0.9)
        >>> near = {"hazard_type": np.array([1.0]), "hazard_xy": np.array([0.5, 0.0])}
        >>> far = {"hazard_type": np.array([1.0]), "hazard_xy": np.array([3.0, 0.0])}
        >>> model.accuracy_at(near).tolist(), model.accuracy_at(far).tolist()
        ([0.9], [0.5])
    """

    def __init__(
        self,
        channel: str = HAZARD_SIGNAL_CHANNEL,
        type_channel: str = HAZARD_TYPE_CHANNEL,
        centers_channel: str = HAZARD_XY_CHANNEL,
        signal_radius: float = 1.5,
        accuracy_inside: float = 0.9,
    ) -> None:
        """Initialize the relative hazard signal model.

        Args:
            channel: The observation-dict key this model produces.
            type_channel: State block holding the per-hazard latent types (entries in {0, 1}).
            centers_channel: State block holding the hazard centres, ``(x, y)`` per hazard.
            signal_radius: Range within which the signal is informative, in metres.
            accuracy_inside: In-range accuracy; must be in ``(0.5, 1]``.

        Raises:
            ValueError: If ``signal_radius`` is not positive or ``accuracy_inside`` is out of
                range.
        """
        super().__init__(channel=channel, type_channel=type_channel, accuracy_inside=accuracy_inside)
        if signal_radius <= 0.0:
            raise ValueError(f"signal_radius must be positive, got {signal_radius}")
        self.centers_channel = centers_channel
        self.state_channels = (type_channel, centers_channel)
        self.signal_radius = float(signal_radius)

    def occupancy(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        centers = np.asarray(clean_state[self.centers_channel], dtype=float).reshape(-1, 2)
        distance = np.linalg.norm(centers, axis=-1)
        return (distance <= self.signal_radius).astype(float)


@register_observation_model("end_effector_presence_signal")
class EndEffectorPresenceSignalObservationModel(_BinarySignalObservationModel):
    """Latent-presence signal informative only while the end-effector is near the obstacle.

    The reach model's obstacle is fixed in the workspace, so "in range" is a function of the
    joint state alone: the hand's analytic position (through the same
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_model.ModifiedDHChain`
    the reward uses) within ``signal_radius`` of the obstacle centre. The signal is one bit wide —
    presence, not type.

    Attributes:
        channel: The observation-dict key this model produces.
        state_channels: The presence block and the joint-position block, in that order.
        obstacle_center: The obstacle centre in the robot base frame, shape ``(3,)``.
        signal_radius: End-effector range within which the signal is informative, in metres.
        accuracy_inside: Probability the in-range signal reports the true presence.
    """

    def __init__(
        self,
        chain: ModifiedDHChain,
        default_joint_positions: ArrayLike,
        obstacle_center: ArrayLike,
        channel: str = HAZARD_SIGNAL_CHANNEL,
        presence_channel: str = OBSTACLE_PRESENCE_CHANNEL,
        joint_position_channel: str = "joint_pos",
        arm_joint_indices: Optional[ArrayLike] = None,
        signal_radius: float = 0.25,
        accuracy_inside: float = 0.9,
    ) -> None:
        """Initialize the end-effector presence signal model.

        Args:
            chain: Kinematics used to place the hand.
            default_joint_positions: The chain's default joint pose, added to the relative joint
                positions to recover absolute angles.
            obstacle_center: The obstacle centre in the robot base frame, ``(x, y, z)``.
            channel: The observation-dict key this model produces.
            presence_channel: State block holding the latent presence bit.
            joint_position_channel: State block holding relative joint positions.
            arm_joint_indices: Which entries of the joint block feed the chain, in the chain's
                order. ``None`` means the leading ``chain.num_joints`` entries, in order.
            signal_radius: End-effector range within which the signal is informative, in metres.
            accuracy_inside: In-range accuracy; must be in ``(0.5, 1]``.

        Raises:
            ValueError: If ``signal_radius`` is not positive, the default pose is not the chain's
                width, or ``accuracy_inside`` is out of range.
        """
        super().__init__(
            channel=channel, type_channel=presence_channel, accuracy_inside=accuracy_inside
        )
        if signal_radius <= 0.0:
            raise ValueError(f"signal_radius must be positive, got {signal_radius}")
        self.chain = chain
        self.default_joint_positions = np.asarray(default_joint_positions, dtype=float).reshape(-1)
        if self.default_joint_positions.shape[0] != chain.num_joints:
            raise ValueError(
                f"default_joint_positions must have {chain.num_joints} entries, got "
                f"{self.default_joint_positions.shape[0]}"
            )
        self.obstacle_center = np.asarray(obstacle_center, dtype=float).reshape(3)
        self.presence_channel = presence_channel
        self.joint_position_channel = joint_position_channel
        self.arm_joint_indices = (
            np.arange(chain.num_joints)
            if arm_joint_indices is None
            else np.asarray(arm_joint_indices, dtype=int).reshape(-1)
        )
        self.state_channels = (presence_channel, joint_position_channel)
        self.signal_radius = float(signal_radius)

    def end_effector_position(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        """Analytic hand position at the joint state in ``clean_state``, shape ``(3,)``."""
        relative = np.asarray(clean_state[self.joint_position_channel], dtype=float).reshape(-1)
        angles = relative[self.arm_joint_indices] + self.default_joint_positions
        return self.chain.end_effector_position(angles)

    def occupancy(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        distance = float(
            np.linalg.norm(self.end_effector_position(clean_state) - self.obstacle_center)
        )
        return np.array([1.0 if distance <= self.signal_radius else 0.0])


class HazardNavigationRewardModel(NavigationRewardModel):
    """Navigation objective plus a speed-scaled penalty for contact with a bad hazard.

    Contact is a *state* predicate: the robot is the origin of its own frame, so it touches hazard
    *j* exactly when that hazard's centre lies within ``radius_j`` of the origin. The penalty
    ``-collision_penalty * (min(speed, v_max) / v_max) ** 2`` reads the planar speed off the
    ``base_lin_vel`` block of the same state — a slow brush costs little, a full-speed hit costs
    ``collision_penalty`` (see the module docstring for the clip).

    Attributes:
        hazard_radii: Per-hazard contact radii, shape ``(H,)``.
        collision_penalty: Magnitude of the bad-contact penalty at full speed; non-negative.
        v_max: Speed at which the severity saturates, in m/s.
        hazard_centers_channel: State block holding the hazard centres.
        hazard_type_channel: State block holding the latent types.
        base_velocity_channel: State block holding the body-frame base velocity.
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        hazard_radii: ArrayLike,
        collision_penalty: float,
        v_max: float,
        hazard_centers_channel: str = HAZARD_XY_CHANNEL,
        hazard_type_channel: str = HAZARD_TYPE_CHANNEL,
        base_velocity_channel: str = "base_lin_vel",
        command_channel: str = "pose_command",
        coarse_weight: float = 0.5,
        coarse_std: float = 2.0,
        fine_weight: float = 0.5,
        fine_std: float = 0.2,
        heading_weight: float = 0.2,
    ) -> None:
        """Initialize the hazard navigation reward.

        Args:
            state_schema: Named blocks of the flat state vector.
            hazard_radii: Contact radius per hazard, in metres; all strictly positive.
            collision_penalty: Penalty magnitude at saturated speed; must be non-negative.
            v_max: Severity saturation speed, in m/s; must be positive.
            hazard_centers_channel: State block holding the hazard centres.
            hazard_type_channel: State block holding the latent types.
            base_velocity_channel: State block holding the body-frame base velocity.
            command_channel: State block holding the base-frame pose command.
            coarse_weight: Weight on the wide ``tanh`` position term.
            coarse_std: Length scale of the wide term, in metres.
            fine_weight: Weight on the narrow ``tanh`` position term.
            fine_std: Length scale of the narrow term, in metres.
            heading_weight: Weight on ``|heading_error|``.

        Raises:
            ValueError: If a radius is not positive, ``collision_penalty`` is negative, ``v_max``
                is not positive, or the hazard blocks disagree in count.
        """
        super().__init__(
            state_schema=state_schema,
            command_channel=command_channel,
            coarse_weight=coarse_weight,
            coarse_std=coarse_std,
            fine_weight=fine_weight,
            fine_std=fine_std,
            heading_weight=heading_weight,
        )
        radii = np.asarray(hazard_radii, dtype=float).reshape(-1)
        if np.any(radii <= 0.0):
            raise ValueError(f"hazard radii must be strictly positive, got {radii.tolist()}")
        if collision_penalty < 0.0:
            raise ValueError(f"collision_penalty must be non-negative, got {collision_penalty}")
        if v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {v_max}")
        if state_schema.width(hazard_centers_channel) != 2 * radii.shape[0]:
            raise ValueError(
                f"{hazard_centers_channel!r} is {state_schema.width(hazard_centers_channel)} wide "
                f"but {radii.shape[0]} hazard radii were given"
            )
        if state_schema.width(hazard_type_channel) != radii.shape[0]:
            raise ValueError(
                f"{hazard_type_channel!r} is {state_schema.width(hazard_type_channel)} wide "
                f"but {radii.shape[0]} hazard radii were given"
            )
        self.hazard_radii = radii
        self.collision_penalty = float(collision_penalty)
        self.v_max = float(v_max)
        self.hazard_centers_channel = hazard_centers_channel
        self.hazard_type_channel = hazard_type_channel
        self.base_velocity_channel = base_velocity_channel

    def planar_speed(self, state: Any) -> float:
        """Planar base speed read off the velocity block, in m/s."""
        velocity = self.state_schema.block(state, self.base_velocity_channel).reshape(-1)
        return float(np.linalg.norm(velocity[:2]))

    def contact_indicators(self, state: Any) -> np.ndarray:
        """Per-hazard contact indicator at the state, shape ``(H,)``."""
        centers = self.state_schema.block(state, self.hazard_centers_channel).reshape(-1, 2)
        return (np.linalg.norm(centers, axis=-1) <= self.hazard_radii).astype(float)

    def bad_contact(self, state: Any) -> float:
        """1.0 when the robot touches a hazard whose latent type is bad, else 0.0."""
        types = self.state_schema.block(state, self.hazard_type_channel).reshape(-1)
        return float(np.any((self.contact_indicators(state) > 0.5) & (types > 0.5)))

    def collision_term(self, state: Any) -> float:
        """The speed-scaled bad-contact penalty at the state (0.0 when not in bad contact)."""
        if self.bad_contact(state) == 0.0:
            return 0.0
        return -self.collision_penalty * float(
            _clipped_severity(self.planar_speed(state), self.v_max)
        )

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        resulting = state if next_state is None else next_state
        return super().reward(state, action, next_state) + self.collision_term(resulting)

    def reward_rows(self, rows: np.ndarray) -> np.ndarray:
        """Vectorized reward over ``(N, total_dim)`` resulting states; matches :meth:`reward`."""
        goal = rows[:, self.state_schema.slice_of(self.command_channel)]
        distance = np.linalg.norm(goal[:, :3], axis=-1)
        heading = np.abs(wrap_angle(goal[:, 3]))
        nav = (
            self.coarse_weight * (1.0 - np.tanh(distance / self.coarse_std))
            + self.fine_weight * (1.0 - np.tanh(distance / self.fine_std))
            - self.heading_weight * heading
        )
        centers = rows[:, self.state_schema.slice_of(self.hazard_centers_channel)].reshape(
            rows.shape[0], -1, 2
        )
        types = rows[:, self.state_schema.slice_of(self.hazard_type_channel)]
        contact = np.linalg.norm(centers, axis=-1) <= self.hazard_radii[np.newaxis, :]
        bad = np.any(contact & (types > 0.5), axis=-1)
        velocity = rows[:, self.state_schema.slice_of(self.base_velocity_channel)]
        speed = np.linalg.norm(velocity[:, :2], axis=-1)
        penalty = -self.collision_penalty * _clipped_severity(speed, self.v_max)
        return nav + np.where(bad, penalty, 0.0)


class HazardNavigationIsaacModel(NavigationIsaacModel):
    """Goal-relative navigation with latent-type hazard discs, run model-is-world.

    The state extends the navigation task's ``base_lin_vel + projected_gravity + pose_command``
    with three blocks: the hazard centres (driven — see :class:`HazardRelativeTransition`), the
    per-hazard latent types (carried, sampled Bernoulli(``p_bad``) once per episode by
    :meth:`initial_state_dist`), and the one-slot terminal flag. The observation keeps the
    parent's Gaussian channels and adds the :class:`RelativeHazardSignalObservationModel` bit per
    hazard, informative only within ``signal_radius`` — so the type has to be *learned* by going
    near, which is what gives a risk-averse planner something to be averse about.

    The task: reach the commanded goal (planar distance below ``success_radius``) without hitting
    a bad hazard. A bad contact sets the terminal slot when ``is_bad_contact_terminal`` and always
    costs ``collision_penalty`` scaled by the squared (clipped) speed ratio.

    Attributes:
        hazards: The ``(x, y, radius)`` triples, in the initial base frame.
        num_hazards: Number of hazard discs.
        hazard_reward: The :class:`HazardNavigationRewardModel` scoring the task.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.4)], p_bad=1.0)
        >>> state = model.initial_state_dist().sample(1)[0]
        >>> float(model.state_schema.block(state, "hazard_type")[0])  # bad for sure at p_bad=1
        1.0
        >>> model.is_terminal(state)
        False
    """

    def __init__(
        self,
        hazards: Sequence[Sequence[float]],
        discount_factor: float = 0.99,
        step_dt: float = 0.2,
        action_presets: Optional[Sequence[ArrayLike]] = None,
        initial_goal: Sequence[float] = (4.0, 0.0, 0.0, 0.0),
        p_bad: float = 0.5,
        collision_penalty: float = 50.0,
        v_max: float = 1.0,
        is_bad_contact_terminal: bool = True,
        success_radius: float = 0.5,
        signal_radius: float = 1.5,
        signal_accuracy: float = 0.9,
        linear_scale: float = 1.0,
        angular_scale: float = 1.0,
        velocity_noise_std: float = 0.1,
        position_noise_std: float = 0.05,
        heading_noise_std: float = 0.05,
        hazard_position_noise_std: Optional[float] = None,
        observation_noise_std: float = 0.1,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the hazard navigation model.

        Args:
            hazards: ``(x, y, radius)`` per hazard, expressed in the initial base frame (the
                frame the episode starts in); at least one, radii strictly positive.
            discount_factor: POMDP discount factor.
            step_dt: Control-step duration in seconds.
            action_presets: ``(v_x, v_y, omega_z)`` commands to plan over. ``None`` (the
                default) uses :data:`DEFAULT_NAVIGATION_ACTION_PRESETS`.
            initial_goal: The initial ``pose_command`` block ``(x, y, z, heading)``.
            p_bad: Probability each hazard's latent type is bad, drawn once per episode.
            collision_penalty: Bad-contact penalty magnitude at saturated speed; non-negative.
            v_max: Severity saturation speed, in m/s.
            is_bad_contact_terminal: Whether a bad contact sets the terminal slot. With
                ``False`` the episode continues and every step spent in bad contact pays the
                penalty again.
            success_radius: Planar goal distance below which the episode succeeds, in metres.
            signal_radius: Range within which the hazard signal is informative, in metres.
            signal_accuracy: In-range signal accuracy, in ``(0.5, 1]``.
            linear_scale: Fraction of the commanded linear velocity actually achieved.
            angular_scale: Fraction of the commanded yaw rate actually achieved.
            velocity_noise_std: Std of the base-velocity process noise, in m/s.
            position_noise_std: Std of the goal-position process noise, in metres.
            heading_noise_std: Std of the heading-error process noise, in radians.
            hazard_position_noise_std: Std of the hazard-centre process noise, in metres;
                ``None`` reuses ``position_noise_std``.
            observation_noise_std: Std of the Gaussian noise on the observed velocity and
                pose-command channels.
            name: Model name; defaults to the class name.

        Raises:
            ValueError: If ``hazards`` is empty or malformed, ``p_bad`` is outside ``[0, 1]``,
                ``initial_goal`` is not 4 entries, or a scalar parameter is out of range.
        """
        normalized = tuple(
            (float(entry[0]), float(entry[1]), float(entry[2])) for entry in hazards
        )
        if not normalized:
            raise ValueError(
                "hazards must name at least one (x, y, radius) disc; for a hazard-free task use "
                "NavigationIsaacModel"
            )
        if not 0.0 <= p_bad <= 1.0:
            raise ValueError(f"p_bad must be in [0, 1], got {p_bad}")
        if success_radius <= 0.0:
            raise ValueError(f"success_radius must be positive, got {success_radius}")
        goal = tuple(float(value) for value in initial_goal)
        if len(goal) != POSE_COMMAND_WIDTH:
            raise ValueError(
                f"initial_goal must have {POSE_COMMAND_WIDTH} entries (x, y, z, heading), "
                f"got {len(goal)}"
            )
        num_hazards = len(normalized)
        schema = IsaacChannelSchema(
            (
                ("base_lin_vel", BASE_VELOCITY_WIDTH),
                ("projected_gravity", 3),
                ("pose_command", POSE_COMMAND_WIDTH),
                (HAZARD_XY_CHANNEL, 2 * num_hazards),
                (HAZARD_TYPE_CHANNEL, num_hazards),
                (EPISODE_DONE_CHANNEL, 1),
            )
        )
        reward_model = HazardNavigationRewardModel(
            state_schema=schema,
            hazard_radii=[entry[2] for entry in normalized],
            collision_penalty=collision_penalty,
            v_max=v_max,
        )
        resolved_presets = (
            DEFAULT_NAVIGATION_ACTION_PRESETS if action_presets is None else action_presets
        )
        observation_models: Dict[str, IsaacObservationModel] = {
            "base_lin_vel": GaussianChannelObservationModel(
                channel="base_lin_vel", noise_std=observation_noise_std
            ),
            "pose_command": GaussianChannelObservationModel(
                channel="pose_command", noise_std=observation_noise_std
            ),
            HAZARD_SIGNAL_CHANNEL: RelativeHazardSignalObservationModel(
                signal_radius=signal_radius, accuracy_inside=signal_accuracy
            ),
        }
        super().__init__(
            state_schema=schema,
            action_presets=[_coerce_action_preset(preset) for preset in resolved_presets],
            discount_factor=discount_factor,
            step_dt=step_dt,
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            velocity_noise_std=velocity_noise_std,
            position_noise_std=position_noise_std,
            heading_noise_std=heading_noise_std,
            reward_model=reward_model,
            observation_models=observation_models,
            reward_range=self._hazard_reward_range(reward_model, p_bad),
            name=name if name is not None else type(self).__name__,
        )
        # The parent wired a GoalRelativeTransition over (velocity, command). Swap in the
        # hazard-carrying transition and widen the driven block so the hazard centres ride the
        # same SE(2) inverse update as the goal — a centre left "carried" would stay glued to
        # the robot instead of to the floor.
        self.goal_transition = HazardRelativeTransition(
            num_hazards=num_hazards,
            step_dt=step_dt,
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            velocity_noise_std=velocity_noise_std,
            position_noise_std=position_noise_std,
            heading_noise_std=heading_noise_std,
            hazard_position_noise_std=hazard_position_noise_std,
        )
        self._transition = self.goal_transition
        self.transition_channels = ("base_lin_vel", "pose_command", HAZARD_XY_CHANNEL)
        self._driven_indices = self._resolve_driven_indices()
        self._carried_indices = self._resolve_carried_indices()

        self.hazards = normalized
        self.num_hazards = num_hazards
        self.hazard_reward = reward_model
        self.initial_goal = goal
        self.p_bad = float(p_bad)
        self.collision_penalty = float(collision_penalty)
        self.v_max = float(v_max)
        self.is_bad_contact_terminal = bool(is_bad_contact_terminal)
        self.success_radius = float(success_radius)
        self.signal_radius = float(signal_radius)
        self.signal_accuracy = float(signal_accuracy)
        self.hazard_position_noise_std = self.goal_transition.hazard_position_noise_std
        self.observation_noise_std = float(observation_noise_std)
        self.step_dt = float(step_dt)
        self.linear_scale = float(linear_scale)
        self.angular_scale = float(angular_scale)
        self.velocity_noise_std = float(velocity_noise_std)
        self.position_noise_std = float(position_noise_std)
        self.heading_noise_std = float(heading_noise_std)

        self._type_slice = self.state_schema.slice_of(HAZARD_TYPE_CHANNEL)
        self._done_index = self.state_schema.slice_of(EPISODE_DONE_CHANNEL).start
        self._warm_observation_caches()

    @staticmethod
    def _hazard_reward_range(
        reward_model: HazardNavigationRewardModel, p_bad: float
    ) -> Tuple[float, float]:
        """Exact per-step reward bounds, enumerated over every term and flag.

        * Both ``tanh`` tracking terms lie in ``(0, weight]``: the maximum ``weight`` is attained
          at zero distance, the infimum 0 as the distance grows (0 is a valid lower bound even
          though unattained).
        * The heading term lies in ``[-heading_weight * pi, 0]`` exactly, because residuals are
          wrapped into ``(-pi, pi]``.
        * The contact penalty lies in ``[-collision_penalty, 0]`` because the severity ratio is
          clipped at 1 (see the module docstring); it can coincide with the worst heading error,
          so the bounds add. It is excluded only when it cannot fire at all: a zero penalty, or
          ``p_bad == 0`` so no bad hazard ever exists (the type is drawn once at reset and the
          transition carries it as a point mass).
        * ``is_bad_contact_terminal`` changes how often the penalty can be paid, not its per-step
          bound, so it does not appear here.
        """
        maximum = reward_model.coarse_weight + reward_model.fine_weight
        minimum = -reward_model.heading_weight * float(np.pi)
        if reward_model.collision_penalty > 0.0 and p_bad > 0.0:
            minimum -= reward_model.collision_penalty
        return (minimum, maximum)

    def _warm_observation_caches(self) -> None:
        # Environment.config_id serializes public attributes recursively, and the Gaussian
        # channel models memoize a per-width normal on first use — so a used model would hash
        # differently from a fresh one. Exercising each density once here (deterministic; no RNG
        # is consumed by log_probability) fills those caches before config_id can ever be read.
        zero = self.state_schema.split(np.zeros(self.state_schema.total_dim))
        assert self.observation_models is not None
        self.observation_models["base_lin_vel"].log_probability(
            zero, np.zeros(BASE_VELOCITY_WIDTH)
        )
        self.observation_models["pose_command"].log_probability(
            zero, np.zeros(POSE_COMMAND_WIDTH)
        )

    # ── Terminal bookkeeping ────────────────────────────────────────────
    def _apply_done(self, rows: np.ndarray) -> None:
        """Set the terminal slot on rows that are in bad contact, sticky, in place."""
        if not self.is_bad_contact_terminal:
            return
        centers = rows[:, self.state_schema.slice_of(HAZARD_XY_CHANNEL)].reshape(
            rows.shape[0], self.num_hazards, 2
        )
        types = rows[:, self._type_slice]
        contact = np.linalg.norm(centers, axis=-1) <= self.hazard_reward.hazard_radii
        bad = np.any(contact & (types > 0.5), axis=-1).astype(float)
        rows[:, self._done_index] = np.maximum(rows[:, self._done_index], bad)

    def _expected_done(self, state: Any, candidate_rows: np.ndarray) -> np.ndarray:
        """The terminal-slot value each candidate row must carry, given the source state."""
        previous = float(np.asarray(state, dtype=float).reshape(-1)[self._done_index])
        if not self.is_bad_contact_terminal:
            return np.full(candidate_rows.shape[0], previous)
        expected = candidate_rows.copy()
        expected[:, self._done_index] = previous
        self._apply_done(expected)
        return expected[:, self._done_index]

    # ── Dynamics ────────────────────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        result = super().sample_next_state(state, action, n_samples)
        self._apply_done(np.atleast_2d(result))  # atleast_2d is a view; edits land in result
        return result

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        """One next state per particle under one action, with no per-particle Python loop.

        The particle-filter belief update calls this once per real step, and PFT-DPW once per
        belief-node expansion, so it is the hot path the vectorization exists for.
        """
        rows = np.atleast_2d(np.asarray(states, dtype=float))
        driven = rows[:, self._driven_indices]
        means = self.goal_transition.mean_next_rows(driven, action)
        noise_std = self.goal_transition.process_noise_std
        samples = means + np.random.normal(0.0, noise_std, size=means.shape)
        heading_column = HazardRelativeTransition._HEADING_INDEX  # pylint: disable=protected-access
        samples[:, heading_column] = wrap_angle(samples[:, heading_column])
        out = rows.copy()
        out[:, self._driven_indices] = samples
        self._apply_done(out)
        return out

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Density with the terminal slot scored by its deterministic rule, not as a copy.

        The factored base scores every carried block as a point mass, which is right for the
        latent types and the gravity block but would call a legitimate 0-to-1 flip of the
        terminal slot impossible. Here the driven block is scored by the transition, the types
        and ``projected_gravity`` as point masses, and the slot against the value the contact
        rule deterministically implies for each candidate.
        """
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))
        source = np.asarray(state, dtype=float).reshape(-1)
        driven = np.asarray(
            self.goal_transition.log_probability(
                source[self._driven_indices], action, candidates[:, self._driven_indices]
            ),
            dtype=float,
        ).reshape(-1)
        types_match = np.all(
            np.isclose(candidates[:, self._type_slice], source[self._type_slice]), axis=-1
        )
        gravity_slice = self.state_schema.slice_of("projected_gravity")
        gravity_match = np.all(
            np.isclose(candidates[:, gravity_slice], source[gravity_slice]), axis=-1
        )
        done_match = np.isclose(
            candidates[:, self._done_index], self._expected_done(source, candidates)
        )
        return np.where(types_match & gravity_match & done_match, driven, -np.inf)

    # ── Reward ──────────────────────────────────────────────────────────
    @property
    def reward_requires_next_state(self) -> bool:
        """True: the contact penalty and terminal slot live on the realised successor.

        With the default (state, action) ordering the terminal state is never scored, so a
        terminal bad contact would never be charged at all.
        """
        return True

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        del action  # the objective scores states, not transitions
        resulting = states if next_states is None else next_states
        return self.hazard_reward.reward_rows(
            np.atleast_2d(np.asarray(resulting, dtype=float))
        )

    # ── Termination / episode structure ─────────────────────────────────
    def is_terminal(self, state: Any) -> bool:
        vector = np.asarray(state, dtype=float).reshape(-1)
        if vector[self._done_index] > _TERMINAL_THRESHOLD:
            return True
        return self.hazard_reward.planar_goal_distance(vector) < self.success_radius

    def initial_state_dist(self) -> Distribution:
        template = self.state_schema.pack(
            {
                "base_lin_vel": np.zeros(BASE_VELOCITY_WIDTH),
                "projected_gravity": np.array([0.0, 0.0, -1.0]),
                "pose_command": np.asarray(self.initial_goal, dtype=float),
                HAZARD_XY_CHANNEL: np.asarray(
                    [entry[:2] for entry in self.hazards], dtype=float
                ).reshape(-1),
                HAZARD_TYPE_CHANNEL: np.zeros(self.num_hazards),
                EPISODE_DONE_CHANNEL: np.zeros(1),
            }
        )
        return _TemplateWithBernoulliSlots(
            template=template, bernoulli_slice=self._type_slice, probability=self.p_bad
        )

    def initial_observation_dist(self) -> Distribution:
        return _ObservationOfInitialState(self)

    # ── Metrics ─────────────────────────────────────────────────────────
    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Per-step measurement channels, all pure functions of ``state``.

        Every channel is state-derived, so the terminal bookkeeping call (``action`` and
        ``next_state`` both ``None``) reports the final state like any other — which is exactly
        the state where a terminal contact or the goal arrival lives. No randomness is consumed.
        """
        del action, next_state
        vector = np.asarray(state, dtype=float).reshape(-1)
        contact = float(np.any(self.hazard_reward.contact_indicators(vector) > 0.5))
        bad = self.hazard_reward.bad_contact(vector)
        speed = self.hazard_reward.planar_speed(vector)
        centers = vector[self.state_schema.slice_of(HAZARD_XY_CHANNEL)].reshape(-1, 2)
        in_zone = float(np.any(np.linalg.norm(centers, axis=-1) <= self.signal_radius))
        goal = float(
            self.hazard_reward.planar_goal_distance(vector) < self.success_radius
        )
        failure = float(vector[self._done_index] > _TERMINAL_THRESHOLD)
        return {
            "recorded_step": 1.0,
            "hazard_contact": contact,
            "bad_hazard_contact": bad,
            "speed_at_contact_mps": speed if contact > 0.5 else 0.0,
            "signal_zone": in_zone,
            "planar_speed_mps": speed,
            "goal_reached": goal,
            "ended_by_goal": goal,
            "ended_by_failure": failure,
            "ended_by_timeout": 1.0 - max(goal, failure),
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        return _hazard_metric_specs(
            speed_channel="planar_speed_mps",
            speed_metric="mean_planar_speed_mps",
            contact_channel="hazard_contact",
            contact_rate="hazard_contact_rate",
            contact_steps="average_hazard_contact_steps",
            bad_channel="bad_hazard_contact",
            bad_rate="bad_hazard_contact_rate",
        )


class _TemplateWithBernoulliSlots(Distribution):
    """Deterministic initial state with one block of independent Bernoulli latent bits.

    Everything the episode starts with is task configuration — the goal, the hazard geometry —
    except the latent bits, which are the one quantity a belief must stay uncertain about. Draws
    come from the global numpy generator so ``np.random.seed`` reproduces episodes.
    """

    def __init__(self, template: np.ndarray, bernoulli_slice: slice, probability: float) -> None:
        self._template = np.asarray(template, dtype=float).reshape(-1)
        self._slice = bernoulli_slice
        self._probability = float(probability)

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        rows = np.tile(self._template, (n_samples, 1))
        width = self._slice.stop - self._slice.start
        rows[:, self._slice] = (
            np.random.random((n_samples, width)) < self._probability
        ).astype(float)
        return [rows[index] for index in range(n_samples)]


class _ObservationOfInitialState(Distribution):
    """Initial observations: perceive freshly drawn initial states through the model's channels."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def sample(self, n_samples: int = 1) -> List[Any]:
        states = self._model.initial_state_dist().sample(n_samples)
        return [self._model.sample_observation(state, None) for state in states]


def _hazard_metric_specs(
    speed_channel: str,
    speed_metric: str,
    contact_channel: str,
    contact_rate: str,
    contact_steps: str,
    bad_channel: str,
    bad_rate: str,
) -> List[StepInfoMetric]:
    """The metric layout the two hazard models share, differing only in channel names/units."""
    return [
        # ANY, not ALL: the task is reach-a-goal — it succeeded if it ever happened. Kept as
        # goal_reaching_rate (not task_completion_rate) per the study brief, so the tuning and
        # comparison configs address one name across both hazard models.
        StepInfoMetric(
            name="goal_reaching_rate",
            channel="goal_reached",
            per_episode=EpisodeReduction.ANY,
        ),
        StepInfoMetric(name=bad_rate, channel=bad_channel, per_episode=EpisodeReduction.ANY),
        StepInfoMetric(
            name=contact_rate, channel=contact_channel, per_episode=EpisodeReduction.ANY
        ),
        StepInfoMetric(
            name=contact_steps, channel=contact_channel, per_episode=EpisodeReduction.SUM
        ),
        # MAX, not MEAN: severity — the worst hit characterizes the episode; contact-free
        # episodes contribute 0, so read this beside the contact rate.
        StepInfoMetric(
            name="max_speed_at_contact_mps",
            channel="speed_at_contact_mps",
            per_episode=EpisodeReduction.MAX,
        ),
        StepInfoMetric(
            name="average_signal_zone_steps",
            channel="signal_zone",
            per_episode=EpisodeReduction.SUM,
        ),
        StepInfoMetric(name=speed_metric, channel=speed_channel, per_episode=EpisodeReduction.MEAN),
        StepInfoMetric(
            name="average_episode_length",
            channel="recorded_step",
            per_episode=EpisodeReduction.SUM,
        ),
        # LAST: why the episode ended is a property of its final recorded state; across episodes
        # these become rates that say whether failures are crashes or timeouts.
        StepInfoMetric(
            name="ended_by_goal_rate", channel="ended_by_goal", per_episode=EpisodeReduction.LAST
        ),
        StepInfoMetric(
            name="ended_by_failure_rate",
            channel="ended_by_failure",
            per_episode=EpisodeReduction.LAST,
        ),
        StepInfoMetric(
            name="ended_by_timeout_rate",
            channel="ended_by_timeout",
            per_episode=EpisodeReduction.LAST,
        ),
    ]


class HazardReachRewardModel(ReachRewardModel):
    """Reach objective plus a speed-scaled penalty for touching the obstacle while present.

    Contact is again a state predicate — the analytic hand position within ``obstacle_radius`` of
    the fixed obstacle centre while the latent presence bit is set. The severity speed is the
    hand's own displacement rate ``|FK(q') - FK(q)| / step_dt``, computed where both states are
    available; when only one state is (the ``next_state is None`` fallback) the speed is taken as
    zero and a contact costs nothing, which understates rather than invents a penalty.

    Attributes:
        obstacle_center: Obstacle centre in the robot base frame, shape ``(3,)``.
        obstacle_radius: Contact radius, in metres.
        presence_channel: State block holding the latent presence bit.
        collision_penalty: Penalty magnitude at saturated hand speed; non-negative.
        ee_speed_max: Hand speed at which the severity saturates, in m/s.
        step_dt: Control-step duration the speed difference quotient divides by.
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        chain: ModifiedDHChain,
        joint_position_channel: str,
        command_channel: str,
        default_joint_positions: ArrayLike,
        obstacle_center: ArrayLike,
        obstacle_radius: float,
        collision_penalty: float,
        ee_speed_max: float,
        step_dt: float,
        presence_channel: str = OBSTACLE_PRESENCE_CHANNEL,
        arm_joint_indices: Optional[ArrayLike] = None,
        command_position_width: int = 3,
        distance_weight: float = -0.2,
        shaping_weight: float = 0.1,
        shaping_std: float = 0.1,
    ) -> None:
        """Initialize the hazard reach reward.

        Args:
            state_schema: Named blocks of the flat state vector.
            chain: Kinematics from the robot base to the tracked body.
            joint_position_channel: Block holding relative joint positions.
            command_channel: Block holding the commanded pose.
            default_joint_positions: The chain's default joint pose, chain order.
            obstacle_center: Obstacle centre in the robot base frame, ``(x, y, z)``.
            obstacle_radius: Contact radius, in metres; strictly positive.
            collision_penalty: Penalty magnitude at saturated hand speed; non-negative.
            ee_speed_max: Severity saturation hand speed, in m/s; strictly positive.
            step_dt: Control-step duration in seconds; strictly positive.
            presence_channel: Block holding the latent presence bit (width 1).
            arm_joint_indices: Which entries of the joint block feed the chain.
            command_position_width: Leading command entries holding the position.
            distance_weight: Weight on the raw distance term; must be non-positive (the declared
                reward range is derived under that sign).
            shaping_weight: Weight on the ``tanh`` closeness bonus; must be non-negative.
            shaping_std: Length scale of the closeness bonus, in metres.

        Raises:
            ValueError: If a scalar parameter is out of range or the presence block is not one
                slot wide.
        """
        super().__init__(
            state_schema=state_schema,
            chain=chain,
            joint_position_channel=joint_position_channel,
            command_channel=command_channel,
            default_joint_positions=default_joint_positions,
            arm_joint_indices=arm_joint_indices,
            command_position_width=command_position_width,
            distance_weight=distance_weight,
            shaping_weight=shaping_weight,
            shaping_std=shaping_std,
        )
        if obstacle_radius <= 0.0:
            raise ValueError(f"obstacle_radius must be positive, got {obstacle_radius}")
        if collision_penalty < 0.0:
            raise ValueError(f"collision_penalty must be non-negative, got {collision_penalty}")
        if ee_speed_max <= 0.0:
            raise ValueError(f"ee_speed_max must be positive, got {ee_speed_max}")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        if distance_weight > 0.0 or shaping_weight < 0.0:
            raise ValueError(
                "the declared reward range assumes distance_weight <= 0 and shaping_weight >= 0, "
                f"got {distance_weight} and {shaping_weight}"
            )
        if state_schema.width(presence_channel) != 1:
            raise ValueError(
                f"{presence_channel!r} must be one slot wide, got "
                f"{state_schema.width(presence_channel)}"
            )
        self.obstacle_center = np.asarray(obstacle_center, dtype=float).reshape(3)
        self.obstacle_radius = float(obstacle_radius)
        self.collision_penalty = float(collision_penalty)
        self.ee_speed_max = float(ee_speed_max)
        self.step_dt = float(step_dt)
        self.presence_channel = presence_channel

    def end_effector_position_of(self, state: Any) -> np.ndarray:
        """Analytic hand position at ``state``, shape ``(3,)`` (or batched for row inputs)."""
        relative = self.state_schema.block(state, self.joint_position_channel)
        angles = relative[..., self.arm_joint_indices] + self.default_joint_positions
        return self.chain.end_effector_position(angles)

    def proximity_contact(self, state: Any) -> float:
        """1.0 when the hand is within the obstacle radius, regardless of presence."""
        distance = float(
            np.linalg.norm(self.end_effector_position_of(state) - self.obstacle_center)
        )
        return float(distance <= self.obstacle_radius)

    def contact(self, state: Any) -> float:
        """1.0 when the hand touches the obstacle while it is present, else 0.0."""
        presence = float(self.state_schema.block(state, self.presence_channel).reshape(-1)[0])
        if presence <= 0.5:
            return 0.0
        return self.proximity_contact(state)

    def end_effector_speed(self, state: Any, next_state: Any) -> float:
        """Hand displacement rate over the step, in m/s."""
        displacement = self.end_effector_position_of(next_state) - self.end_effector_position_of(
            state
        )
        return float(np.linalg.norm(displacement)) / self.step_dt

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        base = super().reward(state, action, next_state)
        resulting = state if next_state is None else next_state
        if self.contact(resulting) == 0.0:
            return base
        speed = 0.0 if next_state is None else self.end_effector_speed(state, next_state)
        return base - self.collision_penalty * float(
            _clipped_severity(speed, self.ee_speed_max)
        )

    def reward_rows(
        self, state_rows: np.ndarray, next_rows: Optional[np.ndarray]
    ) -> np.ndarray:
        """Vectorized reward over row batches; matches :meth:`reward` element-wise."""
        resulting = state_rows if next_rows is None else next_rows
        positions = self.end_effector_position_of(resulting)
        goal = resulting[:, self.state_schema.slice_of(self.command_channel)][
            :, : self.command_position_width
        ]
        distance = np.linalg.norm(positions - goal, axis=-1)
        base = self.distance_weight * distance + self.shaping_weight * (
            1.0 - np.tanh(distance / self.shaping_std)
        )
        presence = resulting[:, self.state_schema.slice_of(self.presence_channel)].reshape(-1)
        obstacle_distance = np.linalg.norm(positions - self.obstacle_center, axis=-1)
        in_contact = (presence > 0.5) & (obstacle_distance <= self.obstacle_radius)
        if next_rows is None:
            return base  # zero speed -> zero penalty, matching the scalar fallback
        previous = self.end_effector_position_of(state_rows)
        speed = np.linalg.norm(positions - previous, axis=-1) / self.step_dt
        penalty = -self.collision_penalty * _clipped_severity(speed, self.ee_speed_max)
        return base + np.where(in_contact, penalty, 0.0)


class HazardReachIsaacModel(ManipulatorIsaacModel):
    """Franka reach with one latent-presence obstacle sphere, run model-is-world.

    The state extends the reach task's ``joint_pos + joint_vel + command + last_action`` with the
    latent presence bit (carried, Bernoulli(``p_present``) once per episode) and the one-slot
    terminal flag (used only when ``is_contact_terminal``). The obstacle itself — centre and
    radius — is fixed workspace geometry held by the model, not state: the arm cannot move the
    world frame, so unlike the navigation hazards nothing needs to ride a frame change. The
    observation adds a one-bit presence signal informative only while the hand is within
    ``signal_radius`` of the obstacle centre.

    The task: bring the hand within ``success_radius`` of the commanded position — and *hold* it
    there; reaching is not terminal, matching the Isaac reach task — without sweeping through the
    obstacle while it is present.

    Attributes:
        hazard_reward: The :class:`HazardReachRewardModel` scoring the task.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> model = HazardReachIsaacModel(p_present=1.0)
        >>> state = model.initial_state_dist().sample(1)[0]
        >>> float(model.state_schema.block(state, "obstacle_present")[0])
        1.0
        >>> model.is_terminal(state)  # contact is not terminal by default
        False
    """

    def __init__(
        self,
        obstacle_center: Sequence[float] = (0.44, 0.20, 0.51),
        obstacle_radius: float = 0.05,
        discount_factor: float = 0.99,
        step_dt: float = 0.1,
        action_presets: Optional[Sequence[ArrayLike]] = None,
        goal_command: Sequence[float] = (0.45, 0.25, 0.52, 1.0, 0.0, 0.0, 0.0),
        p_present: float = 0.5,
        collision_penalty: float = 25.0,
        ee_speed_max: float = 1.0,
        is_contact_terminal: bool = False,
        success_radius: float = 0.15,
        signal_radius: float = 0.15,
        signal_accuracy: float = 0.9,
        tracking_gain: float = 0.5,
        action_scale: float = 0.5,
        default_joint_positions: Optional[Sequence[float]] = None,
        position_noise_std: float = 0.01,
        velocity_noise_std: float = 0.1,
        action_noise_std: float = 1e-3,
        observation_noise_std: float = 0.05,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the hazard reach model.

        Args:
            obstacle_center: Obstacle centre in the robot base frame, ``(x, y, z)``. The default
                sits on the hand's approach to the default goal, outside the signal zone of the
                default start pose — the presence has to be probed by approaching.
            obstacle_radius: Contact radius, in metres.
            discount_factor: POMDP discount factor.
            step_dt: Control-step duration in seconds.
            action_presets: Joint-target commands (7-wide) to plan over. ``None`` (the default)
                uses :data:`DEFAULT_REACH_ACTION_PRESETS`.
            goal_command: The fixed ``command`` block, position (3) + quaternion (4).
            p_present: Probability the obstacle is present, drawn once per episode.
            collision_penalty: Contact penalty magnitude at saturated hand speed; non-negative.
            ee_speed_max: Severity saturation hand speed, in m/s.
            is_contact_terminal: Whether a present-obstacle contact sets the terminal slot.
                ``False`` by default — a reach episode runs to its horizon.
            success_radius: Hand-to-command distance under which the step counts as reached.
            signal_radius: Hand range within which the presence signal is informative, in metres.
            signal_accuracy: In-range signal accuracy, in ``(0.5, 1]``.
            tracking_gain: Per-step lag gain of the joint-position controller, in ``(0, 1]``.
            action_scale: Scale mapping an action into a joint-position target.
            default_joint_positions: The arm's default joint pose, chain order. ``None`` uses
                :data:`FRANKA_DEFAULT_ARM_POSE`.
            position_noise_std: Std of the joint-position process noise, in radians.
            velocity_noise_std: Std of the joint-velocity process noise, in radians per second.
            action_noise_std: Std on the recorded-action block; small but strictly positive.
            observation_noise_std: Std of the Gaussian noise on the observed joint channels.
            name: Model name; defaults to the class name.

        Raises:
            ValueError: If ``p_present`` is outside ``[0, 1]``, ``goal_command`` is not 7 wide,
                or a scalar parameter is out of range.
        """
        if not 0.0 <= p_present <= 1.0:
            raise ValueError(f"p_present must be in [0, 1], got {p_present}")
        if success_radius <= 0.0:
            raise ValueError(f"success_radius must be positive, got {success_radius}")
        goal = tuple(float(value) for value in goal_command)
        if len(goal) != REACH_COMMAND_WIDTH:
            raise ValueError(
                f"goal_command must have {REACH_COMMAND_WIDTH} entries (position + quaternion), "
                f"got {len(goal)}"
            )
        chain = franka_panda_chain()
        resolved_default_pose = tuple(
            float(value)
            for value in (
                FRANKA_DEFAULT_ARM_POSE if default_joint_positions is None else default_joint_positions
            )
        )
        schema = IsaacChannelSchema(
            (
                ("joint_pos", FRANKA_OBSERVED_JOINTS),
                ("joint_vel", FRANKA_OBSERVED_JOINTS),
                ("command", REACH_COMMAND_WIDTH),
                ("last_action", FRANKA_COMMANDED_JOINTS),
                (OBSTACLE_PRESENCE_CHANNEL, 1),
                (EPISODE_DONE_CHANNEL, 1),
            )
        )
        arm_indices = tuple(range(FRANKA_COMMANDED_JOINTS))
        reward_model = HazardReachRewardModel(
            state_schema=schema,
            chain=chain,
            joint_position_channel="joint_pos",
            command_channel="command",
            default_joint_positions=resolved_default_pose,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            collision_penalty=collision_penalty,
            ee_speed_max=ee_speed_max,
            step_dt=step_dt,
            arm_joint_indices=arm_indices,
        )
        resolved_presets = (
            DEFAULT_REACH_ACTION_PRESETS if action_presets is None else action_presets
        )
        observation_models: Dict[str, IsaacObservationModel] = {
            "joint_pos": GaussianChannelObservationModel(
                channel="joint_pos", noise_std=observation_noise_std
            ),
            "joint_vel": GaussianChannelObservationModel(
                channel="joint_vel", noise_std=observation_noise_std
            ),
            HAZARD_SIGNAL_CHANNEL: EndEffectorPresenceSignalObservationModel(
                chain=chain,
                default_joint_positions=resolved_default_pose,
                obstacle_center=obstacle_center,
                arm_joint_indices=arm_indices,
                signal_radius=signal_radius,
                accuracy_inside=signal_accuracy,
            ),
        }
        super().__init__(
            state_schema=schema,
            action_presets=[_coerce_action_preset(preset) for preset in resolved_presets],
            discount_factor=discount_factor,
            step_dt=step_dt,
            tracking_gain=tracking_gain,
            chain=chain,
            default_joint_positions=resolved_default_pose,
            arm_joint_indices=arm_indices,
            actuated_indices=arm_indices,
            action_scale=action_scale,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std,
            action_noise_std=action_noise_std,
            reward_model=reward_model,
            observation_models=observation_models,
            reward_range=self._hazard_reward_range(reward_model, p_present, goal),
            name=name if name is not None else type(self).__name__,
        )
        self.hazard_reward = reward_model
        self.obstacle_center = tuple(float(value) for value in np.asarray(obstacle_center).reshape(3))
        self.obstacle_radius = float(obstacle_radius)
        self.goal_command = goal
        self.p_present = float(p_present)
        self.collision_penalty = float(collision_penalty)
        self.ee_speed_max = float(ee_speed_max)
        self.is_contact_terminal = bool(is_contact_terminal)
        self.success_radius = float(success_radius)
        self.signal_radius = float(signal_radius)
        self.signal_accuracy = float(signal_accuracy)
        self.step_dt = float(step_dt)
        self.tracking_gain = float(tracking_gain)
        self.action_scale = float(action_scale)
        self.default_joint_positions = resolved_default_pose
        self.position_noise_std = float(position_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self.action_noise_std = float(action_noise_std)
        self.observation_noise_std = float(observation_noise_std)

        self._presence_index = self.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL).start
        self._done_index = self.state_schema.slice_of(EPISODE_DONE_CHANNEL).start
        self._warm_observation_caches()

    @staticmethod
    def _hazard_reward_range(
        reward_model: HazardReachRewardModel,
        p_present: float,
        goal_command: Tuple[float, ...],
    ) -> Tuple[float, float]:
        """Exact per-step reward bounds, enumerated over every term and flag.

        * The hand position is bounded: each modified-DH link translates by at most
          ``|a_i| + |d_i|`` and the tool transform by its own translation, so
          ``|FK(q)| <= sum(|a|) + sum(|d|) + |tool|`` for *any* joint vector — including the
          unbounded ones process noise can produce. The largest possible hand-to-goal distance is
          therefore that reach plus the goal's own norm.
        * The distance term (``distance_weight <= 0``, enforced in the reward constructor) is
          therefore bounded below by ``distance_weight * d_max``; the shaping term lies in
          ``(0, shaping_weight]``, contributing 0 to the minimum and its weight to the maximum
          (attained at zero distance).
        * The contact penalty lies in ``[-collision_penalty, 0]`` thanks to the severity clip; it
          stacks with the worst distance term. It is excluded only when it cannot fire: a zero
          penalty or ``p_present == 0`` (the presence bit is drawn once at reset and carried as a
          point mass).
        * ``is_contact_terminal`` changes episode length, not the per-step bound.
        """
        chain = reward_model.chain
        reach = float(
            np.sum(np.abs(np.asarray(chain.link_lengths)))
            + np.sum(np.abs(np.asarray(chain.link_offsets)))
            + np.linalg.norm(np.asarray(chain.tool_transform)[:3, 3])
        )
        max_distance = reach + float(
            np.linalg.norm(np.asarray(goal_command[: reward_model.command_position_width]))
        )
        maximum = reward_model.shaping_weight
        minimum = reward_model.distance_weight * max_distance
        if reward_model.collision_penalty > 0.0 and p_present > 0.0:
            minimum -= reward_model.collision_penalty
        return (minimum, maximum)

    def _warm_observation_caches(self) -> None:
        # Same config_id-stability warm-up as the navigation model: the Gaussian channels
        # memoize per-width normals lazily, and config_id hashes those caches.
        zero = self.state_schema.split(np.zeros(self.state_schema.total_dim))
        assert self.observation_models is not None
        self.observation_models["joint_pos"].log_probability(
            zero, np.zeros(FRANKA_OBSERVED_JOINTS)
        )
        self.observation_models["joint_vel"].log_probability(
            zero, np.zeros(FRANKA_OBSERVED_JOINTS)
        )

    # ── Terminal bookkeeping ────────────────────────────────────────────
    def _contact_rows(self, rows: np.ndarray) -> np.ndarray:
        """Present-obstacle contact indicator per row, shape ``(N,)``."""
        positions = self.hazard_reward.end_effector_position_of(rows)
        distance = np.linalg.norm(
            positions - self.hazard_reward.obstacle_center[np.newaxis, :], axis=-1
        )
        presence = rows[:, self._presence_index]
        return ((presence > 0.5) & (distance <= self.hazard_reward.obstacle_radius)).astype(float)

    def _apply_done(self, rows: np.ndarray) -> None:
        """Set the terminal slot on rows in present-obstacle contact, sticky, in place."""
        if not self.is_contact_terminal:
            return
        rows[:, self._done_index] = np.maximum(
            rows[:, self._done_index], self._contact_rows(rows)
        )

    # ── Dynamics ────────────────────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        result = super().sample_next_state(state, action, n_samples)
        self._apply_done(np.atleast_2d(result))  # atleast_2d is a view; edits land in result
        return result

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        """One next state per particle under one action, with no per-particle Python loop."""
        rows = np.atleast_2d(np.asarray(states, dtype=float))
        transition = self.joint_transition
        positions = rows[:, : transition.position_width]
        command = (
            np.asarray(action, dtype=float).reshape(-1)[: transition.action_dim]
            * transition.action_scale
        )
        next_positions = positions.copy()
        driven = positions[:, transition.actuated_indices]
        next_positions[:, transition.actuated_indices] = driven + transition.tracking_gain * (
            command[np.newaxis, :] - driven
        )
        velocities = (next_positions - positions) / transition.step_dt
        means = np.concatenate(
            [
                next_positions,
                velocities,
                np.tile(
                    np.asarray(action, dtype=float).reshape(-1)[: transition.action_dim],
                    (rows.shape[0], 1),
                ),
            ],
            axis=-1,
        )
        samples = means + np.random.normal(
            0.0, transition.process_noise_std, size=means.shape
        )
        out = rows.copy()
        out[:, self._driven_indices] = samples
        self._apply_done(out)
        return out

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Density with the terminal slot scored by its deterministic contact rule.

        Same construction as the navigation model: driven block by the transition, the presence
        bit (and the untouched command block) as a point mass, and the slot against the value the
        contact rule implies for each candidate — so a legitimate 0-to-1 flip is not scored as
        impossible.
        """
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))
        source = np.asarray(state, dtype=float).reshape(-1)
        driven = np.asarray(
            self.joint_transition.log_probability(
                source[self._driven_indices], action, candidates[:, self._driven_indices]
            ),
            dtype=float,
        ).reshape(-1)
        command_slice = self.state_schema.slice_of(self.command_channel)
        static_match = np.all(
            np.isclose(candidates[:, command_slice], source[command_slice]), axis=-1
        ) & np.isclose(candidates[:, self._presence_index], source[self._presence_index])
        previous_done = float(source[self._done_index])
        if self.is_contact_terminal:
            expected = candidates.copy()
            expected[:, self._done_index] = previous_done
            self._apply_done(expected)
            expected_done = expected[:, self._done_index]
        else:
            expected_done = np.full(candidates.shape[0], previous_done)
        done_match = np.isclose(candidates[:, self._done_index], expected_done)
        return np.where(static_match & done_match, driven, -np.inf)

    # ── Reward ──────────────────────────────────────────────────────────
    @property
    def reward_requires_next_state(self) -> bool:
        """True: the hand-speed severity is a function of both endpoint states."""
        return True

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        del action  # the objective and the severity read the endpoint states alone
        state_rows = np.atleast_2d(np.asarray(states, dtype=float))
        next_rows = (
            None if next_states is None else np.atleast_2d(np.asarray(next_states, dtype=float))
        )
        return self.hazard_reward.reward_rows(state_rows, next_rows)

    # ── Termination / episode structure ─────────────────────────────────
    def is_terminal(self, state: Any) -> bool:
        # Reaching is deliberately not terminal — the reach task holds the pose — so only the
        # contact slot ends an episode, and only when contact is configured terminal.
        vector = np.asarray(state, dtype=float).reshape(-1)
        return bool(vector[self._done_index] > _TERMINAL_THRESHOLD)

    def initial_state_dist(self) -> Distribution:
        template = self.state_schema.pack(
            {
                "joint_pos": np.zeros(FRANKA_OBSERVED_JOINTS),
                "joint_vel": np.zeros(FRANKA_OBSERVED_JOINTS),
                "command": np.asarray(self.goal_command, dtype=float),
                "last_action": np.zeros(FRANKA_COMMANDED_JOINTS),
                OBSTACLE_PRESENCE_CHANNEL: np.zeros(1),
                EPISODE_DONE_CHANNEL: np.zeros(1),
            }
        )
        return _TemplateWithBernoulliSlots(
            template=template,
            bernoulli_slice=self.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL),
            probability=self.p_present,
        )

    def initial_observation_dist(self) -> Distribution:
        return _ObservationOfInitialState(self)

    # ── Metrics ─────────────────────────────────────────────────────────
    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Per-step measurement channels; the hand-speed pair describes the transition.

        The contact, goal and zone channels are pure functions of ``state`` and are reported on
        every call, including the terminal bookkeeping one — that is where a terminal contact
        lives. The hand speed is a property of the transition, so on the bookkeeping call
        (``next_state is None``) the speed channels are omitted rather than reported as a
        stand-in zero. Forward kinematics is deterministic; no randomness is consumed.
        """
        del action
        vector = np.asarray(state, dtype=float).reshape(-1)
        contact = self.hazard_reward.contact(vector)
        proximity = self.hazard_reward.proximity_contact(vector)
        hand = self.hazard_reward.end_effector_position_of(vector)
        goal = float(
            np.linalg.norm(
                hand
                - np.asarray(self.goal_command[: self.hazard_reward.command_position_width])
            )
            < self.success_radius
        )
        in_zone = float(
            np.linalg.norm(hand - self.hazard_reward.obstacle_center) <= self.signal_radius
        )
        failure = float(vector[self._done_index] > _TERMINAL_THRESHOLD)
        info = {
            "recorded_step": 1.0,
            "obstacle_contact": contact,
            "proximity_contact": proximity,
            "signal_zone": in_zone,
            "goal_reached": goal,
            "ended_by_goal": goal,
            "ended_by_failure": failure,
            "ended_by_timeout": 1.0 - max(goal, failure),
        }
        if next_state is not None:
            speed = self.hazard_reward.end_effector_speed(vector, next_state)
            next_contact = self.hazard_reward.contact(next_state)
            info["ee_speed_mps"] = speed
            info["speed_at_contact_mps"] = speed if next_contact > 0.5 else 0.0
        return info

    def get_metric_specs(self) -> List[StepInfoMetric]:
        specs = _hazard_metric_specs(
            speed_channel="ee_speed_mps",
            speed_metric="mean_ee_speed_mps",
            contact_channel="proximity_contact",
            contact_rate="proximity_contact_rate",
            contact_steps="average_obstacle_contact_steps",
            bad_channel="obstacle_contact",
            bad_rate="obstacle_contact_rate",
        )
        # The shared layout counts *contact* steps on the contact channel; for the reach model
        # the count that matters is present-obstacle contact, so re-point that one spec.
        return [
            StepInfoMetric(
                name=spec.name,
                channel="obstacle_contact" if spec.name == "average_obstacle_contact_steps" else spec.channel,
                per_episode=spec.per_episode,
            )
            for spec in specs
        ]


class ConstrainedHazardNavigationIsaacModel(  # pylint: disable=too-many-ancestors
    HazardNavigationIsaacModel, ConstrainedEnvironment
):
    """Hazard navigation with the danger moved from the reward to a constraint channel.

    Mirrors the chance-constrained pattern: a constrained planner reads
    :meth:`constraint_cost` — the bad-contact indicator of the transition's realised successor —
    while the reward keeps only the navigation objective. ``collision_penalty`` is forced to zero
    so the same event is not encoded twice; the declared reward range shrinks accordingly, which
    also narrows anything derived from it (an exploration constant, say) relative to the
    unconstrained twin.

    Note the indicator reads ``next_state`` rather than ``state``: the state being left is
    already fixed and carries no gradient for action selection. Over an episode the two counts
    differ only by the initial state, which starts outside every hazard.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["collision_penalty"] = 0.0
        super().__init__(*args, **kwargs)
        if self.hazard_reward.collision_penalty != 0.0:
            raise ValueError(
                "collision_penalty did not reach the reward model; the danger term is still in "
                "the reward channel"
            )

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        del state, action
        # A fresh length-1 array per call: constrained planners cache the returned vector per
        # belief child, and a shared buffer would alias those caches together.
        return np.array([self.hazard_reward.bad_contact(next_state)], dtype=np.float64)

    def constraint_cost_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Union[np.ndarray, Sequence[Any]],
    ) -> np.ndarray:
        del states, action
        rows = np.atleast_2d(np.asarray(next_states, dtype=float))
        centers = rows[:, self.state_schema.slice_of(HAZARD_XY_CHANNEL)].reshape(
            rows.shape[0], self.num_hazards, 2
        )
        types = rows[:, self.state_schema.slice_of(HAZARD_TYPE_CHANNEL)]
        contact = np.linalg.norm(centers, axis=-1) <= self.hazard_reward.hazard_radii
        bad = np.any(contact & (types > 0.5), axis=-1)
        return bad.astype(np.float64).reshape(-1, 1)


class ConstrainedHazardReachIsaacModel(  # pylint: disable=too-many-ancestors
    HazardReachIsaacModel, ConstrainedEnvironment
):
    """Hazard reach with the contact penalty moved to a constraint channel.

    Same construction as :class:`ConstrainedHazardNavigationIsaacModel`: the constraint cost is
    the present-obstacle contact indicator of the realised successor, and ``collision_penalty``
    is forced to zero so the event is encoded once.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["collision_penalty"] = 0.0
        super().__init__(*args, **kwargs)
        if self.hazard_reward.collision_penalty != 0.0:
            raise ValueError(
                "collision_penalty did not reach the reward model; the danger term is still in "
                "the reward channel"
            )

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        del state, action
        return np.array([self.hazard_reward.contact(next_state)], dtype=np.float64)

    def constraint_cost_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Union[np.ndarray, Sequence[Any]],
    ) -> np.ndarray:
        del states, action
        rows = np.atleast_2d(np.asarray(next_states, dtype=float))
        return self._contact_rows(rows).astype(np.float64).reshape(-1, 1)


# The twins take (*args, **kwargs) so the penalty can be forced without restating twenty
# parameters, but Environment.to_dict reads the constructor signature to decide which attributes
# to serialize — an opaque signature would serialize nothing and from_dict would silently rebuild
# defaults. Publishing the parent's signature restores the round trip; collision_penalty
# re-serializes as the forced 0.0, which the twin constructor forces again on the way back in.
ConstrainedHazardNavigationIsaacModel.__init__.__signature__ = inspect.signature(  # type: ignore[attr-defined]
    HazardNavigationIsaacModel.__init__
)
ConstrainedHazardReachIsaacModel.__init__.__signature__ = inspect.signature(  # type: ignore[attr-defined]
    HazardReachIsaacModel.__init__
)


__all__ = [
    "DEFAULT_NAVIGATION_ACTION_PRESETS",
    "DEFAULT_REACH_ACTION_PRESETS",
    "EPISODE_DONE_CHANNEL",
    "FRANKA_DEFAULT_ARM_POSE",
    "HAZARD_SIGNAL_CHANNEL",
    "HAZARD_TYPE_CHANNEL",
    "HAZARD_XY_CHANNEL",
    "OBSTACLE_PRESENCE_CHANNEL",
    "ConstrainedHazardNavigationIsaacModel",
    "ConstrainedHazardReachIsaacModel",
    "EndEffectorPresenceSignalObservationModel",
    "HazardNavigationIsaacModel",
    "HazardNavigationRewardModel",
    "HazardReachIsaacModel",
    "HazardReachRewardModel",
    "HazardRelativeTransition",
    "RelativeHazardSignalObservationModel",
]
