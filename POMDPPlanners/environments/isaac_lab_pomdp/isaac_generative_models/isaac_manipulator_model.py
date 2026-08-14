# SPDX-License-Identifier: MIT

"""Analytic manipulator model for IsaacLab joint-position-controlled arms.

A task like ``Isaac-Reach-Franka-v0`` does not hand the planner torques either. Its action is a
7-vector of joint-position *targets* riding an implicit PD controller, and the objective is the
distance from the hand to a commanded pose. Both halves of that are structural, so both can be
written down instead of fitted — the same move
:mod:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_unicycle_model`
makes for a velocity-commanded base, and CARLA's kinematic bicycle makes for a car.

Fitting them instead is what fails. One linear map ``s' = A s + B a + b``, ridge-fitted from a few
hundred random transitions of a 7-DoF arm, cannot separate the actions: every branch of the search
scores alike, the argmax never moves, and the episode records a single action index repeated to the
horizon. The fix is not a deeper search — it is a model in which two different commands lead to two
measurably different hands.

Three honest approximations, stated so a study can check them rather than discover them:

* **The joints lag, they do not teleport.** Under a joint-position controller the joint moves a
  fraction of the way to its target each control step. A first-order lag
  ``q' = q + gain * (target - q)`` is the simplest model with that shape. ``tracking_gain`` is
  therefore a property of the controller, not a free knob: measure it from a rollout with
  :func:`calibrate_tracking_gain` rather than assuming a value.
* **Kinematics is exact, dynamics is not.** The hand's pose is an exact trigonometric function of
  the joint angles, and :class:`ModifiedDHChain` computes it offline, with no simulator — so the
  reward is the task's real objective rather than a regression onto it. What the model does *not*
  carry is gravity, inertia, joint coupling or contact; those enter as the process noise the lag
  is wrapped in.
* **Joint velocity is the lag's own difference quotient.** ``(q' - q) / step_dt`` is the velocity
  the position model implies, not an independently modelled state. It is right in steady tracking
  and wrong during a fast reversal, which matters for the observation likelihood (the velocity
  channel is observed) but not for the reach objective (which reads positions only).

Classes:
    ModifiedDHChain: Offline forward kinematics of a serial chain in modified DH parameters.
    JointLagTransition: First-order lag of joint positions toward a commanded joint target.
    ReachRewardModel: The reach task's own objective, computed analytically through the chain.
    ManipulatorIsaacModel: Factored model wired with a joint lag and an analytic reach reward.

Functions:
    calibrate_tracking_gain: Least-squares estimate of the controller lag gain from a rollout.
    franka_panda_chain: The Franka Emika Panda chain from ``panda_link0`` to ``panda_hand``.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model import (
    FactoredIsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_model_pomdp import (
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    RewardModel,
    TransitionModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_model import (
    IsaacObservationModel,
)

#: Modified-DH link lengths ``a_i`` (metres) for the Panda, joint 1 to joint 7.
FRANKA_LINK_LENGTHS = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])

#: Modified-DH link offsets ``d_i`` (metres) for the Panda, joint 1 to joint 7.
FRANKA_LINK_OFFSETS = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0])

#: Modified-DH link twists ``alpha_i`` (radians) for the Panda, joint 1 to joint 7.
FRANKA_LINK_TWISTS = np.array(
    [0.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0]
)

#: Distance from the joint-7 frame to the ``panda_link8`` flange along its z axis, in metres.
FRANKA_FLANGE_OFFSET = 0.107

#: Rotation of ``panda_hand`` about the flange z axis, in radians (the hand is not translated).
FRANKA_HAND_ROTATION = -np.pi / 4.0


def _validated_indices(indices: Optional[ArrayLike], width: int, label: str) -> np.ndarray:
    """Check an index map into a ``width``-wide block, defaulting to the identity map."""
    resolved = np.arange(width) if indices is None else np.asarray(indices, dtype=int).reshape(-1)
    if resolved.size == 0:
        raise ValueError(f"{label} must name at least one joint")
    if np.any(resolved < 0) or np.any(resolved >= width):
        raise ValueError(f"{label} must index a {width}-wide joint block, got {resolved.tolist()}")
    if len(set(resolved.tolist())) != resolved.size:
        raise ValueError(f"{label} names the same joint twice: {resolved.tolist()}")
    return resolved


def _homogeneous_z_rotation(angle: float, offset: float) -> np.ndarray:
    """A 4x4 transform rotating about z by ``angle`` and translating along z by ``offset``."""
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, offset],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass(frozen=True)
class ModifiedDHChain:
    """Forward kinematics of a serial chain, in modified (Craig) Denavit-Hartenberg parameters.

    Each joint contributes ``Rx(alpha_i) Tx(a_i) Rz(theta_i) Tz(d_i)``, and a fixed
    ``tool_transform`` maps the last joint frame onto the frame whose pose the task cares about.
    Everything is pure numpy: the chain is a function of joint angles alone, so it evaluates on
    belief particles with no simulator attached, which is exactly what planning inside a model
    requires.

    Attributes:
        link_lengths: ``a_i`` per joint, in metres.
        link_offsets: ``d_i`` per joint, in metres.
        link_twists: ``alpha_i`` per joint, in radians.
        tool_transform: Fixed 4x4 transform from the last joint frame to the tool frame.

    Example:
        >>> import numpy as np
        >>> chain = franka_panda_chain()
        >>> home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
        >>> position = chain.end_effector_position(home)
        >>> [round(float(position[0]), 3), round(float(position[2]), 3)]
        [0.307, 0.59]
        >>> bool(abs(position[1]) < 1e-9)  # the home pose is in the robot's sagittal plane
        True
    """

    link_lengths: np.ndarray
    link_offsets: np.ndarray
    link_twists: np.ndarray
    tool_transform: np.ndarray

    def __post_init__(self) -> None:
        widths = {
            np.asarray(self.link_lengths).shape,
            np.asarray(self.link_offsets).shape,
            np.asarray(self.link_twists).shape,
        }
        if len(widths) != 1 or len(next(iter(widths))) != 1:
            raise ValueError(
                "link_lengths, link_offsets and link_twists must be 1-D arrays of equal length, "
                f"got shapes {sorted(str(shape) for shape in widths)}"
            )
        if np.asarray(self.tool_transform).shape != (4, 4):
            raise ValueError(
                f"tool_transform must be 4x4, got {np.asarray(self.tool_transform).shape}"
            )

    @property
    def num_joints(self) -> int:
        """Number of revolute joints in the chain."""
        return int(np.asarray(self.link_lengths).shape[0])

    def end_effector_transform(self, joint_angles: ArrayLike) -> np.ndarray:
        """Pose of the tool frame in the chain's base frame.

        Args:
            joint_angles: ``(num_joints,)`` angles, or a ``(N, num_joints)`` batch.

        Returns:
            A ``(4, 4)`` transform, or a ``(N, 4, 4)`` batch of them.

        Raises:
            ValueError: If the trailing dimension is not ``num_joints`` wide.
        """
        requested = np.asarray(joint_angles, dtype=float)
        angles = np.atleast_2d(requested)
        if angles.shape[-1] != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joint angles, got {angles.shape[-1]}")
        links = self._link_transforms(angles)
        pose = links[:, 0]
        for index in range(1, self.num_joints):
            pose = pose @ links[:, index]
        pose = pose @ self.tool_transform
        return pose[0] if requested.ndim == 1 else pose

    def end_effector_position(self, joint_angles: ArrayLike) -> np.ndarray:
        """Translation of the tool frame in the chain's base frame, in metres."""
        return self.end_effector_transform(joint_angles)[..., :3, 3]

    def _link_transforms(self, angles: np.ndarray) -> np.ndarray:
        """Per-joint ``(N, num_joints, 4, 4)`` modified-DH transforms."""
        cos_t, sin_t = np.cos(angles), np.sin(angles)
        cos_a = np.cos(np.asarray(self.link_twists, dtype=float))
        sin_a = np.sin(np.asarray(self.link_twists, dtype=float))
        lengths = np.asarray(self.link_lengths, dtype=float)
        offsets = np.asarray(self.link_offsets, dtype=float)
        transforms = np.zeros((angles.shape[0], self.num_joints, 4, 4))
        transforms[..., 0, 0] = cos_t
        transforms[..., 0, 1] = -sin_t
        transforms[..., 0, 3] = lengths
        transforms[..., 1, 0] = sin_t * cos_a
        transforms[..., 1, 1] = cos_t * cos_a
        transforms[..., 1, 2] = -sin_a
        transforms[..., 1, 3] = -offsets * sin_a
        transforms[..., 2, 0] = sin_t * sin_a
        transforms[..., 2, 1] = cos_t * sin_a
        transforms[..., 2, 2] = cos_a
        transforms[..., 2, 3] = offsets * cos_a
        transforms[..., 3, 3] = 1.0
        return transforms


def franka_panda_chain() -> ModifiedDHChain:
    """The Franka Emika Panda arm from ``panda_link0`` to the ``panda_hand`` frame.

    The parameters are Franka's own published modified-DH table. ``panda_hand`` sits on the
    ``panda_link8`` flange with no translation and a -45 degree roll about its z axis, so the hand
    *position* this chain returns is the flange position — which is what the reach reward measures.

    Returns:
        The seven-joint chain with the flange-and-hand transform as its tool frame.
    """
    return ModifiedDHChain(
        link_lengths=FRANKA_LINK_LENGTHS,
        link_offsets=FRANKA_LINK_OFFSETS,
        link_twists=FRANKA_LINK_TWISTS,
        tool_transform=_homogeneous_z_rotation(FRANKA_HAND_ROTATION, FRANKA_FLANGE_OFFSET),
    )


def calibrate_tracking_gain(
    joint_positions: ArrayLike,
    next_joint_positions: ArrayLike,
    actions: ArrayLike,
    action_scale: float,
) -> float:
    """Estimate the controller's per-step lag gain from a recorded rollout.

    The gain is what makes this model a model of *this* robot rather than of a generic one, and it
    is measurable: it is the single scalar that best explains how far the joints actually moved
    toward their commanded targets. Calibrating it offline from a rollout is how you would
    characterise a real arm, and it needs no privileged access — joint angles and the commands you
    issued are both things the operator already has.

    Args:
        joint_positions: ``(N, J)`` joint positions relative to the controller's default offset.
        next_joint_positions: ``(N, J)`` the positions one control step later.
        actions: ``(N, J)`` the commanded action vectors applied at each step.
        action_scale: The action term's scale, so ``action_scale * action`` is the target.

    Returns:
        The least-squares gain, clipped into ``(0, 1]``: a joint cannot move away from its target
        under a position controller, nor overshoot it within one step under a first-order lag.

    Raises:
        ValueError: If the three arrays disagree in shape or the rollout never commands a move.
    """
    current = np.atleast_2d(np.asarray(joint_positions, dtype=float))
    following = np.atleast_2d(np.asarray(next_joint_positions, dtype=float))
    commanded = np.atleast_2d(np.asarray(actions, dtype=float)) * float(action_scale)
    if not current.shape == following.shape == commanded.shape:
        raise ValueError(
            "joint_positions, next_joint_positions and actions must share a shape, got "
            f"{current.shape}, {following.shape} and {commanded.shape}"
        )
    error = commanded - current
    denominator = float(np.sum(error * error))
    if denominator <= 0.0:
        raise ValueError(
            "the rollout never commands a joint away from where it already is, so the lag gain "
            "is unidentifiable; drive the arm with non-trivial actions before calibrating"
        )
    gain = float(np.sum((following - current) * error) / denominator)
    return float(np.clip(gain, 1e-3, 1.0))


class JointLagTransition(TransitionModel):
    """First-order lag of joint positions toward a commanded joint-position target.

    The driven state block is ``[joint_pos (P), joint_vel (P), last_action (J)]``, in that order.
    One control step moves each *commanded* joint a ``tracking_gain`` fraction of the way to
    ``action_scale * action``, reports the implied difference-quotient velocity, and records the
    action just applied.

    ``P`` and ``J`` differ in general and do on the Franka reach task: the observation reports all
    nine joints of the arm-plus-gripper articulation while the action commands only the seven arm
    joints. ``actuated_indices`` says which observed joint each action entry drives; the rest are
    predicted to hold their position, which is what an uncommanded joint does.

    Positions and the action block are in the controller's *relative* convention — offsets from
    the default joint pose — which is what IsaacLab's ``joint_pos_rel`` observation and its
    ``use_default_offset`` action term both use. Nothing here needs the default pose; only the
    reward, which has to reach real kinematics, does.

    Attributes:
        position_width: Number of observed joints ``P``.
        action_dim: Number of commanded joints ``J``.
        actuated_indices: Which observed joint each action entry drives.
        dim: Width of the driven block, ``2 * P + J``.
        step_dt: Control-step duration in seconds.
        tracking_gain: Fraction of the remaining target error closed per control step.
        action_scale: Scale mapping an action into a joint-position target.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> transition = JointLagTransition(position_width=2, actuated_indices=[0],
        ...                                 step_dt=0.1, tracking_gain=0.5, action_scale=1.0,
        ...                                 position_noise_std=1e-9, velocity_noise_std=1e-9,
        ...                                 action_noise_std=1e-9)
        >>> nxt = transition.sample_next_state(np.zeros(5), np.array([1.0]))
        >>> round(float(nxt[0]), 3)  # the commanded joint moves half way to its target
        0.5
        >>> round(float(nxt[2]), 3)  # and reports the velocity that implies
        5.0
        >>> bool(abs(nxt[1]) < 1e-6)  # the uncommanded joint holds its position
        True
    """

    def __init__(
        self,
        position_width: int,
        step_dt: float,
        tracking_gain: float,
        actuated_indices: Optional[ArrayLike] = None,
        action_scale: float = 1.0,
        position_noise_std: float = 0.01,
        velocity_noise_std: float = 0.1,
        action_noise_std: float = 1e-3,
    ) -> None:
        """Initialize the joint-lag transition.

        Args:
            position_width: Number of observed joints ``P``.
            step_dt: Control-step duration in seconds. Read it from the live task
                (``env.unwrapped.step_dt``) rather than guessing — the implied velocity is
                inversely proportional to it.
            tracking_gain: Fraction of the remaining target error closed per control step, in
                ``(0, 1]``. Measure it with :func:`calibrate_tracking_gain`.
            actuated_indices: Which observed joint each action entry commands. ``None`` (the
                default) means the action commands every observed joint in order.
            action_scale: Scale mapping an action into a joint-position target, from the task's
                action term.
            position_noise_std: Std of the joint-position process noise, in radians. It is applied
                to uncommanded joints too: their prediction is exact, but a zero-variance entry
                would make the whole transition density degenerate.
            velocity_noise_std: Std of the joint-velocity process noise, in radians per second.
                It is a separate parameter because the two blocks differ by a factor of
                ``1 / step_dt`` in scale, and one shared scalar would be wrong for both.
            action_noise_std: Std on the recorded-action block. The block is a copy of the action
                and carries no real uncertainty, but a zero here makes the transition density
                degenerate and every particle weight ``-inf``; keep it small.

        Raises:
            ValueError: If ``position_width`` or ``step_dt`` is not positive, ``tracking_gain`` is
                outside ``(0, 1]``, an actuated index is out of range or repeated, or any noise
                std is not strictly positive.
        """
        if position_width <= 0:
            raise ValueError(f"position_width must be positive, got {position_width}")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        if not 0.0 < tracking_gain <= 1.0:
            raise ValueError(f"tracking_gain must lie in (0, 1], got {tracking_gain}")
        stds = np.array([position_noise_std, velocity_noise_std, action_noise_std], dtype=float)
        if np.any(stds <= 0.0):
            raise ValueError("process noise standard deviations must be strictly positive")
        self.position_width = int(position_width)
        self.actuated_indices = _validated_indices(
            actuated_indices, self.position_width, "actuated_indices"
        )
        self.action_dim = int(self.actuated_indices.shape[0])
        self.dim = 2 * self.position_width + self.action_dim
        self.step_dt = float(step_dt)
        self.tracking_gain = float(tracking_gain)
        self.action_scale = float(action_scale)
        self._std = np.concatenate(
            [
                np.full(self.position_width, stds[0]),
                np.full(self.position_width, stds[1]),
                np.full(self.action_dim, stds[2]),
            ]
        )

    @property
    def process_noise_std(self) -> np.ndarray:
        """Per-entry process-noise std over the driven block, ``(2 * P + J,)``."""
        return self._std.copy()

    def _mean(self, state: Any, action: Any) -> np.ndarray:
        block = np.asarray(state, dtype=float).reshape(-1)[: self.dim]
        positions = block[: self.position_width]
        command = np.asarray(action, dtype=float).reshape(-1)[: self.action_dim]
        driven = positions[self.actuated_indices]
        next_positions = positions.copy()
        next_positions[self.actuated_indices] = driven + self.tracking_gain * (
            command * self.action_scale - driven
        )
        next_velocities = (next_positions - positions) / self.step_dt
        return np.concatenate([next_positions, next_velocities, command])

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        mean = self._mean(state, action)
        samples = mean[np.newaxis, :] + np.random.normal(0.0, self._std, size=(n_samples, self.dim))
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        mean = self._mean(state, action)
        candidates = np.atleast_2d(np.asarray(next_states, dtype=float))[:, : self.dim]
        residual = candidates - mean[np.newaxis, :]
        normalizer = float(np.sum(np.log(self._std)) + 0.5 * self.dim * np.log(2.0 * np.pi))
        return -0.5 * np.sum((residual / self._std[np.newaxis, :]) ** 2, axis=-1) - normalizer


class ReachRewardModel(RewardModel):
    """The reach task's objective, computed through the kinematic chain instead of fitted.

    The reward is a weighted sum of the end-effector's distance to the commanded position and a
    ``tanh``-shaped bonus for being close to it, mirroring IsaacLab's ``position_command_error``
    and ``position_command_error_tanh`` terms. Both read the same analytic hand position, so the
    planner optimizes the quantity the episode is actually scored on rather than a regression onto
    a few hundred samples of it.

    The weights default to the reach task's own: ``-0.2`` on the distance and ``+0.1`` on a
    ``tanh`` bonus of length scale 0.1 m. The task also penalises orientation error, action rate
    and joint velocity; those are omitted here because the position terms carry two orders of
    magnitude more weight than the action-rate and joint-velocity penalties, and because the
    success predicate the episode is scored on is a position threshold alone.

    Attributes:
        chain: The kinematics used to place the hand.
        arm_joint_indices: Which entries of the joint block feed the chain, in the chain's order.
        distance_weight: Weight on the raw distance term (negative penalises distance).
        shaping_weight: Weight on the ``tanh`` closeness bonus.
        shaping_std: Length scale of the ``tanh`` bonus, in metres.

    Example:
        >>> import numpy as np
        >>> schema = IsaacChannelSchema((("joint_pos", 7), ("command", 3)))
        >>> home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
        >>> reward_model = ReachRewardModel(
        ...     state_schema=schema, chain=franka_panda_chain(),
        ...     joint_position_channel="joint_pos", command_channel="command",
        ...     default_joint_positions=np.zeros(7), command_position_width=3)
        >>> at_goal = schema.pack({"joint_pos": home, "command": [0.307, 0.0, 0.590]})
        >>> away = schema.pack({"joint_pos": home, "command": [0.0, 0.5, 0.2]})
        >>> reward_model.reward(at_goal, None, at_goal) > reward_model.reward(away, None, away)
        True
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        chain: ModifiedDHChain,
        joint_position_channel: str,
        command_channel: str,
        default_joint_positions: ArrayLike,
        arm_joint_indices: Optional[ArrayLike] = None,
        command_position_width: int = 3,
        distance_weight: float = -0.2,
        shaping_weight: float = 0.1,
        shaping_std: float = 0.1,
    ) -> None:
        """Initialize the analytic reach reward.

        Args:
            state_schema: Named blocks of the flat state vector.
            chain: Kinematics from the robot base to the tracked body.
            joint_position_channel: Block holding joint positions *relative* to the default pose.
                It may be wider than the chain — an observation of a gripper joint the chain does
                not model — in which case ``arm_joint_indices`` selects the chain's joints.
            command_channel: Block holding the commanded pose in the robot base frame; its first
                ``command_position_width`` entries are the position.
            default_joint_positions: The chain's default joint pose, in the chain's own order,
                added to the relative positions to recover absolute joint angles. This is a
                property of the task config, not a reading from the live simulator.
            arm_joint_indices: Which entries of the joint block feed the chain, in the chain's
                order. ``None`` (the default) means the block is exactly the chain's joints, in
                order. Naming these explicitly is what stops a differently ordered articulation
                from silently reaching to the wrong place.
            command_position_width: How many leading entries of the command block are the position.
            distance_weight: Weight on the raw distance term; negative penalises distance.
            shaping_weight: Weight on the ``tanh(1 - d / std)``-style closeness bonus.
            shaping_std: Length scale of the closeness bonus, in metres.

        Raises:
            ValueError: If a channel is too narrow for the chain or the command, an index is out
                of range, or the default pose is not the chain's width.
        """
        joint_width = state_schema.width(joint_position_channel)
        self.arm_joint_indices = _validated_indices(
            arm_joint_indices, joint_width, "arm_joint_indices"
        )
        if self.arm_joint_indices.shape[0] != chain.num_joints:
            raise ValueError(
                f"the chain has {chain.num_joints} joints but arm_joint_indices selects "
                f"{self.arm_joint_indices.shape[0]} of the {joint_width} in "
                f"{joint_position_channel!r}"
            )
        if state_schema.width(command_channel) < command_position_width:
            raise ValueError(
                f"command_channel {command_channel!r} is "
                f"{state_schema.width(command_channel)} wide, too narrow to hold a "
                f"{command_position_width}-wide position"
            )
        self.state_schema = state_schema
        self.chain = chain
        self.distance_weight = float(distance_weight)
        self.shaping_weight = float(shaping_weight)
        self.shaping_std = float(shaping_std)
        self.joint_position_channel = joint_position_channel
        self.command_channel = command_channel
        self.command_position_width = int(command_position_width)
        self.default_joint_positions = np.asarray(default_joint_positions, dtype=float).reshape(-1)
        if self.default_joint_positions.shape[0] != chain.num_joints:
            raise ValueError(
                f"default_joint_positions must have {chain.num_joints} entries, got "
                f"{self.default_joint_positions.shape[0]}"
            )

    def end_effector_distance(self, state: Any) -> float:
        """Distance from the modelled hand to the commanded position, in metres."""
        relative = self.state_schema.block(state, self.joint_position_channel).reshape(-1)
        goal = self.state_schema.block(state, self.command_channel).reshape(-1)
        angles = relative[self.arm_joint_indices] + self.default_joint_positions
        position = self.chain.end_effector_position(angles)
        return float(np.linalg.norm(position - goal[: self.command_position_width]))

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        del action  # the analytic objective scores where the hand ended up, not how it got there
        distance = self.end_effector_distance(state if next_state is None else next_state)
        shaped = 1.0 - float(np.tanh(distance / self.shaping_std))
        return self.distance_weight * distance + self.shaping_weight * shaped


class ManipulatorIsaacModel(FactoredIsaacModelPOMDP):
    """Factored Isaac model whose dynamics are a joint lag and whose reward is analytic.

    A convenience wiring of :class:`JointLagTransition` and :class:`ReachRewardModel` into
    :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_factored_model.FactoredIsaacModelPOMDP`.
    The driven channels are ``(joint_position, joint_velocity, last_action)``, in that schema
    order; the commanded pose is carried through the transition untouched, because a command is
    task data and not dynamics. The two joint channels share a width and may be wider than the
    action, which is the Franka reach case: nine joints observed, seven commanded.

    Attributes:
        chain: The kinematics the reward reads.
        joint_transition: The :class:`JointLagTransition` driving the arm.
        reach_reward: The :class:`ReachRewardModel` scoring it, or ``None`` when a different
            reward model was supplied.
        joint_position_channel: The block holding relative joint positions.
        joint_velocity_channel: The block holding relative joint velocities.
        last_action_channel: The block holding the previously applied action.
        command_channel: The block holding the commanded pose.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
        ...     GaussianChannelObservationModel)
        >>>
        >>> schema = IsaacChannelSchema(
        ...     (("joint_pos", 7), ("joint_vel", 7), ("command", 7), ("last_action", 7)))
        >>> model = ManipulatorIsaacModel(
        ...     state_schema=schema,
        ...     action_presets=[np.zeros(7), np.full(7, 0.5)],
        ...     discount_factor=0.99,
        ...     step_dt=0.1,
        ...     tracking_gain=0.5,
        ...     chain=franka_panda_chain(),
        ...     default_joint_positions=np.zeros(7),
        ...     observation_models={
        ...         "joint_pos": GaussianChannelObservationModel(channel="joint_pos")},
        ... )
        >>> state = schema.pack({"joint_pos": np.zeros(7), "joint_vel": np.zeros(7),
        ...                      "command": np.zeros(7), "last_action": np.zeros(7)})
        >>> moved = model.sample_next_state(state, model.get_actions()[1])
        >>> bool(np.any(np.abs(schema.block(moved, "joint_pos")) > 0.1))  # the arm moved
        True
        >>> schema.block(moved, "command").tolist()  # the command is not dynamics
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    def __init__(
        self,
        state_schema: IsaacChannelSchema,
        action_presets: Sequence[ArrayLike],
        discount_factor: float,
        step_dt: float,
        tracking_gain: float,
        chain: ModifiedDHChain,
        default_joint_positions: ArrayLike,
        arm_joint_indices: Optional[ArrayLike] = None,
        actuated_indices: Optional[ArrayLike] = None,
        joint_position_channel: str = "joint_pos",
        joint_velocity_channel: str = "joint_vel",
        last_action_channel: str = "last_action",
        command_channel: str = "command",
        action_scale: float = 1.0,
        position_noise_std: float = 0.01,
        velocity_noise_std: float = 0.1,
        action_noise_std: float = 1e-3,
        reward_model: Optional[RewardModel] = None,
        observation_models: Optional[Mapping[str, IsaacObservationModel]] = None,
        raw_observation_schema: Optional[IsaacChannelSchema] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the joint-lag-driven Isaac model.

        Args:
            state_schema: Named blocks of the flat state vector.
            action_presets: Finite list of joint-target commands to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            step_dt: Control-step duration in seconds, from the live task.
            tracking_gain: Per-step lag gain; measure it with :func:`calibrate_tracking_gain`.
            chain: Kinematics from the robot base to the tracked body.
            default_joint_positions: The chain's default joint pose, in the chain's own order.
            arm_joint_indices: Which entries of the joint block feed the chain, in the chain's
                order. ``None`` means the joint block is exactly the chain's joints, in order.
            actuated_indices: Which entry of the joint block each action entry commands. ``None``
                means the action commands every observed joint in order.
            joint_position_channel: Block holding relative joint positions.
            joint_velocity_channel: Block holding relative joint velocities.
            last_action_channel: Block holding the previously applied action.
            command_channel: Block holding the commanded pose, carried through the transition.
            action_scale: Scale mapping an action into a joint-position target.
            position_noise_std: Std of the joint-position process noise, in radians.
            velocity_noise_std: Std of the joint-velocity process noise, in radians per second.
            action_noise_std: Std on the recorded-action block; small but strictly positive.
            reward_model: Objective to optimize. ``None`` (the default) builds a
                :class:`ReachRewardModel` over ``chain``, which is the point of this class —
                pass one explicitly only to score a different task.
            observation_models: ``{channel: IsaacObservationModel}``.
            raw_observation_schema: Named blocks of the world's flat raw observation.
            reward_range: Optional ``(min, max)`` reward bounds.
            name: Model name, also used to label planner output.

        Raises:
            ValueError: If the two joint channels disagree in width, or the action channel is not
                as wide as the number of commanded joints.
        """
        driven = (joint_position_channel, joint_velocity_channel, last_action_channel)
        position_width = state_schema.width(joint_position_channel)
        if state_schema.width(joint_velocity_channel) != position_width:
            raise ValueError(
                f"{joint_velocity_channel!r} is {state_schema.width(joint_velocity_channel)} wide "
                f"but {joint_position_channel!r} is {position_width}; the two joint channels "
                "describe the same joints and must agree"
            )
        self.chain = chain
        self.joint_position_channel = joint_position_channel
        self.joint_velocity_channel = joint_velocity_channel
        self.last_action_channel = last_action_channel
        self.command_channel = command_channel
        self.joint_transition = JointLagTransition(
            position_width=position_width,
            actuated_indices=actuated_indices,
            step_dt=step_dt,
            tracking_gain=tracking_gain,
            action_scale=action_scale,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std,
            action_noise_std=action_noise_std,
        )
        if state_schema.width(last_action_channel) != self.joint_transition.action_dim:
            raise ValueError(
                f"{last_action_channel!r} is {state_schema.width(last_action_channel)} wide but "
                f"{self.joint_transition.action_dim} joints are commanded"
            )
        objective = reward_model or ReachRewardModel(
            state_schema=state_schema,
            chain=chain,
            joint_position_channel=joint_position_channel,
            command_channel=command_channel,
            default_joint_positions=default_joint_positions,
            arm_joint_indices=arm_joint_indices,
        )
        self.reach_reward = objective if isinstance(objective, ReachRewardModel) else None
        super().__init__(
            state_schema=state_schema,
            action_presets=action_presets,
            discount_factor=discount_factor,
            transition=self.joint_transition,
            reward_model=objective,
            transition_channels=driven,
            observation_models=observation_models,
            raw_observation_schema=raw_observation_schema,
            reward_range=reward_range,
            name=name,
        )
