# SPDX-License-Identifier: MIT

"""The IsaacLab tasks under study and how each one's success is measured.

None of these tasks ships a "success" termination term — probing them finds only failure and
timeout terms (``base_contact``, ``cart_out_of_bounds``, ``time_out``). Task completion is
therefore defined here, per task, by a predicate injected into the world. That is the split the
metrics design intends: the *measurement* is environment-specific, while the channel name, the
aggregation and the confidence intervals are shared.

Every threshold in this module is a judgement call, because no task supplies one. Each probe
therefore records the whole per-step series and reports the episode's decisive extreme beside the
boolean, so a reader can apply their own threshold instead of taking the one chosen here.

Classes:
    TaskSpec: One task and the per-task measurement configuration it needs.
    ThresholdSuccessProbe: A "stayed under a threshold" predicate that keeps what it measured.
    DistanceSuccessProbe: A "got within a threshold distance of the goal" predicate.
    ReachSuccessProbe: Reach success, on the end effector's distance to its commanded pose.
    NavigationSuccessProbe: Navigation success, on the base's distance to its commanded position.
    CartpoleUprightProbe: Cartpole success, on the pole staying up and the cart staying in bounds.

Functions:
    policy_observation: Read the agent's ``policy`` observation group.
    make_success_extractor: Build a task's success predicate.
    make_contact_sensor_injector: Attach a contact sensor to a task that ships none.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_helpers import first_row


@dataclass
class TaskSpec:
    """One IsaacLab task and the per-task measurement configuration it needs.

    Attributes:
        task_id: Registered IsaacLab task id.
        contact_sensor_key: Scene key holding the contact sensor to read.
        contact_body_regex: Body pattern for an injected sensor; ``None`` when the
            task already ships one.
        success_kind: Which success predicate to build for this task.
        success_reduction: How per-step success collapses per episode. A
            "never failed" predicate needs ``ALL``; a "goal reached" predicate
            needs ``ANY``.
        success_threshold: Threshold the task's own predicate compares against — a distance in
            metres for ``reach`` and ``navigate``, an angle in radians for ``upright``.
        ee_body: End-effector body name for the ``reach`` predicate.
        model_kind: Which planner-side model to build — ``"linear"`` for the ridge-fitted
            linear-Gaussian surrogate, ``"manipulator"`` for the analytic joint-lag model with
            forward kinematics and the task's own reach objective, ``"navigation"`` for the
            goal-relative base model with the task's own pose-tracking objective.
    """

    task_id: str
    contact_sensor_key: str
    contact_body_regex: Optional[str] = None
    success_kind: str = "no_failure"
    failure_term: Optional[str] = None
    success_reduction: str = "all"
    success_threshold: float = 0.1
    ee_body: str = ""
    model_kind: str = "linear"


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="Isaac-Velocity-Flat-Anymal-C-v0",
        contact_sensor_key="contact_forces",
        success_kind="no_failure",
        failure_term="base_contact",
    ),
    TaskSpec(
        task_id="Isaac-Reach-Franka-v0",
        contact_sensor_key="injected_contacts",
        contact_body_regex="panda_hand",
        success_kind="reach",
        success_reduction="any",
        success_threshold=0.15,
        ee_body="panda_hand",
        model_kind="manipulator",
    ),
    TaskSpec(
        task_id="Isaac-Cartpole-v0",
        contact_sensor_key="injected_contacts",
        contact_body_regex="pole",
        success_kind="upright",
        failure_term="cart_out_of_bounds",
        # A right angle: the task resets the pole uniformly in +/-45deg, so a tighter threshold
        # would fail episodes on their first step whatever the policy does. See
        # CartpoleUprightProbe for the full argument.
        success_threshold=float(np.pi / 2.0),
    ),
    TaskSpec(
        task_id="Isaac-Navigation-Flat-Anymal-C-v0",
        contact_sensor_key="contact_forces",
        success_kind="navigate",
        success_reduction="any",
        # The task declares no arrival radius of its own. Half a metre is about half the robot's
        # body length and a fifth of a typical commanded distance, so it reads as "the base is at
        # the goal" without being a threshold the actuation cannot resolve at 5 Hz.
        success_threshold=0.5,
        model_kind="navigation",
    ),
]

#: Which planner-side model each task's own success predicate admits. A model applied to the wrong
#: task fails deep inside a manager read, after a SimulationApp has already claimed several GB.
MODEL_KIND_SUCCESS_KIND = {"manipulator": "reach", "navigation": "navigate"}


# ── World-side helpers ──────────────────────────────────────────────────


def policy_observation(env: Any) -> np.ndarray:
    """Read the agent's ``policy`` observation group (partial, sensor-derived).

    Used as both the state and the observation extractor so the planner-side
    vectorized model, which requires equal state and observation dimensions, has
    a single consistent space to work in.
    """
    manager = getattr(env.unwrapped, "observation_manager", None)
    if manager is not None:
        return first_row(manager.compute_group("policy"))
    return first_row(env.unwrapped.obs_buf)


def make_contact_sensor_injector(body_regex: str) -> Callable[[Any], None]:
    """Build an ``env_cfg_modifier`` attaching a contact sensor to the robot.

    Args:
        body_regex: Body-name pattern under the robot prim to sense.

    Returns:
        A callable mutating a parsed task config in place.
    """

    def _inject(cfg: Any) -> None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from isaaclab.sensors import ContactSensorCfg

        # Without activating the reporter API on the asset's bodies the sensor
        # raises "could not find any bodies with contact reporter API".
        spawn = getattr(cfg.scene.robot, "spawn", None)
        if spawn is not None:
            spawn.activate_contact_sensors = True
        # history_length lets the peak force between control steps be seen; one
        # env.step spans several physics substeps.
        cfg.scene.injected_contacts = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_regex}",
            history_length=3,
            track_air_time=False,
        )

    return _inject


def _term_is_set(env: Any, term_name: str) -> bool:
    manager = getattr(env.unwrapped, "termination_manager", None)
    getter = getattr(manager, "get_term", None)
    if getter is None:
        return False
    try:
        return bool(first_row(getter(term_name))[0])
    except (KeyError, ValueError, IndexError):
        return False


def _require_term_is_set(env: Any, term_name: str) -> bool:
    """Read a termination term, raising rather than reporting ``False`` when it is missing.

    :func:`_term_is_set` treats an absent term as "did not fire", which is the right reading for a
    predicate that merely wants to know whether a failure happened. It is the wrong reading for a
    predicate built *out of* that term: a renamed term would silently drop half the test and the
    success rate would climb, which looks like a result rather than a bug. That is the failure this
    whole predicate exists to prevent, so it must not reappear in the predicate itself.

    Raises:
        RuntimeError: If the task exposes no termination manager or no term by that name.
    """
    manager = getattr(env.unwrapped, "termination_manager", None)
    getter = getattr(manager, "get_term", None)
    if getter is None:
        raise RuntimeError(
            f"the success predicate needs the '{term_name}' termination term, but this task "
            "exposes no termination manager"
        )
    try:
        return bool(first_row(getter(term_name))[0])
    except (KeyError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"the success predicate needs the '{term_name}' termination term, which this task "
            "does not declare; the term has been renamed or removed"
        ) from error


class ThresholdSuccessProbe(ABC):
    """A "stayed under a threshold" predicate that also keeps the values it measured.

    A bare boolean says the robot did not succeed but not by how much, and those are different
    diagnoses: 0.6 m against a 0.5 m arrival radius means the action set is slightly too coarse,
    2.3 m means the planner is not steering at all. The values are free -- the predicate already
    computes them every step. It also records how each episode ended, because "ran out of time"
    and "fell over" are different failures and the success rate alone cannot tell them apart.

    Recording the whole series is what makes a chosen threshold auditable. None of these tasks
    ships one, so every threshold here is a judgement call; publishing the distribution beside the
    rate lets a reader apply their own instead of taking mine.

    Attributes:
        threshold: Value at or below which a step counts as a success.
        measurements: The measured value for every step since the last :meth:`reset`.
        terminated: Whether the last step reported a task termination (a failure term).
        truncated: Whether the last step reported a timeout.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    def __init__(self, threshold: float) -> None:
        """Initialize the probe.

        Args:
            threshold: Success threshold, in the units the subclass measures.
        """
        self.threshold = threshold
        self.measurements: List[float] = []
        self.terminated = False
        self.truncated = False

    def reset(self) -> None:
        """Forget the measurements and end-of-episode flags recorded so far."""
        self.measurements = []
        self.terminated = False
        self.truncated = False

    @abstractmethod
    def measure(self, env: Any) -> float:
        """The quantity this predicate thresholds, for the step just taken."""

    @abstractmethod
    def episode_summary(self) -> Dict[str, Any]:
        """The per-episode extremes of :attr:`measurements`, named and in their own units."""

    def summary(self) -> Dict[str, Any]:
        """Per-episode diagnostics to fold into the episode summary."""
        if not self.measurements:
            return {}
        return {
            **self.episode_summary(),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def __call__(self, env: Any, info: Dict[str, Any], terminated: bool, truncated: bool) -> bool:
        del info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        measured = self.measure(env)
        self.measurements.append(measured)
        return measured <= self.threshold


class DistanceSuccessProbe(ThresholdSuccessProbe):
    """A "got within a threshold distance of the goal" predicate.

    Reduced with ``ANY``, so the episode's *closest* approach is what decides it and is therefore
    the number worth reporting.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    #: Prefix of the summary keys this probe reports, so two tasks' distances are never averaged.
    distance_label = "goal"

    def episode_summary(self) -> Dict[str, Any]:
        return {
            f"min_{self.distance_label}_distance_m": round(min(self.measurements), 4),
            f"final_{self.distance_label}_distance_m": round(self.measurements[-1], 4),
        }


class ReachSuccessProbe(DistanceSuccessProbe):
    """Reach success: the end effector is within a threshold of its commanded pose.

    Attributes:
        ee_body: Body whose distance to the command is measured.
    """

    distance_label = "reach"

    def __init__(self, ee_body: str, threshold: float) -> None:
        """Initialize the probe.

        Args:
            ee_body: Body whose distance to the command is measured.
            threshold: Success distance, in metres.
        """
        super().__init__(threshold)
        self.ee_body = ee_body

    def measure(self, env: Any) -> float:
        return _reach_distance(env, self.ee_body)


class NavigationSuccessProbe(DistanceSuccessProbe):
    """Navigation success: the base is within a threshold of its commanded position.

    The distance is read out of the task's own ``policy`` observation rather than off the
    articulation's world pose. That is not fastidiousness: the observation is the only thing the
    planner sees, so scoring it on the same quantity is what keeps the measurement honest -- and
    the observation genuinely carries it, as the goal arrives already expressed in the base frame.
    """

    def measure(self, env: Any) -> float:
        goal = policy_observation(env)[_navigation_command_slice()]
        return float(np.linalg.norm(goal[:2]))


def _navigation_command_slice() -> slice:
    """Where the base-frame pose command sits in the navigation task's policy observation."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        navigation_state_schema,
    )

    return navigation_state_schema().slice_of("pose_command")


class CartpoleUprightProbe(ThresholdSuccessProbe):
    """Cartpole success: the pole never fell past the threshold and the cart never left its bounds.

    **Why the task's own terms are not enough.** ``Isaac-Cartpole-v0`` declares exactly one failure
    term, ``cart_out_of_bounds``, and leaves the pole angle unconstrained. Scored on that alone a
    policy that does nothing while the pole swings through a full rotation records a perfect 1.0,
    so the metric measures whether the cart drifted three metres and nothing about balancing. A
    predicate that cannot fail the do-nothing policy cannot support any claim about a planner.

    **Why the threshold is a right angle.** The task resets the pole uniformly in ``+/-45deg``, so
    any threshold below 45 degrees fails a share of episodes on their first step whatever the
    policy does -- that measures the reset draw, not the controller. The threshold must clear the
    reset range, which leaves the right angle as the one principled choice above it: past
    horizontal the pole is falling rather than being balanced, and there is no tuning freedom in
    picking it. The per-episode worst angle is reported beside the rate so a reader can apply a
    stricter one.

    The angle is read from the task's own ``policy`` observation, which reports joint positions
    *relative to the default pose* -- and the default pole position is upright, so the observation
    entry is already the deviation from vertical. The cart bound is left to the task's own
    termination term rather than re-derived here.

    Attributes:
        pole_joint: Name of the revolute joint whose angle is thresholded.
    """

    def __init__(self, threshold: float, pole_joint: str = "cart_to_pole") -> None:
        """Initialize the probe.

        Args:
            threshold: Largest tolerated absolute pole angle from vertical, in radians.
            pole_joint: Name of the pole joint in the articulation.
        """
        super().__init__(threshold)
        self.pole_joint = pole_joint

    def measure(self, env: Any) -> float:
        index = list(env.unwrapped.scene["robot"].joint_names).index(self.pole_joint)
        return float(abs(policy_observation(env)[index]))

    def episode_summary(self) -> Dict[str, Any]:
        return {
            "max_pole_angle_rad": round(max(self.measurements), 4),
            "max_pole_angle_deg": round(float(np.degrees(max(self.measurements))), 2),
        }

    def __call__(self, env: Any, info: Dict[str, Any], terminated: bool, truncated: bool) -> bool:
        upright = super().__call__(env, info, terminated, truncated)
        return upright and not _require_term_is_set(env, "cart_out_of_bounds")


def make_success_extractor(spec: TaskSpec) -> Callable[[Any, Dict[str, Any], bool, bool], bool]:
    """Build the task's success predicate.

    None of these tasks declares a success termination term, so completion is
    defined here: locomotion and cartpole succeed by *not* failing, the reach
    task succeeds when the end effector is within a threshold of its commanded
    pose, and the navigation task when the base is.

    Args:
        spec: The task configuration.

    Returns:
        A ``(env, info, terminated, truncated) -> bool`` predicate.
    """
    if spec.success_kind == "reach":
        return ReachSuccessProbe(spec.ee_body, spec.success_threshold)
    if spec.success_kind == "navigate":
        return NavigationSuccessProbe(spec.success_threshold)
    if spec.success_kind == "upright":
        return CartpoleUprightProbe(spec.success_threshold)

    failure_term = spec.failure_term or ""

    def _no_failure(env: Any, info: Dict[str, Any], terminated: bool, truncated: bool) -> bool:
        del info, terminated, truncated
        return not _term_is_set(env, failure_term)

    return _no_failure


def _reach_distance(env: Any, ee_body: str) -> float:
    """Distance from the end effector to its commanded pose, in metres."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    command = env.unwrapped.command_manager.get_command("ee_pose")
    goal_in_base = first_row(command)[:3]
    root_pos = first_row(robot.data.root_pos_w)[:3]
    body_index = list(robot.body_names).index(ee_body)
    ee_pos = np.asarray(robot.data.body_pos_w.detach().cpu().numpy())[0, body_index, :3]
    return float(np.linalg.norm(ee_pos - (root_pos + goal_in_base)))
