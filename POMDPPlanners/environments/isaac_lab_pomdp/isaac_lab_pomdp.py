# SPDX-License-Identifier: MIT

"""IsaacLab POMDP world wrapper environment.

This module adapts a registered `IsaacLab <https://isaac-sim.github.io/IsaacLab/>`_
task (``Isaac-*-v0``) to the POMDPPlanners
:class:`~POMDPPlanners.core.environment.Environment` interface so it can serve as
the **ground-truth world** in an
:class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`.

Like a Gymnasium env, an IsaacLab env is *forward-only*: it exposes
``reset()`` / ``step(action)`` and is a black-box physics simulator with no
transition/observation density. It therefore cannot act as a planner's
generative model, and :meth:`IsaacLabPOMDP.transition_log_probability` /
:meth:`IsaacLabPOMDP.observation_log_probability` intentionally raise
:class:`NotImplementedError`. In the two-environment episode design the planner
keeps its own generative model (``policy.environment``) and this wrapper only
advances the single true state forward, one step per real interaction.

Unlike Gymnasium, IsaacLab gives a genuine ``observation = h(state)`` split: the
ground-truth robot pose/joint state is read directly from the physics engine
(``scene[asset].data``), while the observation is a simulated **sensor** buffer
(e.g. a ``RayCaster`` LiDAR). The two extractors are configurable per task.

One wrapper covers the many registered tasks because they share one entry point
(``parse_env_cfg`` + ``gymnasium.make``), one step API
(``(obs_dict, reward, terminated, truncated, info)``) and one scene accessor
(``env.unwrapped.scene[...]``). Task-specific asset/sensor names become override
hooks (``state_extractor`` / ``observation_extractor``).

Caveats:
    * **One SimulationApp per process.** IsaacLab launches a single global
      ``SimulationApp``; two IsaacLab envs cannot coexist in one process, so the
      world and ``policy.environment`` cannot *both* be IsaacLab in-process, and
      the multiprocessing task managers cannot fork the GPU simulator. Drive
      single-process :class:`EpisodeRunner` runs.
    * **num_envs must be 1.** A world drives one true trajectory; batched
      (particle) use is a model-side concern this world adapter does not cover.
    * **Density methods are unsupported** (forward-only world).

Classes:
    IsaacLabPOMDP: Forward-only adapter exposing an IsaacLab task as a world.
"""

import importlib
from enum import Enum
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction, StepInfoMetric

if TYPE_CHECKING:
    from POMDPPlanners.core.simulation import StepData


class IsaacLabStepChannel(Enum):
    """Per-step channels this world reports through ``step_info``.

    Names state the measured quantity, not a category. There is deliberately no
    generic "impact" channel: a differently-measured quantity (peak force in N,
    lost kinetic energy in J, a collision count) is a *different* channel and
    must not reuse this name, because averaging those together would be
    meaningless.
    """

    #: Whether the configured success predicate held on this step (1.0 / 0.0).
    SUCCESS = "success"
    #: Contact impulse over this control step, in newton-seconds.
    CONTACT_IMPULSE_NS = "contact_impulse_ns"
    #: Largest instantaneous contact force during this control step, in newtons.
    CONTACT_PEAK_FORCE_N = "contact_peak_force_n"


class IsaacLabMetric(Enum):
    """Metric names this world declares."""

    #: Fraction of episodes in which the configured success predicate held,
    #: under the configured per-episode reduction (``ANY`` for "goal reached at
    #: some point", ``ALL`` for "never failed").
    SUCCESS_RATE = "success_rate"
    #: Mean over episodes of the episode's largest per-step contact impulse (N*s).
    MAX_CONTACT_IMPULSE_NS = "max_contact_impulse_ns"
    #: Mean over episodes of the episode's largest instantaneous contact force (N).
    MAX_CONTACT_PEAK_FORCE_N = "max_contact_peak_force_n"


#: The metric each impact channel rolls up into. Kept as one mapping so a new impact channel
#: cannot be added without saying what it aggregates to.
IMPACT_METRIC_BY_CHANNEL = {
    IsaacLabStepChannel.CONTACT_IMPULSE_NS: IsaacLabMetric.MAX_CONTACT_IMPULSE_NS,
    IsaacLabStepChannel.CONTACT_PEAK_FORCE_N: IsaacLabMetric.MAX_CONTACT_PEAK_FORCE_N,
}


# Module-level SimulationApp handle. IsaacLab permits exactly one per process;
# guard so a second IsaacLabPOMDP reuses it instead of relaunching.
_SIMULATION_APP: Optional[Any] = None


def _launch_simulation_app(headless: bool, enable_cameras: bool = False) -> Any:
    """Launch (once per process) and return the IsaacLab ``SimulationApp``.

    ``enable_cameras`` turns on offscreen rendering so ``env.render()`` can return
    RGB frames even under a headless launch; it must be set on the *first* launch
    because IsaacLab permits exactly one ``SimulationApp`` per process.
    """
    global _SIMULATION_APP  # pylint: disable=global-statement
    if _SIMULATION_APP is None:
        # pylint: disable-next=import-outside-toplevel,import-error
        from isaaclab.app import AppLauncher

        _SIMULATION_APP = AppLauncher(headless=headless, enable_cameras=enable_cameras).app
    return _SIMULATION_APP


def _build_isaac_env(
    task_id: str,
    num_envs: int,
    device: str,
    env_cfg_kwargs: Dict[str, Any],
    headless: bool,
    render_mode: Optional[str] = None,
    env_cfg_modifier: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Build a registered IsaacLab task env.

    Isolated as a module-level seam so unit tests can inject a fake env without
    launching Isaac Sim. Passing ``render_mode="rgb_array"`` enables offscreen
    cameras and forwards the render mode to ``gymnasium.make`` so ``env.render()``
    yields RGB frames of the simulator viewport.

    ``env_cfg_modifier`` is invoked on the parsed task config before the env is
    built, which is the only point at which the scene can still be changed. It
    exists so a task that ships no contact sensor can have one attached for
    impact measurement; ``parse_env_cfg`` keyword arguments cannot add scene
    entities.
    """
    _launch_simulation_app(headless, enable_cameras=render_mode == "rgb_array")
    import gymnasium as gym  # pylint: disable=import-outside-toplevel

    # Importing the tasks package registers the ``Isaac-*`` gym ids as a side
    # effect; use importlib so there is no unused bound name.
    importlib.import_module("isaaclab_tasks")
    parse_env_cfg = importlib.import_module("isaaclab_tasks.utils").parse_env_cfg

    cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, **env_cfg_kwargs)
    if env_cfg_modifier is not None:
        env_cfg_modifier(cfg)
    if render_mode is not None:
        return gym.make(task_id, cfg=cfg, render_mode=render_mode)
    return gym.make(task_id, cfg=cfg)


def _to_numpy(value: Any) -> np.ndarray:
    """Move a torch tensor (or array-like) to a detached CPU numpy array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_float(value: Any) -> float:
    """Read a per-env scalar (shape ``(num_envs,)``) as a Python float."""
    return float(_to_numpy(value).reshape(-1)[0])


def _scalar_bool(value: Any) -> bool:
    """Read a per-env boolean (shape ``(num_envs,)``) as a Python bool."""
    return bool(_to_numpy(value).reshape(-1)[0])


def control_substeps(env: Any) -> int:
    """Physics substeps per control step, from the live simulator's timing.

    Args:
        env: The live IsaacLab env.

    Returns:
        The number of physics substeps one control step spans, at least 1. Falls
        back to 1 when the env does not report its timing.
    """
    step_dt = getattr(env.unwrapped, "step_dt", None)
    physics_dt = getattr(env.unwrapped, "physics_dt", None)
    if not step_dt or not physics_dt:
        return 1
    return max(1, int(round(float(step_dt) / float(physics_dt))))


def step_duration(env: Any) -> float:
    """Duration of one control step in seconds, or 1.0 when the env omits it."""
    step_dt = getattr(env.unwrapped, "step_dt", None)
    return float(step_dt) if step_dt is not None else 1.0


def contact_force_samples(env: Any, sensor_key: str) -> np.ndarray:
    """Contact force samples covering the current control step.

    IsaacLab documents the history buffer as newest-first: "In the history
    dimension, the first index is the most recent and the last index is the
    oldest." Only the leading :func:`control_substeps` entries belong to the step
    just taken; trailing ones belong to control steps already accounted for, so
    slicing from the end would silently report stale forces.

    Args:
        env: The live IsaacLab env.
        sensor_key: Scene key of the ``ContactSensor``.

    Returns:
        The sensor's world-frame net forces in newtons, leading axis this step's
        samples and trailing axes the sensor's own body layout. A sensor keeping
        no history contributes its single end-of-step reading, so a spike between
        substeps is invisible; widen ``history_length`` to the task's decimation
        to capture it.

    Raises:
        RuntimeError: If the sensor exposes no contact force buffer at all.
    """
    data = env.unwrapped.scene[sensor_key].data
    history = getattr(data, "net_forces_w_history", None)
    if history is not None:
        return _to_numpy(history)[0][: control_substeps(env)]

    forces = getattr(data, "net_forces_w", None)
    if forces is None:
        raise RuntimeError(
            f"No contact force buffer found on scene sensor "
            f"'{sensor_key}'; pass a custom impact_extractor."
        )
    return _to_numpy(forces)[0][np.newaxis, ...]


class IsaacLabPOMDP(Environment):
    """Forward-only adapter exposing an IsaacLab task as a world POMDP.

    The wrapper drives a registered IsaacLab task as the ground-truth world of an
    episode. It calls ``env.step(action)`` exactly once per real interaction and
    serves the resulting next state, observation and reward from a small cache,
    because the POMDPPlanners episode loop requests those three quantities through
    separate method calls while IsaacLab produces them atomically. The state is
    read from the physics engine (``scene[state_asset].data``) and the observation
    from a sensor (``scene[observation_sensor].data``), giving a genuine
    ``observation = h(state)`` split.

    Note:
        This is a *world* environment, not a generative model. It cannot sample a
        transition from an arbitrary state, so belief particle propagation and
        density queries are unsupported and raise ``NotImplementedError`` /
        ``RuntimeError``. Pair it with a generative model environment on the
        planner (``policy.environment``).

    Attributes:
        task_id: Registered IsaacLab task id passed to ``gymnasium.make``.
        num_envs: Number of parallel sim envs; must be 1 for a world adapter.
        device: Torch device the simulator runs on (e.g. ``"cuda"``).
        env_cfg_kwargs: Extra keyword arguments forwarded to ``parse_env_cfg``.
        state_asset: Scene key for the ground-truth articulation.
        observation_sensor: Scene key for the observation sensor.
        headless: Whether to launch the simulator without a GUI.
        seed: Optional seed applied to the first ``reset`` for reproducibility.

    Example:
        Constructed like a Gymnasium world but reading state and sensor buffers
        directly from the IsaacLab scene (illustrative — requires a working Isaac
        Sim install, so it is not run as a doctest):

        .. code-block:: python

            from POMDPPlanners.environments.isaac_lab_pomdp import IsaacLabPOMDP

            world = IsaacLabPOMDP(
                task_id="Isaac-Velocity-Flat-Anymal-C-v0",
                discount_factor=0.99,
                observation_sensor="lidar",
            )
            state = world.initial_state_dist().sample()[0]
            action = world.space_info  # supply a valid task action here
            next_state, observation, reward = world.sample_next_step(state, action)
            terminal = world.is_terminal(next_state)
    """

    def __init__(
        self,
        task_id: str,
        discount_factor: float,
        num_envs: int = 1,
        device: str = "cuda",
        env_cfg_kwargs: Optional[Dict[str, Any]] = None,
        state_asset: str = "robot",
        observation_sensor: str = "lidar",
        state_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        observation_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        action_space_type: SpaceType = SpaceType.CONTINUOUS,
        observation_space_type: SpaceType = SpaceType.CONTINUOUS,
        headless: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        contact_sensor_key: Optional[str] = None,
        success_termination_term: Optional[str] = None,
        success_reduction: EpisodeReduction = EpisodeReduction.ANY,
        impact_extractor: Optional[Callable[[Any], float]] = None,
        impact_channel: IsaacLabStepChannel = IsaacLabStepChannel.CONTACT_IMPULSE_NS,
        success_extractor: Optional[Callable[[Any, Dict[str, Any], bool, bool], bool]] = None,
        env_cfg_modifier: Optional[Callable[[Any], None]] = None,
        record_video: bool = False,
        name: Optional[str] = None,
        reward_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the IsaacLab world environment.

        Args:
            task_id: Registered IsaacLab task id (e.g. ``"Isaac-Reach-Franka-v0"``).
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            num_envs: Parallel sim envs; must be 1 for a forward-only world.
            device: Torch device the simulator runs on. Defaults to ``"cuda"``.
            env_cfg_kwargs: Extra keyword args for ``parse_env_cfg``. Defaults to none.
            state_asset: Scene key for the ground-truth articulation. Defaults to
                ``"robot"``.
            observation_sensor: Scene key for the observation sensor. Defaults to
                ``"lidar"``.
            state_extractor: Optional ``env -> np.ndarray`` reading the ground-truth
                state. Defaults to concatenating root pose/velocity and joint state.
            observation_extractor: Optional ``env -> np.ndarray`` reading the sensor
                observation. Defaults to the sensor's ray-hit buffer.
            action_space_type: Action space category. Defaults to CONTINUOUS.
            observation_space_type: Observation space category. Defaults to CONTINUOUS.
            headless: Launch the simulator without a GUI. Defaults to True.
            render_mode: Optional Gymnasium render mode forwarded to
                ``gymnasium.make``. Pass ``"rgb_array"`` to enable offscreen
                cameras so :meth:`render` returns RGB frames of the simulator
                viewport (for video capture). Defaults to None (no rendering).
            seed: Optional seed applied to the first ``reset``. Defaults to None.
            contact_sensor_key: Scene key of a ``ContactSensor`` to read impact
                magnitude from. ``None`` (default) disables impact measurement,
                so no ``impact`` channel is reported and no impact metric is
                declared. Most manipulation tasks ship no contact sensor; one can
                be attached through ``env_cfg_modifier``.
            success_termination_term: Name of the termination-manager term that
                marks task success. ``None`` (default) disables success
                measurement. Most IsaacLab tasks declare only *failure* and
                timeout terms and have no success term at all, in which case an
                explicit ``success_extractor`` is required — a missing term
                raises rather than being guessed at, because inferring success
                from "terminated but not truncated" would count every failure as
                a success.
            success_reduction: How per-step success collapses to one value per
                episode. ``ANY`` (default) suits a goal that is reached at some
                point. Use ``ALL`` for a "never failed" predicate such as a
                legged robot staying upright — under ``ANY`` a robot that falls
                on the final step would still be scored a success because the
                earlier steps were fine.
            impact_extractor: Optional ``env -> float`` reading impact magnitude.
                Defaults to the peak contact impulse over the sensor's bodies.
            impact_channel: Which channel the impact reading is reported under,
                and so which metric it rolls up into. An override that measures a
                *different quantity* — the peak force in newtons rather than the
                impulse in newton-seconds, say — must say so here: reporting it
                under the impulse channel would average newtons and
                newton-seconds into one number that means nothing, and nothing
                downstream could tell.

                Raises ``ValueError`` for a channel that is not an impact channel.
            success_extractor: Optional
                ``(env, info, terminated, truncated) -> bool``. Defaults to
                reading ``success_termination_term`` from the termination
                manager, falling back to ``terminated and not truncated``.
            env_cfg_modifier: Optional ``cfg -> None`` applied to the parsed task
                config before the env is built, e.g. to attach a
                ``ContactSensorCfg``. Set its ``history_length`` to the task's
                decimation, otherwise peak force is under-reported: one
                ``env.step`` covers several physics substeps and the force buffer
                is read only at the end.
            record_video: Buffer an RGB frame per step so
                :meth:`cache_visualization` can write an episode video. Requires
                ``render_mode="rgb_array"``. Defaults to False.
            name: Environment identifier. Defaults to ``"IsaacLabPOMDP-<task_id>"``.
            reward_range: Optional ``(min, max)`` reward bounds. Defaults to None.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to False.

        Raises:
            ValueError: If ``num_envs`` is not 1, or if ``record_video`` is set
                without ``render_mode="rgb_array"``.
        """
        if num_envs != 1:
            raise ValueError(
                "IsaacLabPOMDP is a forward-only world environment and requires "
                f"num_envs=1 (got {num_envs}); a world drives a single true trajectory."
            )
        if record_video and render_mode != "rgb_array":
            raise ValueError(
                "IsaacLabPOMDP(record_video=True) requires render_mode='rgb_array'; "
                "the simulator only produces RGB frames when offscreen cameras are enabled."
            )

        self.task_id = task_id
        self.num_envs = num_envs
        self.device = device
        self.env_cfg_kwargs: Dict[str, Any] = dict(env_cfg_kwargs) if env_cfg_kwargs else {}
        self.state_asset = state_asset
        self.observation_sensor = observation_sensor
        self.action_space_type = action_space_type
        self.observation_space_type = observation_space_type
        self.headless = headless
        self.render_mode = render_mode
        self.seed = seed
        self.contact_sensor_key = contact_sensor_key
        self.success_termination_term = success_termination_term
        self.success_reduction = success_reduction
        self.record_video = record_video

        # Custom extractors are kept private so they neither participate in
        # ``config_id`` nor are picked up by ``to_dict`` parameter introspection.
        self._state_extractor = state_extractor
        self._observation_extractor = observation_extractor
        self._impact_extractor = impact_extractor
        if impact_channel not in IMPACT_METRIC_BY_CHANNEL:
            raise ValueError(
                f"impact_channel must be one of "
                f"{[channel.value for channel in IMPACT_METRIC_BY_CHANNEL]}, "
                f"got {impact_channel}"
            )
        self.impact_channel = impact_channel
        self._success_extractor = success_extractor
        self._env_cfg_modifier = env_cfg_modifier

        # Live-simulator state: rebuilt lazily and never serialized.
        self._env: Optional[Any] = None
        self._live_state: Optional[np.ndarray] = None
        self._terminated: bool = False
        self._seeded: bool = False
        self._pending: Optional[Dict[str, Any]] = None
        self._frames: List[np.ndarray] = []

        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else f"IsaacLabPOMDP-{task_id}",
            space_info=SpaceInfo(
                action_space=action_space_type,
                observation_space=observation_space_type,
            ),
            reward_range=reward_range,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    # ── Serialization: drop the non-picklable live handle ───────────────
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        state["_live_state"] = None
        state["_terminated"] = False
        state["_seeded"] = False
        state["_pending"] = None
        state["_frames"] = []
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._env = None
        self._live_state = None
        self._terminated = False
        self._seeded = False
        self._pending = None
        self._frames = []

    # ── Live simulator management ───────────────────────────────────────
    def _get_env(self) -> Any:
        if self._env is None:
            self._env = _build_isaac_env(
                self.task_id,
                self.num_envs,
                self.device,
                self.env_cfg_kwargs,
                self.headless,
                self.render_mode,
                self._env_cfg_modifier,
            )
        return self._env

    @property
    def task_env(self) -> Any:
        """The underlying IsaacLab task environment.

        Exposed for the same reason as :attr:`action_space`: a caller that builds an *analytic*
        planner-side model has to read the task's own configuration first — the control-step
        duration, the action term's scale, the articulation's joint names and its default joint
        pose. Those are properties of the robot and the task, fixed before the episode starts, and
        reading them here is calibration, not privileged access to the live state.

        Returns:
            The wrapped ``gymnasium`` environment. Accessing it builds the simulator if it is not
            running yet.
        """
        return self._get_env()

    @property
    def action_space(self) -> Any:
        """The underlying task's Gymnasium action space.

        Exposed because a caller usually has to build its own discretization of
        a continuous IsaacLab task (VOPP, for instance, addresses actions by
        integer index into a fixed set) and needs the action dimension to do so
        before any stepping happens.

        Returns:
            The task's ``gymnasium`` action space. Accessing it builds the
            simulator if it is not running yet.
        """
        return self._get_env().action_space

    def render(self) -> np.ndarray:
        """Return the current simulator viewport as an RGB frame.

        Renders what the simulator sees (``env.render()``) as an ``(H, W, 3)``
        ``uint8`` array, suitable for assembling into a video with
        :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_visualizer.IsaacLabPOMDPVisualizer`.

        Returns:
            An ``(H, W, 3)`` ``uint8`` RGB frame of the simulator viewport.

        Raises:
            RuntimeError: If the world was not constructed with
                ``render_mode="rgb_array"`` or the simulator returns no frame.
        """
        if self.render_mode != "rgb_array":
            raise RuntimeError(
                "IsaacLabPOMDP.render requires render_mode='rgb_array'; construct "
                "the world with render_mode='rgb_array' to capture RGB frames."
            )
        frame = self._get_env().render()
        if frame is None:
            raise RuntimeError(
                "IsaacLab returned no RGB frame; ensure the simulator launched "
                "with cameras enabled (render_mode='rgb_array')."
            )
        return _to_numpy(frame).astype(np.uint8)

    def _reset(self) -> np.ndarray:
        env = self._get_env()
        if self.seed is not None and not self._seeded:
            env.reset(seed=self.seed)
            self._seeded = True
        else:
            env.reset()
        state = self._extract_state(env)
        self._live_state = state
        self._terminated = False
        self._pending = None
        self._frames = []
        self._capture_frame()
        return state

    def _capture_frame(self) -> None:
        if self.record_video:
            self._frames.append(self.render())

    def _extract_state(self, env: Any) -> np.ndarray:
        if self._state_extractor is not None:
            return np.asarray(self._state_extractor(env))
        return self._default_state_extractor(env)

    def _extract_observation(self, env: Any) -> np.ndarray:
        if self._observation_extractor is not None:
            return np.asarray(self._observation_extractor(env))
        return self._default_observation_extractor(env)

    def _default_state_extractor(self, env: Any) -> np.ndarray:
        """Read ground-truth state from the articulation's physics buffers."""
        data = env.unwrapped.scene[self.state_asset].data
        components = (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        )
        parts = [
            _to_numpy(getattr(data, attr))[0].reshape(-1)
            for attr in components
            if getattr(data, attr, None) is not None
        ]
        if not parts:
            raise RuntimeError(
                f"No ground-truth state fields found on scene asset "
                f"'{self.state_asset}'; pass a custom state_extractor."
            )
        return np.concatenate(parts)

    def _default_observation_extractor(self, env: Any) -> np.ndarray:
        """Read the sensor observation buffer (e.g. RayCaster LiDAR hits)."""
        data = env.unwrapped.scene[self.observation_sensor].data
        for attr in ("ray_hits_w", "distances", "output"):
            value = getattr(data, attr, None)
            if value is not None:
                return _to_numpy(value)[0].reshape(-1)
        raise RuntimeError(
            f"No sensor observation buffer found on scene sensor "
            f"'{self.observation_sensor}'; pass a custom observation_extractor."
        )

    # ── Impact / success measurement ────────────────────────────────────
    def _extract_impact(self, env: Any) -> Optional[float]:
        if self._impact_extractor is not None:
            return float(self._impact_extractor(env))
        if self.contact_sensor_key is None:
            return None
        return self._default_impact_extractor(env)

    def _default_impact_extractor(self, env: Any) -> float:
        """Read the worst body's contact impulse (N*s) over this control step.

        The impulse is estimated as ``mean(|F|) * step_dt`` over the force
        samples belonging to *this* step, which keeps the measurement a property
        of the transition rather than of the sensor's configuration:

        * The history buffer is sliced to the step's decimation. Summing the
          whole buffer instead would fold in forces from previous control steps
          when ``history_length`` exceeds the decimation, and drop forces when it
          is shorter — making the number depend on how the sensor was set up.
        * Averaging then scaling by ``step_dt`` reduces exactly to
          ``sum(|F|) * physics_dt`` when a full step of substeps is available,
          and degrades gracefully to the single end-of-step reading when the
          sensor keeps no history at all.

        The maximum is taken over bodies only: an episode is characterized by its
        worst contact point, not by the average across the robot.
        """
        sensor_key = self.contact_sensor_key
        if sensor_key is None:
            raise RuntimeError(
                "The default impact extractor needs a contact sensor; set contact_sensor_key "
                "or pass an impact_extractor."
            )
        samples = contact_force_samples(env, sensor_key)
        # (samples, bodies, 3) -> magnitude per body per sample.
        magnitudes = np.linalg.norm(samples.reshape(samples.shape[0], -1, 3), axis=-1)
        # Mean force (N) times the control-step duration gives an impulse (N*s),
        # comparable across tasks with different control rates.
        return float(magnitudes.mean(axis=0).max()) * step_duration(env)

    def _extract_success(
        self, env: Any, info: Dict[str, Any], terminated: bool, truncated: bool
    ) -> Optional[bool]:
        if self._success_extractor is not None:
            return bool(self._success_extractor(env, info, terminated, truncated))
        if self.success_termination_term is None:
            return None
        del terminated, truncated
        return self._default_success_extractor(env)

    def _default_success_extractor(self, env: Any) -> bool:
        """Read the configured success term from the termination manager.

        There is deliberately no fallback. Inferring success from
        ``terminated and not truncated`` looks reasonable but is wrong for most
        IsaacLab tasks: they terminate on *failure* (a robot falling, a cart
        leaving its bounds) and only truncate on timeout, so that rule reports
        every failure as a success and silently inverts the metric. A
        misconfigured term name must fail loudly instead.

        Raises:
            RuntimeError: If the task exposes no termination manager, or no term
                by the configured name.
        """
        manager = getattr(env.unwrapped, "termination_manager", None)
        getter = getattr(manager, "get_term", None)
        if getter is None:
            raise RuntimeError(
                f"success_termination_term='{self.success_termination_term}' was configured "
                "but this task exposes no termination manager; pass a custom success_extractor."
            )
        try:
            return _scalar_bool(getter(self.success_termination_term))
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                f"Termination term '{self.success_termination_term}' not found on task "
                f"'{self.task_id}'. Most IsaacLab tasks declare only failure and timeout "
                "terms, so success usually needs an explicit success_extractor."
            ) from error

    def _to_isaac_action(self, action: Any) -> Any:
        import torch  # pylint: disable=import-outside-toplevel

        if self.space_info.action_space == SpaceType.DISCRETE:
            return torch.as_tensor([[int(action)]], device=self.device)
        tensor = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _states_equal(self, state_a: Any, state_b: Any) -> bool:
        return np.array_equal(np.asarray(state_a), np.asarray(state_b))

    def _ensure_stepped(self, state: Any, action: Any) -> Dict[str, Any]:
        """Advance the world one step for ``(state, action)`` (once) and cache it.

        Serves a pending cache when the same ``(state, action)`` is requested by
        the separate reward / next-state / observation calls of a single episode
        step. Raises when asked to step from a state other than the live one.
        """
        pending = self._pending
        if (
            pending is not None
            and self._states_equal(pending["state"], state)
            and self.hash_action(pending["action"]) == self.hash_action(action)
        ):
            return pending

        if self._live_state is None or not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "IsaacLabPOMDP is a forward-only world environment; it cannot "
                "resample from an arbitrary state. Give the planner a separate "
                "model environment (policy.environment) and only step the world "
                "forward from its live state."
            )

        env = self._get_env()
        _, reward, terminated, truncated, info = env.step(self._to_isaac_action(action))
        # Keep terminated and truncated apart, not just their disjunction: the
        # difference between "the task ended" and "the clock ran out" is exactly
        # the task-completion signal.
        is_terminated = _scalar_bool(terminated)
        is_truncated = _scalar_bool(truncated)
        done = is_terminated or is_truncated
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": self._extract_state(env),
            "observation": self._extract_observation(env),
            "reward": _scalar_float(reward),
            "terminated": done,
            "is_terminated": is_terminated,
            "is_truncated": is_truncated,
            "impact": self._extract_impact(env),
            "success": self._extract_success(env, info or {}, is_terminated, is_truncated),
        }
        self._pending = pending
        self._live_state = pending["next_state"]
        self._terminated = done
        self._capture_frame()
        return pending

    # ── Environment interface ───────────────────────────────────────────
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        if n_samples != 1:
            raise ValueError("IsaacLabPOMDP is forward-only and only supports n_samples=1")
        return self._ensure_stepped(state, action)["next_state"]

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        del action
        if n_samples != 1:
            raise ValueError("IsaacLabPOMDP is forward-only and only supports n_samples=1")
        if self._pending is not None and self._states_equal(
            self._pending["next_state"], next_state
        ):
            return self._pending["observation"]
        raise RuntimeError(
            "IsaacLabPOMDP.sample_observation was queried for a next state other "
            "than the one just produced by the live step; a forward-only sensor "
            "world cannot synthesize an observation for an arbitrary state."
        )

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del next_state
        return self._ensure_stepped(state, action)["reward"]

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report impact severity and task success for the step just taken.

        Values are served from the cache filled by the single ``env.step`` of
        this interaction — a forward-only world cannot re-measure a transition,
        so the arguments are used only to confirm the request refers to it.

        Args:
            state: The state the step was taken from.
            action: The action taken.
            next_state: The realised successor state.

        Returns:
            A mapping that may contain ``"impact"`` (peak contact impulse, N*s)
            and ``"success"`` (1.0 / 0.0). A channel is absent when this world was
            not configured to measure it, so that an unmeasured quantity is never
            confused with a measured zero.
        """
        del state, action
        pending = self._pending
        if pending is None or not self._states_equal(pending["next_state"], next_state):
            return {}

        info: Dict[str, float] = {}
        if pending["impact"] is not None:
            info[self.impact_channel.value] = float(pending["impact"])
        if pending["success"] is not None:
            info[IsaacLabStepChannel.SUCCESS.value] = float(pending["success"])
        return info

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the metrics this world's configured measurements support.

        Only channels this world actually emits are declared, so the declared
        names always match the produced ones. A task with no contact sensor
        reports no impact metric rather than a fabricated zero.

        Returns:
            Specs for the success rate and/or the impact metric matching the
            configured ``impact_channel``, depending on which measurements were
            configured.
        """
        specs: List[StepInfoMetric] = []
        if self.success_termination_term is not None or self._success_extractor is not None:
            specs.append(
                StepInfoMetric(
                    name=IsaacLabMetric.SUCCESS_RATE.value,
                    channel=IsaacLabStepChannel.SUCCESS.value,
                    per_episode=self.success_reduction,
                )
            )
        if self.contact_sensor_key is not None or self._impact_extractor is not None:
            specs.append(
                StepInfoMetric(
                    name=IMPACT_METRIC_BY_CHANNEL[self.impact_channel].value,
                    channel=self.impact_channel.value,
                    per_episode=EpisodeReduction.MAX,
                )
            )
        return specs

    def cache_visualization(
        self, history: "List[StepData]", output_dir: Path, episode_index: int
    ) -> None:
        """Write the buffered simulator frames as an episode video.

        Args:
            history: Unused. Frames are captured live during the episode because
                the simulator viewport cannot be reconstructed from recorded
                states.
            output_dir: Directory to write into.
            episode_index: Zero-based episode index, used to name the file.

        Raises:
            RuntimeError: If the world was not constructed with
                ``record_video=True``.
        """
        del history
        if not self.record_video:
            raise RuntimeError(
                "IsaacLabPOMDP.cache_visualization requires record_video=True; "
                "construct the world with record_video=True (and "
                "render_mode='rgb_array') to capture an episode video."
            )
        if not self._frames:
            self.logger.warning("No frames buffered for episode %s; skipping video", episode_index)
            return

        # Imported lazily: the visualizer module is only needed when a video is
        # actually written, and importing it pulls in matplotlib.
        # pylint: disable-next=import-outside-toplevel
        from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_visualizer import (
            IsaacLabPOMDPVisualizer,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / f"agent_path_{episode_index}.mp4"
        IsaacLabPOMDPVisualizer(self).frames_to_video(list(self._frames), cache_path)
        self.logger.info("Cached episode video to %s", cache_path)

    @property
    def frames(self) -> List[np.ndarray]:
        """The RGB frames buffered for the current episode.

        Returns:
            The frames captured since the last reset. Empty when
            ``record_video`` is False.
        """
        return self._frames

    def is_terminal(self, state: Any) -> bool:
        if self._live_state is not None and not self._states_equal(state, self._live_state):
            raise RuntimeError(
                "IsaacLabPOMDP.is_terminal was queried for a state other than the "
                "live world state; this forward-only world only knows whether its "
                "current state is terminal."
            )
        return self._terminated

    def initial_state_dist(self) -> Distribution:
        parent = self

        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                # pylint: disable=protected-access
                return [parent._reset() for _ in range(n_samples)]

        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        parent = self

        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                # pylint: disable=protected-access
                if parent._live_state is None:
                    parent._reset()
                observation = parent._extract_observation(parent._get_env())
                return [np.asarray(observation) for _ in range(n_samples)]

        return InitialObservation()

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(np.asarray(observation1), np.asarray(observation2))

    def hash_observation(self, observation: Any) -> Hashable:
        return np.asarray(observation).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        array = np.asarray(action)
        if array.ndim == 0:
            return action  # scalar (e.g. discrete int) is already hashable
        return array.tobytes()

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action, next_states
        raise NotImplementedError(
            "IsaacLabPOMDP is a forward-only world environment with no transition "
            "density. Belief updates must run on the planner's model environment."
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del next_state, action, observations
        raise NotImplementedError(
            "IsaacLabPOMDP is a forward-only world environment with no observation "
            "density. Belief updates must run on the planner's model environment."
        )
