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
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType

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
) -> Any:
    """Build a registered IsaacLab task env.

    Isolated as a module-level seam so unit tests can inject a fake env without
    launching Isaac Sim. Passing ``render_mode="rgb_array"`` enables offscreen
    cameras and forwards the render mode to ``gymnasium.make`` so ``env.render()``
    yields RGB frames of the simulator viewport.
    """
    _launch_simulation_app(headless, enable_cameras=render_mode == "rgb_array")
    import gymnasium as gym  # pylint: disable=import-outside-toplevel

    # Importing the tasks package registers the ``Isaac-*`` gym ids as a side
    # effect; use importlib so there is no unused bound name.
    importlib.import_module("isaaclab_tasks")
    parse_env_cfg = importlib.import_module("isaaclab_tasks.utils").parse_env_cfg

    cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, **env_cfg_kwargs)
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
            name: Environment identifier. Defaults to ``"IsaacLabPOMDP-<task_id>"``.
            reward_range: Optional ``(min, max)`` reward bounds. Defaults to None.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            use_queue_logger: Whether to use queue-based logging. Defaults to False.

        Raises:
            ValueError: If ``num_envs`` is not 1.
        """
        if num_envs != 1:
            raise ValueError(
                "IsaacLabPOMDP is a forward-only world environment and requires "
                f"num_envs=1 (got {num_envs}); a world drives a single true trajectory."
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

        # Custom extractors are kept private so they neither participate in
        # ``config_id`` nor are picked up by ``to_dict`` parameter introspection.
        self._state_extractor = state_extractor
        self._observation_extractor = observation_extractor

        # Live-simulator state: rebuilt lazily and never serialized.
        self._env: Optional[Any] = None
        self._live_state: Optional[np.ndarray] = None
        self._terminated: bool = False
        self._seeded: bool = False
        self._pending: Optional[Dict[str, Any]] = None

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
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._env = None
        self._live_state = None
        self._terminated = False
        self._seeded = False
        self._pending = None

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
            )
        return self._env

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
        return state

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
        _, reward, terminated, truncated, _ = env.step(self._to_isaac_action(action))
        done = _scalar_bool(terminated) or _scalar_bool(truncated)
        pending = {
            "state": np.asarray(state).copy(),
            "action": action,
            "next_state": self._extract_state(env),
            "observation": self._extract_observation(env),
            "reward": _scalar_float(reward),
            "terminated": done,
        }
        self._pending = pending
        self._live_state = pending["next_state"]
        self._terminated = done
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
