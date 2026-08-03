# SPDX-License-Identifier: MIT

"""Plan on IsaacLab worlds with VOPP and report impact severity / task completion.

This is the validation run for the generic per-step metrics channel. It exercises
the whole path end to end on a real physics simulator:

1. :class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP`
   measures impact and task success **at step time**, where the contact forces and
   termination terms actually exist, and reports them through ``step_info``.
2. :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp_episode_runner.VOPPEpisodeRunner`
   carries those measurements out of the episode in ``step_infos``.
3. :func:`~POMDPPlanners.core.simulation.step_info_metrics.aggregate_step_info_metrics`
   turns them into metrics with 95% confidence intervals.

**Why the success predicate is per task.** None of the three tasks ships a
"success" termination term — probing them shows only failure and timeout terms
(``base_contact``, ``cart_out_of_bounds``, ``time_out``). Task completion is
therefore defined per task by an injected ``success_extractor``, which is exactly
the split the design intends: the *measurement* is environment-specific, the
*channel name*, the aggregation and the confidence intervals are shared.

**Why a contact sensor is injected.** Only the locomotion task ships one. For the
other two an ``env_cfg_modifier`` attaches a ``ContactSensorCfg`` and switches on
the asset's contact-reporter API, without which the sensor finds no bodies.

**Planner-side model.** IsaacLab's dynamics and reward are not analytic, so each
task first runs a short warm-up of random actions, then fits a
``LinearGaussianTransition`` and a ``LinearRewardModel`` from the collected
samples. These are crude first-order system-identification baselines: they steer
behaviour but do not solve locomotion. The metrics are the deliverable here, not
the control quality.

Because IsaacLab launches a single global ``SimulationApp`` per process, each task
runs in its own child process (each rendering Isaac Sim uses ~4.25 GB of GPU
memory). The child is launched through this script's own CLI, so its command line
names the script and the task and it can be found and killed like any other
process -- unlike a ``multiprocessing``/loky worker, whose argv hides it and which
is therefore easy to orphan while it still holds GPU memory.

Run it with any interpreter, including the plain project venv::

    python isaac_vopp_metrics_example.py

If ``isaaclab`` is not importable, the script re-executes itself under the Isaac
env's Python (``ISAAC_PYTHON``) and accepts the Omniverse EULA.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import (
    IsaacLabMetric,
    IsaacLabStepChannel,
)

ISAAC_PYTHON = os.environ.get(
    "ISAAC_PYTHON", "/home/kobi/Documents/tmp/isaac_sim_playground/env_isaacsim/bin/python"
)

GAMMA = 0.99
VIDEO_FPS = 10
WARMUP_TRANSITIONS = 200
NUM_ACTION_PRESETS = 8
OBSERVATION_NOISE_STD = 0.1
BELIEF_PARTICLES = 256
PLANNING_PARTICLES = 128
PLANNING_DEPTH = 8
PLANNING_ITERATIONS = 16
OBSERVATION_RESOLUTION = 5.0
DEFAULT_EPISODES = 3
DEFAULT_STEPS = 40
# A rendering Isaac Sim needs ~4-6 GB; anything at this scale is a real workload,
# not desktop compositing.
GPU_BUSY_THRESHOLD_MB = 2000.0


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
        success_threshold: Distance threshold for the ``reach`` predicate.
        ee_body: End-effector body name for the ``reach`` predicate.
    """

    task_id: str
    contact_sensor_key: str
    contact_body_regex: Optional[str] = None
    success_kind: str = "no_failure"
    failure_term: Optional[str] = None
    success_reduction: str = "all"
    success_threshold: float = 0.1
    ee_body: str = ""


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
    ),
    TaskSpec(
        task_id="Isaac-Cartpole-v0",
        contact_sensor_key="injected_contacts",
        contact_body_regex="pole",
        success_kind="no_failure",
        failure_term="cart_out_of_bounds",
    ),
]


# ── Isaac interpreter bootstrap ─────────────────────────────────────────


def _reexec_under_isaac_if_needed() -> None:
    """Re-run this script under the Isaac interpreter when isaaclab is missing."""
    if importlib.util.find_spec("isaaclab") is not None:
        return
    if os.environ.get("_ISAAC_REEXEC") == "1":
        sys.exit(
            f"ERROR: 'isaaclab' is not importable even under ISAAC_PYTHON ({ISAAC_PYTHON}). "
            "Install Isaac Sim + IsaacLab and this project ('pip install -e .') into that env."
        )
    if not os.path.exists(ISAAC_PYTHON):
        sys.exit(
            f"ERROR: Isaac interpreter not found at {ISAAC_PYTHON}. "
            "Set the ISAAC_PYTHON env var to the Python of your Isaac Sim env."
        )
    os.environ["_ISAAC_REEXEC"] = "1"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.execv(ISAAC_PYTHON, [ISAAC_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])


# ── World-side helpers ──────────────────────────────────────────────────


def _first_row(value: Any) -> np.ndarray:
    """Detach a torch tensor (or array-like) to the first environment's row."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)[0].reshape(-1)


def _policy_observation(env: Any) -> np.ndarray:
    """Read the agent's ``policy`` observation group (partial, sensor-derived).

    Used as both the state and the observation extractor so the planner-side
    vectorized model, which requires equal state and observation dimensions, has
    a single consistent space to work in.
    """
    manager = getattr(env.unwrapped, "observation_manager", None)
    if manager is not None:
        return _first_row(manager.compute_group("policy"))
    return _first_row(env.unwrapped.obs_buf)


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
        return bool(_first_row(getter(term_name))[0])
    except (KeyError, ValueError, IndexError):
        return False


def make_success_extractor(spec: TaskSpec) -> Callable[[Any, Dict[str, Any], bool, bool], bool]:
    """Build the task's success predicate.

    None of these tasks declares a success termination term, so completion is
    defined here: locomotion and cartpole succeed by *not* failing, and the reach
    task succeeds when the end effector is within a threshold of its commanded
    pose.

    Args:
        spec: The task configuration.

    Returns:
        A ``(env, info, terminated, truncated) -> bool`` predicate.
    """
    if spec.success_kind == "reach":

        def _reach_success(env: Any, info: Dict[str, Any], terminated: bool, truncated: bool):
            del info, terminated, truncated
            return _reach_distance(env, spec.ee_body) <= spec.success_threshold

        return _reach_success

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
    goal_in_base = _first_row(command)[:3]
    root_pos = _first_row(robot.data.root_pos_w)[:3]
    body_index = list(robot.body_names).index(ee_body)
    ee_pos = np.asarray(robot.data.body_pos_w.detach().cpu().numpy())[0, body_index, :3]
    return float(np.linalg.norm(ee_pos - (root_pos + goal_in_base)))


def build_action_presets(action_dim: int, num_presets: int, seed: int) -> np.ndarray:
    """Build a finite representative action set: the zero action plus samples.

    VOPP addresses actions by integer index into a fixed set, so a continuous
    task needs an explicit discretization.

    Args:
        action_dim: Dimension of the task's continuous action vector.
        num_presets: Total number of presets, including the zero action.
        seed: Seed for the random presets.

    Returns:
        A ``(num_presets, action_dim)`` float32 array.
    """
    rng = np.random.default_rng(seed)
    presets = [np.zeros(action_dim, dtype=np.float32)]
    for _ in range(max(0, num_presets - 1)):
        presets.append(rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32))
    return np.asarray(presets, dtype=np.float32)


class WorldDriver:
    """Adapts the forward-only IsaacLab world to VOPP's tensor hooks.

    VOPP hands its hooks torch tensors, but the world is forward-only: it only
    accepts the exact numpy state it last produced. Round-tripping a state
    through torch risks breaking that identity, so this driver keeps the
    authoritative numpy state itself and ignores the tensor it is handed.

    Attributes:
        world: The live IsaacLab world.
        action_presets: The action vectors VOPP's integer indices address.
    """

    def __init__(self, world: Any, action_presets: np.ndarray, device: Any) -> None:
        self.world = world
        self.action_presets = action_presets
        self._device = device
        self._state: Optional[np.ndarray] = None
        self._previous_state: Optional[np.ndarray] = None
        self._action: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        """Reset the world and return the fresh true state."""
        state = np.asarray(self.world.initial_state_dist().sample()[0])
        self._state = state
        self._previous_state = None
        self._action = None
        return state

    def _as_tensor(self, array: np.ndarray) -> Any:
        import torch  # pylint: disable=import-outside-toplevel

        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self._device).unsqueeze(
            0
        )

    def transition(self, states: Any, actions: Any) -> Any:
        """VOPP ``world_transition`` hook: step the live simulator once."""
        del states
        self._action = self.action_presets[int(actions[0])]
        self._previous_state = self._state
        next_state = np.asarray(self.world.sample_next_state(self._previous_state, self._action))
        self._state = next_state
        return self._as_tensor(next_state)

    def observation(self, next_states: Any, actions: Any) -> Any:
        """VOPP ``world_observation`` hook: serve the sensor reading just taken."""
        del next_states, actions
        observation = self.world.sample_observation(self._state, self._action)
        return self._as_tensor(np.asarray(observation))

    def step_info(self) -> Dict[str, float]:
        """VOPP ``world_step_info`` hook: the world's own per-step measurements."""
        return self.world.step_info(self._previous_state, self._action, self._state)

    def terminal(self) -> bool:
        """VOPP ``world_terminal`` hook: the world decides, not the surrogate model."""
        return bool(self.world.is_terminal(self._state))


def collect_warmup_samples(
    driver: WorldDriver, num_transitions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out random actions to fit the planner-side dynamics and reward.

    Args:
        driver: Driver wrapping the live world.
        num_transitions: Number of transitions to collect.

    Returns:
        Arrays of states, action vectors, next states and rewards.
    """
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    rewards: List[float] = []

    rng = np.random.default_rng(0)
    state = driver.reset()
    for _ in range(num_transitions):
        action = driver.action_presets[int(rng.integers(len(driver.action_presets)))]
        reward = driver.world.reward(state, action)
        next_state = driver.world.sample_next_state(state, action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(float(reward))
        state = next_state
        if driver.world.is_terminal(state):
            state = driver.reset()
    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        np.asarray(next_states, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
    )


def build_vectorized_model(
    samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    action_presets: np.ndarray,
    device: Any,
) -> Any:
    """Fit the linear dynamics/reward and wrap them in the vectorized model."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp import (
        GaussianObservationModel,
        LinearGaussianTransition,
        LinearRewardModel,
    )

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_vectorized_model import (
        IsaacLabVectorizedModel,
    )

    states, actions, next_states, rewards = samples
    transition = LinearGaussianTransition.fit(states, actions, next_states)
    reward_model = LinearRewardModel.fit(states, actions, next_states, rewards)
    observation_model = GaussianObservationModel(
        observation_dim=states.shape[1], noise_std=OBSERVATION_NOISE_STD
    )
    return IsaacLabVectorizedModel(
        transition=transition,
        observation_model=observation_model,
        reward_model=reward_model,
        action_presets=action_presets,
        device=device,
        observation_resolution=OBSERVATION_RESOLUTION,
    )


def build_world(spec: TaskSpec, cache_dir: Path, record_video: bool) -> Any:
    """Construct the IsaacLab world configured for this task's measurements."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.isaac_lab_pomdp import IsaacLabPOMDP

    modifier = (
        make_contact_sensor_injector(spec.contact_body_regex)
        if spec.contact_body_regex is not None
        else None
    )
    return IsaacLabPOMDP(
        task_id=spec.task_id,
        discount_factor=GAMMA,
        state_extractor=_policy_observation,
        observation_extractor=_policy_observation,
        contact_sensor_key=spec.contact_sensor_key,
        success_extractor=make_success_extractor(spec),
        success_reduction=EpisodeReduction(spec.success_reduction),
        env_cfg_modifier=modifier,
        render_mode="rgb_array" if record_video else None,
        record_video=record_video,
        output_dir=cache_dir,
    )


def run_episodes(
    driver: WorldDriver, model: Any, num_episodes: int, num_steps: int
) -> Tuple[List[List[Dict[str, float]]], List[Dict[str, Any]]]:
    """Run VOPP episodes and collect their per-step measurements.

    Args:
        driver: Driver wrapping the live world.
        model: The fitted vectorized model VOPP plans inside.
        num_episodes: Number of episodes to run.
        num_steps: Maximum steps per episode.

    Returns:
        The per-episode step-info sequences, and a per-episode summary.
    """
    import torch  # pylint: disable=import-outside-toplevel

    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.planners.vectorized_planners import VOPPEpisodeRunner, VOPPPlanner

    planner = VOPPPlanner(
        model,
        num_actions=model.num_actions,
        num_particles=PLANNING_PARTICLES,
        max_depth=PLANNING_DEPTH,
        discount_factor=GAMMA,
        num_planning_iterations=PLANNING_ITERATIONS,
    )
    runner = VOPPEpisodeRunner(
        planner,
        model,
        num_belief_particles=BELIEF_PARTICLES,
        max_steps=num_steps,
        world_transition=driver.transition,
        world_observation=driver.observation,
        world_step_info=driver.step_info,
        world_terminal=driver.terminal,
    )

    episodes: List[List[Dict[str, float]]] = []
    summaries: List[Dict[str, Any]] = []
    for episode_index in range(num_episodes):
        state = driver.reset()
        torch.manual_seed(episode_index)
        result = runner.run_episode(
            torch.as_tensor(np.asarray(state, dtype=np.float32), device=model.device).unsqueeze(0)
        )
        episodes.append(result.step_infos)
        summaries.append(
            {
                "episode": episode_index,
                "steps": result.num_steps,
                "reached_terminal": result.reached_terminal_state,
                "model_return": float(sum(result.rewards)),
                "plan_time_s": round(result.total_plan_time, 3),
                "unique_actions": len(set(result.action_indices)),
            }
        )
        if episode_index == 0 and driver.world.record_video:
            driver.world.cache_visualization([], driver.world.output_dir, episode_index)
    return episodes, summaries


def summarize(
    spec: TaskSpec, episodes: Sequence[Sequence[Dict[str, float]]]
) -> List[Dict[str, Any]]:
    """Aggregate per-step channels into named metrics with 95% CIs."""
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.core.simulation.step_info_metrics import (
        StepInfoMetric,
        aggregate_step_info_metrics,
    )

    specs = [
        StepInfoMetric(
            name=IsaacLabMetric.SUCCESS_RATE.value,
            channel=IsaacLabStepChannel.SUCCESS.value,
            per_episode=EpisodeReduction(spec.success_reduction),
        ),
        StepInfoMetric(
            name=IsaacLabMetric.MAX_CONTACT_IMPULSE_NS.value,
            channel=IsaacLabStepChannel.CONTACT_IMPULSE_NS.value,
            per_episode=EpisodeReduction.MAX,
        ),
    ]
    return [
        {
            "metric": metric.name,
            "value": metric.value,
            "ci_lower": metric.lower_confidence_bound,
            "ci_upper": metric.upper_confidence_bound,
            "n": len(episodes),
        }
        for metric in aggregate_step_info_metrics(episodes, specs)
    ]


def run_task(spec: TaskSpec, cache_dir: Path, num_episodes: int, num_steps: int) -> Dict[str, Any]:
    """Run one task end to end and return its metrics and per-episode summary."""
    task_dir = cache_dir / spec.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    world = build_world(spec, task_dir, record_video=True)
    action_dim = int(np.asarray(world.action_space.shape)[-1])
    action_presets = build_action_presets(action_dim, NUM_ACTION_PRESETS, seed=0)

    import torch  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    driver = WorldDriver(world, action_presets, device)

    samples = collect_warmup_samples(driver, WARMUP_TRANSITIONS)
    model = build_vectorized_model(samples, action_presets, device)
    episodes, summaries = run_episodes(driver, model, num_episodes, num_steps)

    video = task_dir / "agent_path_0.mp4"
    return {
        "task_id": spec.task_id,
        "metrics": summarize(spec, episodes),
        "episodes": summaries,
        "video": str(video) if video.exists() else None,
        "state_dim": int(samples[0].shape[1]),
        "action_dim": action_dim,
        "num_action_presets": len(action_presets),
    }


def _spawn_task_process(
    spec: TaskSpec, cache_dir: Path, episodes: int, steps: int
) -> Optional[Dict[str, Any]]:
    """Run one task in a dedicated child process and return its result.

    The child is launched with this script's own CLI rather than through
    ``multiprocessing``. That matters for operability: a ``multiprocessing``
    spawn child (and a joblib/loky worker alike) runs as
    ``python -c 'from multiprocessing.spawn import spawn_main; ...'``, whose
    command line carries no trace of this script — so a ``pkill`` by name kills
    the parent and silently orphans a worker still holding several GB of GPU
    memory. Here the child's argv is
    ``python isaac_vopp_metrics_example.py --single-process --task <id>``:
    greppable, killable by name, and guaranteed to be a fresh interpreter that
    exits when the task is done, which the one-SimulationApp-per-process limit
    requires anyway.

    Args:
        spec: The task to run.
        cache_dir: Directory for outputs.
        episodes: Episodes to run.
        steps: Maximum steps per episode.

    Returns:
        The task's result payload, or ``None`` if the child failed.
    """
    out = cache_dir / f"{spec.task_id}.json"
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-process",
        "--task",
        spec.task_id,
        "--episodes",
        str(episodes),
        "--steps",
        str(steps),
        "--cache-dir",
        str(cache_dir),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not out.exists():
        print(f"WARNING: task {spec.task_id} failed (exit {completed.returncode})")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def format_report(results: Sequence[Dict[str, Any]]) -> str:
    """Render the collected results as a Markdown report."""
    lines = ["| Metric | Task | Mean | 95% CI | n |", "| --- | --- | --- | --- | --- |"]
    for result in results:
        for metric in result["metrics"]:
            low, high = metric["ci_lower"], metric["ci_upper"]
            ci = (
                "unbounded"
                if not np.isfinite(low) or not np.isfinite(high)
                else f"[{low:.4g}, {high:.4g}]"
            )
            lines.append(
                f"| {metric['metric']} | {result['task_id']} | {metric['value']:.4g} | {ci} | {metric['n']} |"
            )
    lines.append("")
    lines.append(
        "| Task | Episode | Steps | Terminal | Model return | Plan time (s) | Distinct actions |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        for episode in result["episodes"]:
            lines.append(
                f"| {result['task_id']} | {episode['episode']} | {episode['steps']} | "
                f"{episode['reached_terminal']} | {episode['model_return']:.4g} | "
                f"{episode['plan_time_s']} | {episode['unique_actions']} |"
            )
    lines.append("")
    for result in results:
        lines.append(f"- `{result['task_id']}` video: {result['video']}")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", help="Task id to run (repeatable).")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cache-dir", type=Path, default=Path("isaac_vopp_metrics"))
    parser.add_argument("--single-process", action="store_true", help="Run one task in-process.")
    return parser.parse_args(argv)


def _abort_if_gpu_already_busy() -> None:
    """Refuse to start when another process already holds significant GPU memory.

    A rendering Isaac Sim needs roughly 4-6 GB. If a previous run was killed by
    pattern rather than by process group its worker survives — its command line
    is ``spawn_main(...) --multiprocessing-fork`` and carries no trace of this
    script — and keeps holding that memory. Launching on top of such an orphan
    exhausts the device and shows up as an unrelated
    ``cudaErrorLaunchFailure`` deep inside the planner, or hangs the machine.
    Failing here with the offending PID is far cheaper to act on.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return  # No nvidia-smi: nothing to assert, let the run proceed.

    busy = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    heavy = [line for line in busy if _reported_megabytes(line) >= GPU_BUSY_THRESHOLD_MB]
    if heavy:
        sys.exit(
            "ERROR: the GPU is already in use by:\n  "
            + "\n  ".join(heavy)
            + "\nEach rendering Isaac Sim needs ~4-6 GB, so starting now risks exhausting the "
            "device. If these are orphaned workers from a killed run, stop them by PID "
            "(pkill on this script's name does NOT match them)."
        )


def _reported_megabytes(csv_line: str) -> float:
    """Parse the ``used_memory`` column of an ``nvidia-smi`` CSV row."""
    parts = csv_line.split(",")
    if len(parts) < 2:
        return 0.0
    digits = "".join(char for char in parts[1] if char.isdigit() or char == ".")
    return float(digits) if digits else 0.0


def main(argv: Optional[List[str]] = None) -> None:
    """Run every selected task, each in its own process, and print the report."""
    args = parse_args(argv)
    selected = [spec for spec in TASKS if args.task is None or spec.task_id in args.task]
    if not selected:
        sys.exit(f"No matching task. Known: {[spec.task_id for spec in TASKS]}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Re-exec before spawning: children inherit this interpreter, so switching
    # here is what makes isaaclab importable in them too.
    _reexec_under_isaac_if_needed()

    _abort_if_gpu_already_busy()

    if args.single_process:
        # Leaf mode: this process owns the one SimulationApp and writes the JSON.
        results = []
        for spec in selected:
            result = run_task(spec, args.cache_dir, args.episodes, args.steps)
            (args.cache_dir / f"{spec.task_id}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            results.append(result)
    else:
        # One SimulationApp per process is a hard IsaacLab limit, so each task
        # runs in its own child process rather than in a loop here.
        results = [
            result
            for result in (
                _spawn_task_process(spec, args.cache_dir, args.episodes, args.steps)
                for spec in selected
            )
            if result is not None
        ]

    report = format_report(results)
    # Only the top-level run owns report.md. A child writing it too would leave a
    # partial report on disk between tasks, which reads as a finished run.
    if not args.single_process:
        (args.cache_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    if args.single_process:
        # This process launched a SimulationApp, which leaves non-daemon threads
        # running: returning normally would hang here forever, still holding
        # several GB of GPU memory, with the parent blocked waiting on us. All
        # output is already on disk, so skip interpreter teardown entirely.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # pylint: disable=protected-access


if __name__ == "__main__":
    main()
