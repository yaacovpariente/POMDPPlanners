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

**Why the success predicate is per task.** None of these tasks ships a "success"
termination term — probing them shows only failure and timeout terms
(``base_contact``, ``cart_out_of_bounds``, ``time_out``). Task completion is
therefore defined per task in :mod:`isaac_vopp_tasks` and injected as a
``success_extractor``, which is exactly the split the design intends: the
*measurement* is environment-specific, the *channel name*, the aggregation and
the confidence intervals are shared.

Cartpole shows why the shipped terms are not enough on their own. Its only
failure term bounds the *cart*, leaving the pole angle unconstrained, so a policy
that does nothing while the pole spins through a full rotation scores 1.0. Its
predicate therefore also requires the pole to stay within a right angle of
vertical — measured, not assumed: a do-nothing, a frozen-action and a
random-action rollout each score 0/10 under it.

**Why a contact sensor is injected.** Only the locomotion tasks ship one. For the
others an ``env_cfg_modifier`` attaches a ``ContactSensorCfg`` and switches on
the asset's contact-reporter API, without which the sensor finds no bodies.

**Planner-side model.** Three are wired here, selectable per task and overridable
with ``--model-kind``.

The default fits a ``LinearGaussianTransition`` and a ``LinearRewardModel`` from a
short warm-up of random actions -- a crude system-identification baseline that
steers behaviour but does not solve locomotion.

On ``Isaac-Reach-Franka-v0`` it does not steer at all: one linear map over a 7-DoF
arm scores every action alike, so the planner emits a single action index for the
whole episode (``unique_actions == 1``) and never reaches. That task therefore
uses ``ManipulatorIsaacModel`` -- a joint lag whose gain is calibrated from the
same warm-up rollout, exact forward kinematics through the Panda's DH chain, and
the reach task's own distance objective. Only the lag is fitted.

``Isaac-Navigation-Flat-Anymal-C-v0`` fails the same way for a different reason:
its observation carries no base position, so a fitted map over it cannot learn
where turning takes the robot. That task uses ``NavigationIsaacModel``, which
integrates the goal *in the base frame* forward under the velocity command --
exactly the quantity the observation does carry -- and scores it with the task's
own pose-tracking reward. Only the command-tracking scales are fitted.

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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from isaac_vopp_planner_models import (
    GAMMA,
    build_manipulator_model,
    build_navigation_model,
    build_vectorized_model,
)
from isaac_vopp_tasks import (
    MODEL_KIND_SUCCESS_KIND,
    TASKS,
    TaskSpec,
    ThresholdSuccessProbe,
    make_contact_sensor_injector,
    make_success_extractor,
    policy_observation,
)
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import (
    IsaacLabMetric,
    IsaacLabStepChannel,
)

ISAAC_PYTHON = os.environ.get(
    "ISAAC_PYTHON", "/home/kobi/Documents/tmp/isaac_sim_playground/env_isaacsim/bin/python"
)

VIDEO_FPS = 10
WARMUP_TRANSITIONS = 200
# Control steps each warm-up action is held for. A command redrawn every step measures a permanent
# transient rather than the system's tracking; see collect_warmup_samples for the measurement.
WARMUP_ACTION_HOLD_STEPS = 10
NUM_ACTION_PRESETS = 8
BELIEF_PARTICLES = 256
PLANNING_PARTICLES = 128
PLANNING_DEPTH = 8
PLANNING_ITERATIONS = 16
DEFAULT_EPISODES = 3
DEFAULT_STEPS = 40
# A rendering Isaac Sim needs ~4-6 GB; anything at this scale is a real workload,
# not desktop compositing.
GPU_BUSY_THRESHOLD_MB = 2000.0


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
    presets: List[np.ndarray] = [np.zeros(action_dim, dtype=np.float32)]
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
    driver: WorldDriver, num_transitions: int, hold_steps: int = WARMUP_ACTION_HOLD_STEPS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out held random actions to fit the planner-side dynamics and reward.

    **Why the action is held.** A system-identification rollout has to excite the system at the
    timescale it responds on. Redrawing a fresh command every control step does not: it measures a
    permanent transient. Measured on the ANYmal navigation task, the fraction of a commanded
    velocity the base achieves reads 0.25 when the command changes every 0.2 s, 0.47 when it is
    held four steps and 0.71 when it is held ten -- the gait needs about a second to reach the
    velocity it was asked for. Calibrating on the redrawn rollout therefore characterises a robot
    that is never allowed to follow its command, and every model fitted on it under-predicts how
    far the robot travels.

    Holding costs no action coverage here, because the action set is a finite preset table: 200
    transitions held ten at a time still draw twenty commands from a table of eight.

    **Episode boundaries are dropped, not fitted.** IsaacLab auto-resets inside ``step()``, so the
    successor of a terminal transition is a fresh episode's observation rather than the result of
    the action. Keeping those rows teaches a fitted model that some action teleports, and biases a
    calibrated one by the size of the jump. Collection continues until ``num_transitions`` usable
    rows exist, so every model is still fitted on the same budget.

    Args:
        driver: Driver wrapping the live world.
        num_transitions: Number of usable transitions to collect, excluding dropped boundaries.
        hold_steps: Control steps each drawn action is held for before a new one is drawn. The
            schedule restarts after an episode reset.

    Returns:
        Arrays of states, action vectors, next states and rewards.

    Raises:
        ValueError: If ``hold_steps`` is not positive.
    """
    if hold_steps <= 0:
        raise ValueError(f"hold_steps must be positive, got {hold_steps}")
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    rewards: List[float] = []

    rng = np.random.default_rng(0)
    state = driver.reset()
    action = driver.action_presets[0]
    held_for = 0
    while len(states) < num_transitions:
        if held_for == 0:
            action = driver.action_presets[int(rng.integers(len(driver.action_presets)))]
        reward = driver.world.reward(state, action)
        next_state = driver.world.sample_next_state(state, action)
        ended = driver.world.is_terminal(next_state)
        if not ended:
            # A transition whose successor is IsaacLab's post-reset observation is not a transition
            # of the system: the env auto-resets inside step(), so the "next state" is a fresh
            # episode metres away. Fitting it teaches every model that some action teleports.
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(float(reward))
        state = next_state
        held_for = (held_for + 1) % hold_steps
        if ended:
            # Restart the hold too. Resuming a half-finished block would hand the freshly reset
            # system a command it has only a few steps left to follow, which is the transient this
            # whole hold exists to avoid measuring.
            state = driver.reset()
            held_for = 0
    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        np.asarray(next_states, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
    )


def build_world(
    spec: TaskSpec,
    cache_dir: Path,
    record_video: bool,
    success_extractor: Optional[Callable[[Any, Dict[str, Any], bool, bool], bool]] = None,
) -> Any:
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
        state_extractor=policy_observation,
        observation_extractor=policy_observation,
        contact_sensor_key=spec.contact_sensor_key,
        success_extractor=success_extractor or make_success_extractor(spec),
        success_reduction=EpisodeReduction(spec.success_reduction),
        env_cfg_modifier=modifier,
        render_mode="rgb_array" if record_video else None,
        record_video=record_video,
        output_dir=cache_dir,
    )


def run_episodes(
    driver: WorldDriver,
    model: Any,
    num_episodes: int,
    num_steps: int,
    probe: Optional[ThresholdSuccessProbe] = None,
) -> Tuple[List[List[Dict[str, float]]], List[Dict[str, Any]]]:
    """Run VOPP episodes and collect their per-step measurements.

    Args:
        driver: Driver wrapping the live world.
        model: The vectorized model VOPP plans inside.
        num_episodes: Number of episodes to run.
        num_steps: Maximum steps per episode.
        probe: Probe to read the task's own per-episode measurements from, when it has one.

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
        if probe is not None:
            probe.reset()
        torch.manual_seed(episode_index)
        result = runner.run_episode(
            torch.as_tensor(np.asarray(state, dtype=np.float32), device=model.device).unsqueeze(0)
        )
        episodes.append(result.step_infos)
        steps = max(1, result.num_steps)
        summary: Dict[str, Any] = {
            "episode": episode_index,
            "steps": result.num_steps,
            "reached_terminal": result.reached_terminal_state,
            "model_return": float(sum(result.rewards)),
            "plan_time_s": round(result.total_plan_time, 3),
            "plan_time_per_step_s": round(result.total_plan_time / steps, 3),
            "unique_actions": len(set(result.action_indices)),
        }
        if probe is not None:
            summary.update(probe.summary())
        summaries.append(summary)
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

    success_extractor = make_success_extractor(spec)
    probe = success_extractor if isinstance(success_extractor, ThresholdSuccessProbe) else None
    world = build_world(spec, task_dir, record_video=True, success_extractor=success_extractor)
    action_dim = int(np.asarray(world.action_space.shape)[-1])
    action_presets = build_action_presets(action_dim, NUM_ACTION_PRESETS, seed=0)

    import torch  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    driver = WorldDriver(world, action_presets, device)

    samples = collect_warmup_samples(driver, WARMUP_TRANSITIONS)
    if spec.model_kind == "manipulator":
        model = build_manipulator_model(world, samples, action_presets, device)
    elif spec.model_kind == "navigation":
        model = build_navigation_model(world, samples, action_presets, device)
    else:
        model = build_vectorized_model(samples, action_presets, device)
    episodes, summaries = run_episodes(driver, model, num_episodes, num_steps, probe)

    video = task_dir / "agent_path_0.mp4"
    return {
        "task_id": spec.task_id,
        "metrics": summarize(spec, episodes),
        "episodes": summaries,
        "video": str(video) if video.exists() else None,
        "state_dim": int(samples[0].shape[1]),
        "action_dim": action_dim,
        "num_action_presets": len(action_presets),
        "model_kind": spec.model_kind,
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
        # Forwarded explicitly: the child re-reads TASKS from source, so a --model-kind override
        # applied in the parent would otherwise be lost on the way down.
        "--model-kind",
        spec.model_kind,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not out.exists():
        print(f"WARNING: task {spec.task_id} failed (exit {completed.returncode})")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


#: The per-episode quantity each predicate is scored on, as ``{summary key: unit}``. Each task's
#: number keeps its own unit rather than sharing a column heading, because a metre of goal error
#: and a degree of pole lean are not comparable and a shared heading would invite reading them so.
EPISODE_MEASUREMENT_UNITS = {
    "min_reach_distance_m": "m",
    "min_goal_distance_m": "m",
    "max_pole_angle_deg": "deg",
}


def _episode_measurement(episode: Dict[str, Any]) -> str:
    """The episode's decisive measurement with its unit, whichever probe recorded it."""
    for key, unit in EPISODE_MEASUREMENT_UNITS.items():
        if key in episode:
            return f"{float(episode[key]):.4g} {unit}"
    return "n/a"


def _episode_ending(episode: Dict[str, Any]) -> str:
    """How the episode ended, as the world reported it on its last step.

    ``reached_terminal`` alone conflates a failure with a timeout, and for a fixed-length task the
    difference is the whole diagnosis: 40 steps and truncated is the task running its course, 12
    steps and terminated is the robot on the floor.
    """
    if episode.get("terminated"):
        return "terminated"
    if episode.get("truncated"):
        return "truncated"
    return "cut off" if episode.get("reached_terminal") else "ran out of steps"


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
        "| Task | Model | Episode | Steps | Model return | Plan time/step (s) | "
        "Distinct actions | Decisive measurement | Ended |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        for episode in result["episodes"]:
            lines.append(
                f"| {result['task_id']} | {result.get('model_kind', 'linear')} | "
                f"{episode['episode']} | {episode['steps']} | "
                f"{episode['model_return']:.4g} | "
                f"{episode.get('plan_time_per_step_s', episode['plan_time_s'])} | "
                f"{episode['unique_actions']} | "
                f"{_episode_measurement(episode)} | "
                f"{_episode_ending(episode)} |"
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
    parser.add_argument(
        "--model-kind",
        choices=["linear", "manipulator", "navigation"],
        default=None,
        help="Override the planner-side model for every selected task, so an analytic model and "
        "the fitted linear baseline can be compared on identical settings.",
    )
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
    if args.model_kind is not None:
        # Each analytic model is built for one shape of task. Applied to the wrong one it would
        # fail deep inside a manager read, after a SimulationApp had already been launched and
        # several GB of GPU claimed; say so before paying that.
        required = MODEL_KIND_SUCCESS_KIND.get(args.model_kind)
        incompatible = [
            spec.task_id
            for spec in selected
            if required is not None and spec.success_kind != required
        ]
        if incompatible:
            sys.exit(
                f"--model-kind {args.model_kind} applies only to a '{required}' task, but "
                f"{incompatible} were selected. Restrict the run with --task, or drop the "
                "override."
            )
        selected = [replace(spec, model_kind=args.model_kind) for spec in selected]
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
