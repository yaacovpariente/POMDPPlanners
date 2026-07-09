# SPDX-License-Identifier: MIT

"""Plan on the nuPlan world with POMCPOW and record a top-down (BEV) video of the driven ego.

This is a *two-environment* POMDP demo, mirroring ``carla_pomcpow_example.py``:

* **World** — :class:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp.NuPlanPOMDP` is a
  forward-only adapter over a nuPlan closed-loop :class:`Simulation`. It supplies the
  ground-truth transition, the partial ``{ego, agents}`` observation, and the reward.
* **Model** —
  :class:`~POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models.nuplan_kinematic_model_pomdp.KinematicNuPlanModelPOMDP`
  is the planner-side generative model POMCPOW searches inside: a real kinematic-bicycle ego
  transition plus a factored per-agent observation model.

Unlike CARLA (which renders a native chase camera), nuPlan exposes no camera here, so this
script builds its **own** top-down bird's-eye-view MP4 from the episode ``History``: the ego's
driven trajectory plus the tracked agents, frame by frame.

World backend
    The real nuPlan world needs the ``nuplan-devkit`` **and** a scenario from the nuPlan
    dataset, supplied through ``NuPlanPOMDP(scenario_loader=...)``. When the devkit/dataset are
    unavailable (the default), this script runs against a lightweight **synthetic** world that
    speaks the exact same nuPlan state/observation schema — an accelerating ego, a slower lead
    vehicle, and an adjacent-lane vehicle — so the POMCPOW episode and its BEV video run
    anywhere. Pass ``--real-nuplan`` (with a configured ``scenario_loader``) to plan on the
    genuine nuPlan simulation instead.

Run it with the project venv::

    python nuplan_pomcpow_example.py --seconds 5 --cache-dir nuplan_pomcpow_run

Requirements:
    - ``matplotlib`` (BEV rendering) and ``ffmpeg`` on PATH to encode the MP4 (a ``.gif``
      fallback is written when ffmpeg is absent).
    - Only for ``--real-nuplan``: the ``nuplan-devkit`` importable and a ``scenario_loader``.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.environments.nuplan_pomdp.nuplan_belief import PerceivedAgentsBelief
from POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models import (
    KinematicNuPlanModelPOMDP,
    NuPlanModelPOMDP,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    NuPlanPOMDP,
    assemble_state,
    relative_agent_row,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.simulations.episodes import run_episode

# Shared discount for the world and the model (episode-runner consistency).
GAMMA = 0.95
# nuPlan simulation iteration length; SECONDS / DT iterations span the requested duration.
DT = 0.1
# Fixed agent-slot count carried by both the world and the model.
MAX_TRACKED_AGENTS = 5


class _SyntheticNuPlanSession:
    """A lightweight forward-only world in the exact nuPlan state/observation schema.

    Stands in for a live nuPlan :class:`Simulation` when the devkit/dataset are unavailable.
    The ego is propagated by the same kinematic-bicycle model the planner uses; scripted other
    vehicles advance along the road at constant speed. State/observation follow
    ``[ego(7) | agent slots(K*5)]`` exactly, so the planner cannot tell it apart from the real
    world by shape.
    """

    def __init__(self, max_tracked_agents: int, dt: float, collision_distance: float) -> None:
        self._max_tracked_agents = max_tracked_agents
        self._dt = dt
        self._collision_distance = collision_distance
        self._wheelbase = 2.8
        self._ego = np.zeros(EGO_STATE_WIDTH)
        # Other vehicles as world rows [x, y, yaw, speed]: a slower lead vehicle in-lane and
        # a faster vehicle in the left-adjacent lane.
        self._agents = np.array(
            [
                [28.0, 0.0, 0.0, 2.0],
                [16.0, 3.7, 0.0, 3.0],
            ]
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        del seed
        self._ego = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])  # 3 m/s forward at origin
        self._agents = np.array([[28.0, 0.0, 0.0, 2.0], [16.0, 3.7, 0.0, 3.0]])
        state = self._state()
        return state, self._observation(state)

    def step(
        self, acceleration: float, steering_angle: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
        self._advance_ego(acceleration, steering_angle)
        self._agents[:, 0] += self._agents[:, 3] * self._dt  # other vehicles roll along +x
        state = self._state()
        return state, self._observation(state), self._collided(state)

    def _advance_ego(self, acceleration: float, steering_angle: float) -> None:
        x, y, yaw, vel_x, vel_y = self._ego[:5]
        speed = float(np.hypot(vel_x, vel_y))
        speed_next = max(0.0, speed + acceleration * self._dt)
        yaw_rate = (speed_next / self._wheelbase) * np.tan(steering_angle)
        yaw_next = yaw + yaw_rate * self._dt
        vx_next = speed_next * np.cos(yaw_next)
        vy_next = speed_next * np.sin(yaw_next)
        self._ego = np.array(
            [
                x + vx_next * self._dt,
                y + vy_next * self._dt,
                yaw_next,
                vx_next,
                vy_next,
                y + vy_next * self._dt,  # lateral offset from the y=0 route baseline
                yaw_next,  # heading error vs the +x baseline
            ]
        )

    def _agent_rows(self) -> List[np.ndarray]:
        ego_x, ego_y, ego_yaw = float(self._ego[0]), float(self._ego[1]), float(self._ego[2])
        rows = []
        for other_x, other_y, other_yaw, speed in self._agents:
            rows.append(
                relative_agent_row(ego_x, ego_y, ego_yaw, other_x, other_y, other_yaw, speed)
            )
        return rows

    def _state(self) -> np.ndarray:
        return assemble_state(self._ego, self._agent_rows(), self._max_tracked_agents)

    def _observation(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        agents_end = EGO_STATE_WIDTH + self._max_tracked_agents * AGENT_SLOT_WIDTH
        return {
            "ego": state[:EGO_STATE_WIDTH].copy(),
            "agents": state[EGO_STATE_WIDTH:agents_end].copy(),
        }

    def _collided(self, state: np.ndarray) -> bool:
        agents_end = EGO_STATE_WIDTH + self._max_tracked_agents * AGENT_SLOT_WIDTH
        rows = state[EGO_STATE_WIDTH:agents_end].reshape(self._max_tracked_agents, AGENT_SLOT_WIDTH)
        present = rows[rows[:, 0] == 1.0]
        if present.shape[0] == 0:
            return False
        return bool(np.min(np.hypot(present[:, 1], present[:, 2])) < self._collision_distance)


class _SyntheticNuPlanPOMDP(NuPlanPOMDP):
    """A :class:`NuPlanPOMDP` whose live session is the schema-faithful synthetic world.

    Overrides the ``_get_session`` seam (the same one the unit tests substitute) so the episode
    runs without the nuPlan devkit while every public method behaves exactly as the real world.
    """

    def __init__(self, synthetic_session: _SyntheticNuPlanSession, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._synthetic_session = synthetic_session

    def _get_session(self) -> Any:
        return self._synthetic_session


def build_world(seed: int, real_nuplan: bool) -> NuPlanPOMDP:
    """Construct the forward-only nuPlan world.

    Returns the genuine :class:`NuPlanPOMDP` when ``real_nuplan`` is set (you must wire a
    ``scenario_loader`` for that path); otherwise returns the synthetic in-schema world so the
    demo runs without the devkit/dataset. The world emits the raw ``{ego, agents}`` observation
    unchanged — perception lives solely on the planner's model.
    """
    if real_nuplan:
        return NuPlanPOMDP(
            discount_factor=GAMMA,
            max_tracked_agents=MAX_TRACKED_AGENTS,
            fixed_delta_seconds=DT,
            seed=seed,
            scenario_loader=None,  # <- supply a nuPlan AbstractScenario loader here
        )
    session = _SyntheticNuPlanSession(
        max_tracked_agents=MAX_TRACKED_AGENTS, dt=DT, collision_distance=2.0
    )
    return _SyntheticNuPlanPOMDP(
        synthetic_session=session,
        discount_factor=GAMMA,
        max_tracked_agents=MAX_TRACKED_AGENTS,
        fixed_delta_seconds=DT,
        seed=seed,
    )


def build_model() -> KinematicNuPlanModelPOMDP:
    """Construct the planner-side model with a kinematic ego transition and safe-driving reward.

    Uses :class:`KinematicNuPlanModelPOMDP` ((acceleration, steering_angle) propagated over
    ``dt``) so POMCPOW's lookahead sees that accelerating produces motion and the car actually
    drives. The obstacle-aware desired speed targets the full speed on a clear road (lead gap
    >= ``safe_distance``) and ramps to zero as a lead obstacle nears ``stop_gap``, so the
    planner brakes *before* the terminal collision box without freezing in traffic.
    """
    return KinematicNuPlanModelPOMDP(
        discount_factor=GAMMA,
        dt=DT,
        max_tracked_agents=MAX_TRACKED_AGENTS,
        desired_speed=6.0,
        collision_gap=5.0,
        safe_distance=15.0,
        stop_gap=7.0,
    )


def build_planner(model: NuPlanModelPOMDP, depth: int, timeout: int) -> POMCPOW:
    """Construct a POMCPOW planner over the given model."""
    actions = model.get_actions()
    # UCB exploration on the scale of the worst-case return: the collision penalty dominates
    # the per-step reward range, scaled by the planning depth.
    reward_scale = getattr(model, "collision_penalty", 100.0)
    return POMCPOW(
        environment=model,
        discount_factor=GAMMA,
        depth=depth,
        exploration_constant=reward_scale * depth,
        k_o=4.0,
        alpha_o=0.1,
        k_a=float(len(actions)),
        alpha_a=0.0,
        name="POMCPOW-NuPlan",
        action_sampler=DiscreteActionSampler(actions),
        time_out_in_seconds=timeout,
    )


def seed_belief(model: NuPlanModelPOMDP, n_particles: int, initial_observation: dict) -> Belief:
    """Seed a :class:`PerceivedAgentsBelief` that starts from — and re-acquires — observed traffic.

    ``run_episode`` sources the *true* start state from the forward-only world itself, so the
    belief only needs a plausible starting cloud. Each particle is ego near the origin with
    jitter and its agent block seeded from the world's first raw ``agents`` (so the planner sees
    the traffic in view at step 0). On every update the belief stamps each particle's agent block
    with the observation's perceived ``agents`` — perception itself is the planner model's,
    applied by its ``encode_observation``, not the world or the belief.
    """
    log_weights = np.log(np.ones(n_particles) / n_particles)
    width = EGO_STATE_WIDTH + model.max_tracked_agents * AGENT_SLOT_WIDTH
    observed_agents = np.asarray(initial_observation["agents"], dtype=float)
    particles = []
    for _ in range(n_particles):
        particle = np.zeros(width)
        particle[:EGO_STATE_WIDTH] = np.random.normal(0.0, 0.1, size=EGO_STATE_WIDTH)
        particle[EGO_STATE_WIDTH:] = observed_agents  # seed the detected traffic
        particles.append(particle)
    return PerceivedAgentsBelief(
        particles=particles,
        log_weights=log_weights,
        max_tracked_agents=model.max_tracked_agents,
        resampling=True,
    )


# Approximate vehicle footprint (m) drawn for the ego and other agents in the BEV.
_VEHICLE_LENGTH = 4.6
_VEHICLE_WIDTH = 2.0
# Half-window (m) of the follow-camera framing the ego each frame.
_VIEW_HALF = 22.0


def _world_agents(state: np.ndarray) -> List[Tuple[float, float, float]]:
    """World-frame ``(x, y, yaw)`` of every present agent slot in ``state``."""
    ego_x, ego_y, ego_yaw = float(state[0]), float(state[1]), float(state[2])
    cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
    rows = state[EGO_STATE_WIDTH : EGO_STATE_WIDTH + MAX_TRACKED_AGENTS * AGENT_SLOT_WIDTH].reshape(
        MAX_TRACKED_AGENTS, AGENT_SLOT_WIDTH
    )
    agents = []
    for row in rows:
        if row[0] != 1.0:
            continue
        rel_x, rel_y, rel_yaw = float(row[1]), float(row[2]), float(row[3])
        world_x = ego_x + cos_yaw * rel_x - sin_yaw * rel_y
        world_y = ego_y + sin_yaw * rel_x + cos_yaw * rel_y
        agents.append((world_x, world_y, ego_yaw + rel_yaw))
    return agents


def _vehicle_corners(center_x: float, center_y: float, yaw: float) -> np.ndarray:
    """Four world-frame corners of a vehicle footprint centred at ``(center_x, center_y)``."""
    half_l, half_w = _VEHICLE_LENGTH / 2.0, _VEHICLE_WIDTH / 2.0
    local = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([center_x, center_y])


def _draw_road(axis: Any, ego_x: float) -> None:
    """Draw a straight multi-lane road with dashed lane markings around the ego."""
    from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel

    lane_centres = (0.0, 3.7)  # the synthetic scene's two same-direction lanes
    road_lo, road_hi = min(lane_centres) - 1.85, max(lane_centres) + 1.85
    axis.add_patch(
        Rectangle(
            (ego_x - 60.0, road_lo),
            120.0,
            road_hi - road_lo,
            facecolor="#3a3a3a",
            edgecolor="none",
            zorder=0,
        )
    )
    for boundary in (road_lo, road_hi):
        axis.plot(
            [ego_x - 60.0, ego_x + 60.0], [boundary, boundary], color="#f5d020", lw=2, zorder=1
        )
    axis.plot(
        [ego_x - 60.0, ego_x + 60.0],
        [(lane_centres[0] + lane_centres[1]) / 2.0] * 2,
        color="white",
        lw=1.2,
        ls=(0, (12, 12)),
        zorder=1,
    )


def render_topdown_video(history: History, out_path: Path, fps: int) -> Path:
    """Render a top-down bird's-eye-view MP4 (or GIF fallback) of the driven episode.

    Draws a follow-camera BEV — a straight multi-lane road, the ego as an oriented vehicle
    rectangle with its driven trail, and every tracked agent as an oriented rectangle — frame by
    frame, in the world frame reconstructed from the episode ``History`` states. This is the
    nuPlan-native visualization style (top-down boxes on the road), the counterpart to CARLA's
    chase camera; nuPlan has no 3D renderer, so novel photorealistic views are not available.
    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.animation import (  # pylint: disable=import-outside-toplevel
        FFMpegWriter,
        FuncAnimation,
        PillowWriter,
    )
    from matplotlib.patches import Polygon  # pylint: disable=import-outside-toplevel

    states = [np.asarray(step.state, dtype=float) for step in history.history]
    states.append(np.asarray(history.history[-1].next_state, dtype=float))
    ego_xy = np.array([[s[0], s[1]] for s in states])

    fig, axis_obj = plt.subplots(figsize=(9, 6), facecolor="#101418")
    axis: Any = axis_obj

    def draw(frame: int) -> list:
        axis.clear()
        state = states[frame]
        ego_x, ego_y, ego_yaw = float(state[0]), float(state[1]), float(state[2])
        axis.set_facecolor("#1b2027")
        axis.set_xlim(ego_x - _VIEW_HALF * 1.5, ego_x + _VIEW_HALF * 1.5)
        axis.set_ylim(ego_y - _VIEW_HALF, ego_y + _VIEW_HALF)
        axis.set_aspect("equal")
        axis.tick_params(colors="#8a94a6")
        axis.set_title(f"POMCPOW on nuPlan (BEV) — step {frame}/{len(states) - 1}", color="#e6ebf2")
        _draw_road(axis, ego_x)
        axis.plot(
            ego_xy[: frame + 1, 0], ego_xy[: frame + 1, 1], "-", color="#4da3ff", lw=2, zorder=2
        )
        axis.add_patch(
            Polygon(_vehicle_corners(ego_x, ego_y, ego_yaw), closed=True, color="#1f77ff", zorder=4)
        )
        for agent_x, agent_y, agent_yaw in _world_agents(state):
            axis.add_patch(
                Polygon(
                    _vehicle_corners(agent_x, agent_y, agent_yaw),
                    closed=True,
                    color="#ff4d4d",
                    zorder=3,
                )
            )
        speed = float(np.hypot(state[3], state[4]))
        axis.text(
            0.02,
            0.96,
            f"ego speed: {speed:4.1f} m/s",
            transform=axis.transAxes,
            va="top",
            fontsize=12,
            color="#e6ebf2",
        )
        return []  # blit disabled: no artists need to be returned

    animation = FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    if shutil.which("ffmpeg") is not None:
        animation.save(str(out_path), writer=FFMpegWriter(fps=fps), dpi=130)
        plt.close(fig)
        return out_path
    gif_path = out_path.with_suffix(".gif")
    animation.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return gif_path


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line options for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=5.0, help="Simulated duration to plan.")
    parser.add_argument("--cache-dir", type=Path, default=Path("nuplan_pomcpow_run"))
    parser.add_argument("--n-particles", type=int, default=60, help="Belief particle count.")
    parser.add_argument("--depth", type=int, default=20, help="POMCPOW planning horizon.")
    parser.add_argument("--timeout", type=int, default=1, help="POMCPOW seconds per step.")
    parser.add_argument("--seed", type=int, default=0, help="World reset seed.")
    parser.add_argument("--fps", type=int, default=10, help="BEV video frame rate.")
    parser.add_argument(
        "--real-nuplan",
        action="store_true",
        help="Plan on the genuine nuPlan Simulation (requires the devkit and a scenario_loader) "
        "instead of the synthetic in-schema world.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Plan with POMCPOW on nuPlan and render the driven ego's top-down BEV video."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    np.random.seed(args.seed)
    num_steps = round(args.seconds / DT)

    world = build_world(seed=args.seed, real_nuplan=args.real_nuplan)
    initial_observation = world.initial_observation_dist().sample(1)[0]
    model = build_model()
    planner = build_planner(model, depth=args.depth, timeout=args.timeout)
    belief = seed_belief(model, args.n_particles, initial_observation)

    backend = "real nuPlan simulation" if args.real_nuplan else "synthetic in-schema world"
    print(
        f"Planning {args.seconds:.1f}s ({num_steps} iters @ {DT}s) with POMCPOW on the "
        f"{backend}..."
    )
    history = run_episode(
        environment=world,
        policy=planner,
        initial_belief=belief,
        num_steps=num_steps,
        logger=None,
    )

    cache_dir = args.cache_dir / f"{type(world).__name__}_{type(model).__name__}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = render_topdown_video(history, cache_dir / "agent_path_0.mp4", fps=args.fps)

    total_reward = float(sum(float(step.reward or 0.0) for step in history.history))
    print(
        f"\nDone | {history.actual_num_steps} steps | total reward {total_reward:.1f} | "
        f"terminal collision: {history.reach_terminal_state}"
    )
    print(f"Saved top-down BEV video to {video_path}")


if __name__ == "__main__":
    main()
