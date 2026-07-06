# SPDX-License-Identifier: MIT

"""Plan on the CARLA world with POMCPOW and record 7 seconds of the driven ego.

This is a *two-environment* POMDP demo:

* **World** — :class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP`
  is a forward-only adapter over a live CARLA server. It supplies the ground-truth
  transition, the partial ``{gnss, agents, ...}`` sensor observation, the reward, and
  the chase-camera frames.
* **Model** —
  :class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_factored_model_pomdp.FactoredCarlaModelPOMDP`
  is the planner-side generative model POMCPOW searches inside. Its observation model is
  a real factored per-agent detector; its transition is the documented **identity
  placeholder**, so the planner currently predicts *no* motion (see the caveat below).

The episode loop and the video are handled by the project's own planner visualizer,
:func:`~POMDPPlanners.utils.planner_episode_visualization.visualize_planner_episode`,
which runs the two-environment episode via ``run_episode`` and then calls the world's
``cache_visualization`` hook — for ``CarlaPOMDP`` that writes CARLA's native
chase-camera footage as an MP4. The video lands at
``<cache-dir>/<planner-name>/agent_path_0.mp4``.

Run it with the project venv (it does not launch here — see requirements):
    python carla_pomcpow_example.py --seconds 7 --cache-dir carla_pomcpow_run

Requirements:
    - A running CARLA simulator (``0.9.15``) reachable at ``--host``/``--port`` and the
      ``carla`` Python package importable in this env. ``CarlaPOMDP`` imports ``carla``
      lazily, so this file imports fine without a server, but running it needs one.
    - ``ffmpeg`` on PATH to encode the MP4.

Learned dynamics (``--dreamer-checkpoint``):
    By default ``FactoredCarlaModelPOMDP.sample_next_state`` returns the state unchanged
    (an identity placeholder), so POMCPOW's lookahead sees a frozen world and effectively
    optimizes immediate reward plus random rollouts. Pass ``--dreamer-checkpoint PATH`` to
    a trained CarDreamer/DreamerV3 checkpoint (https://github.com/ucd-dare/CarDreamer) to
    instead plan inside a
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_dreamer_model_pomdp.DreamerCarlaModelPOMDP`
    wrapping the learned Dreamer world model, so the planner anticipates motion. The
    checkpoint's observation space must expose the CARLA ``{gnss, agents}`` schema and its
    action space must accept the ``(throttle, steer, brake)`` control triple. Requires
    ``jax``/``dreamerv3`` importable; the rest of this script is unchanged::

        python carla_pomcpow_example.py --dreamer-checkpoint ./logdir/carla/checkpoint.ckpt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.environments.carla_pomdp.carla_belief import PerceivedAgentsBelief
from POMDPPlanners.environments.carla_pomdp.carla_perception import (
    CarlaPerceptionPipeline,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
    CarlaPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    CarlaModelPOMDP,
    DreamerCarlaModelPOMDP,
    FactoredCarlaModelPOMDP,
    KinematicCarlaModelPOMDP,
    build_cardreamer_model,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.utils.planner_episode_visualization import visualize_planner_episode

# Shared discount for the world and the model (episode-runner consistency).
GAMMA = 0.95
# Synchronous CARLA tick length; SECONDS / DT ticks span the requested duration.
DT = 0.05


def build_world(seed: int, host: str, port: int) -> CarlaPOMDP:
    """Construct the forward-only CARLA world with onboard perception and chase-camera recording.

    The world's observation model runs a
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.pipeline.CarlaPerceptionPipeline`,
    so the emitted ``agents`` block is the *perceived* object list — vehicles clustered and
    tracked from the lidar, a fused lidar/camera forward hazard and a camera-inferred red/amber
    light folded in — rather than a ground-truth oracle. The planner's model scores that same
    perceived observation.
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=DEFAULT_MAX_TRACKED_AGENTS,
        sensor_fusion=True,
        stop_for_traffic_lights=True,
        obstacle_detection_range=30.0,
        dt=DT,
    )
    return CarlaPOMDP(
        discount_factor=GAMMA,
        record_camera=True,
        fixed_delta_seconds=DT,
        town="Town02",  # compact, dense map so traffic stays near the ego
        num_vehicles=80,  # heavy surrounding traffic (default is 30)
        num_walkers=0,  # walker-AI spawn segfaults native CARLA on Town02; vehicles suffice
        randomize_spawn=False,  # deterministic, known-busy ego spawn point
        perception_pipeline=pipeline,
        seed=seed,
        host=host,
        port=port,
    )


def build_factored_model() -> FactoredCarlaModelPOMDP:
    """Construct the planner-side model with a kinematic ego transition and safe-driving reward.

    Uses :class:`KinematicCarlaModelPOMDP` (throttle/steer/brake propagated over ``dt``)
    rather than the bare ``FactoredCarlaModelPOMDP`` identity placeholder, so POMCPOW's
    lookahead sees that throttle produces motion and the car actually drives. The safety
    parameters below are the tuned "safe city-driving" preset: a modest target speed and an
    obstacle-aware desired speed that targets the full speed on a clear road (lead gap
    >= ``safe_distance``) and ramps to zero as a lead obstacle nears ``stop_gap``, so the
    planner brakes *before* the terminal collision box without freezing in traffic.
    """
    return KinematicCarlaModelPOMDP(
        discount_factor=GAMMA,
        dt=DT,
        desired_speed=5.0,
        collision_gap=5.0,
        safe_distance=15.0,
        stop_gap=7.0,
    )


def _dreamer_spaces(model: CarlaModelPOMDP, initial_observation: dict):
    """Build DreamerV3 ``embodied`` observation/action spaces from the CARLA schema.

    The returned spaces must match the space the CarDreamer checkpoint was trained with;
    they are derived here from the world's actual initial observation and the model's
    discrete control set.
    """
    import embodied  # type: ignore[import]  # pylint: disable=import-outside-toplevel,import-error

    obs_space = {
        key: embodied.Space(np.float32, np.asarray(value).shape)
        for key, value in initial_observation.items()
    }
    act_space = {"action": embodied.Space(np.float32, (len(model.action_presets[0]),))}
    return obs_space, act_space


def build_dreamer_model(
    checkpoint: str, size: str, initial_observation: dict
) -> DreamerCarlaModelPOMDP:
    """Construct a Dreamer-backed model from a CarDreamer/DreamerV3 checkpoint.

    Loads the trained DreamerV3 world model at ``checkpoint`` (CarDreamer;
    https://github.com/ucd-dare/CarDreamer), wraps it in a ``DreamerCarlaModelPOMDP``, and
    seeds the belief posterior from the world's ``initial_observation``. The planner then
    searches inside the learned RSSM dynamics instead of the factored placeholder.
    """
    reference = build_factored_model()
    obs_space, act_space = _dreamer_spaces(reference, initial_observation)
    world_model = build_cardreamer_model(
        checkpoint_path=checkpoint,
        obs_space=obs_space,
        act_space=act_space,
        action_presets=reference.action_presets,
        config_size=size,
    )
    return DreamerCarlaModelPOMDP(
        world_model,
        discount_factor=GAMMA,
        action_presets=reference.action_presets,
        initial_observation=initial_observation,
    )


def build_model(
    dreamer_checkpoint: Optional[str],
    dreamer_size: str,
    initial_observation: dict,
) -> CarlaModelPOMDP:
    """Construct the planner-side generative model POMCPOW searches inside.

    Returns the factored reference model by default, or — when ``dreamer_checkpoint`` is
    given — a ``DreamerCarlaModelPOMDP`` wrapping the trained CarDreamer world model, so the
    planner anticipates learned motion instead of the factored identity placeholder.
    """
    if dreamer_checkpoint is None:
        return build_factored_model()
    return build_dreamer_model(dreamer_checkpoint, dreamer_size, initial_observation)


def build_planner(model: CarlaModelPOMDP, depth: int) -> POMCPOW:
    """Construct a POMCPOW planner over the given model."""
    actions = model.get_actions()
    # UCB exploration on the scale of the worst-case return: the largest per-step
    # reward magnitude (the collision penalty dominates the reward range) times depth.
    # A Dreamer model predicts reward from its latent and exposes no analytic penalty,
    # so fall back to a comparable default scale in that case.
    reward_scale = getattr(model, "collision_penalty", 100.0)
    exploration_constant = reward_scale * depth
    return POMCPOW(
        environment=model,
        discount_factor=GAMMA,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=4.0,
        alpha_o=0.1,
        k_a=float(len(actions)),
        alpha_a=0.0,
        name="POMCPOW-CARLA",
        action_sampler=DiscreteActionSampler(actions),
        time_out_in_seconds=1,
    )


def seed_belief(model: CarlaModelPOMDP, n_particles: int, initial_observation: dict) -> Belief:
    """Seed a particle belief that starts from — and keeps re-acquiring — observed traffic.

    ``run_episode`` sources the *true* start state from the forward-only world itself, so
    the belief only needs a plausible starting cloud for the filter to refine. For the
    Dreamer model the state is a latent, so seed from the RSSM posterior encoding of the
    world's initial observation and use a plain filter.

    For the factored/kinematic model each particle is ego near the origin with jitter, with
    its agent block seeded from the world's first *perceived* ``agents`` (so the planner sees
    the traffic in view at step 0). The belief is a
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_belief.PerceivedAgentsBelief`, which on
    every update stamps each particle's agent block with the observation's perceived ``agents``
    block — the perception itself is done by the world's observation model (see
    :func:`build_world`), not the belief. Without this stamping a plain particle filter could
    never acquire a vehicle that appears mid-episode; the perceived traffic and signals are used
    by the planner, not a reactive override.
    """
    log_weights = np.log(np.ones(n_particles) / n_particles)
    if isinstance(model, DreamerCarlaModelPOMDP):
        particles = model.initial_state_dist().sample(n_particles)
        return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True)
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


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line options for the demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=7.0, help="Simulated duration to plan.")
    parser.add_argument("--cache-dir", type=Path, default=Path("carla_pomcpow_run"))
    parser.add_argument("--n-particles", type=int, default=100, help="Belief particle count.")
    parser.add_argument("--depth", type=int, default=25, help="POMCPOW planning horizon.")
    parser.add_argument("--seed", type=int, default=0, help="World reset seed.")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument(
        "--dreamer-checkpoint",
        type=str,
        default=None,
        help="Path to a trained CarDreamer/DreamerV3 checkpoint. When set, POMCPOW plans "
        "inside the learned Dreamer world model instead of the factored placeholder.",
    )
    parser.add_argument(
        "--dreamer-size",
        type=str,
        default="medium",
        help="DreamerV3 config size preset the checkpoint was trained at (e.g. small/medium/large).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Plan with POMCPOW on CARLA and cache the driven ego's chase-camera footage."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    num_steps = round(args.seconds / DT)

    world = build_world(seed=args.seed, host=args.host, port=args.port)
    # The Dreamer model seeds its belief posterior from the world's first real observation,
    # so fetch it up front (this resets the live world); the factored model ignores it.
    initial_observation = world.initial_observation_dist().sample(1)[0]
    model = build_model(
        dreamer_checkpoint=args.dreamer_checkpoint,
        dreamer_size=args.dreamer_size,
        initial_observation=initial_observation,
    )
    planner = build_planner(model, depth=args.depth)
    belief = seed_belief(model, args.n_particles, initial_observation)

    # Separate each run's output by the world and planner-side model used, so a factored
    # run and a Dreamer run never overwrite each other's video.
    cache_dir = args.cache_dir / f"{type(world).__name__}_{type(model).__name__}"

    which = "Dreamer world model" if args.dreamer_checkpoint else "factored placeholder model"
    print(
        f"Planning {args.seconds:.1f}s ({num_steps} ticks @ {DT}s) with POMCPOW on CARLA "
        f"using the {which}..."
    )
    visualize_planner_episode(
        planner=planner,
        environment=world,
        belief=belief,
        n_episodes=1,
        cache_dir=cache_dir,
        num_steps=num_steps,
        n_jobs=1,
    )
    video_path = cache_dir / planner.name / "agent_path_0.mp4"
    print(f"\nDone | saved chase-camera video to {video_path}")


if __name__ == "__main__":
    main()
