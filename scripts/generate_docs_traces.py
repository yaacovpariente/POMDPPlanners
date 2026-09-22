# SPDX-License-Identifier: MIT
"""Regenerate the episode traces the environment docs replay in 3D.

Each environment page embeds a live viewer (``.. episode-viewer::``) that
replays one real episode. This script produces those episodes: one short
episode per world, planned by PFT-DPW with a fixed simulation count, run
through :class:`LocalSimulationsAPI` like every other run in this project, so
the trace is what a real planner did rather than a scripted demo.

Seeds are fixed. The initial belief is sampled under a seed derived from the
world's slug, the simulator seeds each episode from the environment name, the
policy name and the episode index, and PFT-DPW runs a fixed number of
simulations per decision rather than a wall-clock budget. Worlds that run in
Python and NumPy alone then write the same trace on every run (checked: tiger,
battleship, capture the flag, chicheck invaders, the three mazes, firefighting,
occupancy, push, snake). Worlds whose dynamics run in a C++ ``_native`` module
(cartpole, laser tag, light-dark, mountain car, pacman, rock sample, safety
ant, continuous push) do not: each native module keeps its own random
generator, which the simulator does not seed, so their episodes differ from
run to run.

The run itself lands in ``results/docs-traces/`` (never committed). Only each
world's ``trace_0.json`` is copied, re-serialised without whitespace, to
``docs/environments/traces/<world>.json``.

Usage::

    python scripts/generate_docs_traces.py                  # every world
    python scripts/generate_docs_traces.py --only tiger push
    python scripts/generate_docs_traces.py --n-jobs 16

Racetrack steps highway-env, so it needs ``pip install highway-env``; the
others need only the package. Racetrack is a realistic world: the planner
plans on ``KnownTrackModel`` (the package's own model of the same circuit)
while the episode steps the real simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "docs" / "environments" / "traces"
RESULTS_DIR = REPO_ROOT / "results" / "docs-traces"

# The simulator requires policy names unique across a batch, so each world's
# planner is named after it: ``PFT_DPW_tiger``.
POLICY_PREFIX = "PFT_DPW"
# Beliefs are most of a trace's bytes, so the particle count is what keeps the
# committed files small.
N_PARTICLES = 50
# Decimal places kept in the committed copy; far below what a replay can show.
FLOAT_DECIMALS = 5


def _laser_tag_actions() -> List[np.ndarray]:
    """``[dx, dy, tag_flag]`` proposals for the continuous LaserTag world.

    The generic unit-circle sampler draws a 2-vector, which that world reads as
    a move with the tag flag missing, so the planner could never tag. Eight unit
    moves plus a tag give it the whole choice. A list keeps the sampler the
    package's own picklable ``DiscreteActionSampler``: the run's cache hashes
    the policy, and a class defined in this script could not be pickled.
    """
    moves = [
        np.array([math.cos(a), math.sin(a), 0.0])
        for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    ]
    return moves + [np.array([0.0, 0.0, 1.0])]


@dataclass(frozen=True)
class DocsWorld:
    """One world a docs page replays.

    Attributes:
        slug: File stem of the committed trace, ``docs/environments/traces/<slug>.json``.
        build: Returns ``(world, planner model, initial belief)``. The model is the
            world itself except for Racetrack.
        num_steps: Episode horizon; kept short so the trace stays small.
        depth: PFT-DPW search depth.
        n_simulations: PFT-DPW simulations per decision.
    """

    slug: str
    build: Callable[[], Tuple[Any, Any, Any]]
    num_steps: int
    depth: int = 10
    n_simulations: int = 1000


def _env(
    module: str, cls: str, n_particles: int = N_PARTICLES, **kwargs: Any
) -> Callable[[], Tuple[Any, Any, Any]]:
    def build() -> Tuple[Any, Any, Any]:
        import importlib

        from POMDPPlanners.utils.belief_factory import create_environment_belief

        env = getattr(importlib.import_module(module), cls)(**kwargs)
        return env, env, create_environment_belief(env, n_particles=n_particles)

    return build


def _battleship() -> Tuple[Any, Any, Any]:
    """Battleship with the exact-layout belief its trace exporter is written for.

    The factory's default, the vectorized particle belief, carries
    ``consistent_indices`` as an array rather than a method, and the exporter
    fails on it ("'numpy.ndarray' object is not callable"), writing no trace.
    """
    from POMDPPlanners.environments.battleship_pomdp import BattleshipPOMDP
    from POMDPPlanners.environments.battleship_pomdp.battleship_belief import BattleshipBelief

    env = BattleshipPOMDP(board_size=5, ship_lengths=(3, 2, 2))
    return env, env, BattleshipBelief.from_environment(env, n_particles=N_PARTICLES)


def _racetrack() -> Tuple[Any, Any, Any]:
    """The realistic pair: highway-env world, KnownTrackModel planner.

    Wired as the package's own end-to-end episode test wires it: the map is
    walked from the world's first reset, and the belief is seeded around that
    true start. ``seed=0`` makes the first reset in the worker land on the same
    start, because an unpickled world resets with its seed again.
    """
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_belief import TrackedAgentsBelief
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import (
        KnownTrackModel,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import RacetrackPOMDP
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import EGO_STATE_WIDTH
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
        geometry_from_world,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0)
    start = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    geometry, _ = geometry_from_world(world)
    model = KnownTrackModel(
        discount_factor=0.95,
        track_geometry=geometry,
        observation_mode=world.observation_mode,
        max_tracked_agents=world.max_tracked_agents,
        max_detection_range_m=world.max_detection_range_m,
    )
    rng = np.random.default_rng(0)
    particles = np.tile(start, (N_PARTICLES, 1))
    particles[:, :EGO_STATE_WIDTH] += rng.normal(0.0, 0.05, size=(N_PARTICLES, EGO_STATE_WIDTH))
    belief = TrackedAgentsBelief(
        particles=particles,
        log_weights=np.full(N_PARTICLES, -np.log(N_PARTICLES)),
        observation_mode=world.observation_mode,
        max_tracked_agents=world.max_tracked_agents,
    )
    # A fresh world, so the one pickled into the run resets with seed=0 first.
    fresh = RacetrackPOMDP(discount_factor=0.95, seed=0)
    return fresh, model, belief


ENV = "POMDPPlanners.environments"
WORLDS: List[DocsWorld] = [
    DocsWorld("tiger", _env(ENV, "TigerPOMDP", discount_factor=0.95), num_steps=12),
    DocsWorld("battleship", _battleship, num_steps=25),
    DocsWorld(
        "capture_the_flag",
        _env(ENV, "CaptureTheFlagPOMDP", grid_size=(9, 7), midline=4, n_blue=2, n_red=2),
        num_steps=30,
    ),
    DocsWorld(
        "cartpole",
        _env(ENV, "CartPolePOMDP", discount_factor=0.95, noise_cov=np.eye(4) * 0.01),
        num_steps=40,
    ),
    DocsWorld("chicheck_invaders", _env(ENV, "ChicheckInvadersPOMDP"), num_steps=30),
    DocsWorld("laser_tag", _env(ENV, "LaserTagPOMDP", discount_factor=0.95), num_steps=25),
    DocsWorld(
        "continuous_laser_tag",
        _env(ENV, "ContinuousLaserTagPOMDP", discount_factor=0.95),
        num_steps=25,
    ),
    DocsWorld(
        "light_dark", _env(ENV, "ContinuousLightDarkPOMDP", discount_factor=0.95), num_steps=25
    ),
    DocsWorld("discrete_maze", _env(ENV, "DiscreteMazePOMDP", discount_factor=0.95), num_steps=30),
    DocsWorld(
        "continuous_maze", _env(ENV, "ContinuousMazePOMDP", discount_factor=0.95), num_steps=30
    ),
    DocsWorld("t_maze", _env(ENV, "TMazePOMDP", discount_factor=0.95), num_steps=25),
    DocsWorld(
        "mountain_car", _env(ENV, "MountainCarPOMDP", discount_factor=0.99), num_steps=60, depth=20
    ),
    DocsWorld("multiagent_firefighting", _env(ENV, "MultiAgentFirefightingPOMDP"), num_steps=20),
    DocsWorld(
        "occupancy_grid_mapping",
        # A particle is a whole map, so fewer of them keep the trace small.
        _env(ENV, "OccupancyGridMappingPOMDP", n_particles=20),
        num_steps=30,
    ),
    DocsWorld("pacman", _env(ENV, "PacManPOMDP", maze_size=(7, 7), num_ghosts=1), num_steps=30),
    DocsWorld("push", _env(ENV, "PushPOMDP", discount_factor=0.95, grid_size=10), num_steps=30),
    DocsWorld(
        "continuous_push",
        _env(
            f"{ENV}.push_pomdp.continuous_push_pomdp", "ContinuousPushPOMDP", discount_factor=0.95
        ),
        num_steps=30,
    ),
    DocsWorld(
        "rock_sample",
        _env(ENV, "RockSamplePOMDP", map_size=(5, 5), rock_positions=[(0, 0), (2, 2), (3, 3)]),
        num_steps=30,
    ),
    DocsWorld(
        "safety_ant_velocity",
        _env(ENV, "SafeAntVelocityPOMDP", discount_factor=0.95, safe_velocity_threshold=2.0),
        num_steps=30,
    ),
    DocsWorld(
        "snake",
        _env(f"{ENV}.snake_pomdp", "SnakePOMDP", grid_size=12, target_length=10),
        num_steps=40,
    ),
    DocsWorld("racetrack", _racetrack, num_steps=40, n_simulations=300),
]


def _action_sampler(model: Any) -> Any:
    """Discrete actions when the model lists them, else a continuous sampler."""
    from POMDPPlanners.utils.action_samplers import DiscreteActionSampler, UnitCircleActionSampler

    if type(model).__name__ == "ContinuousLaserTagPOMDP":
        return DiscreteActionSampler(_laser_tag_actions())
    actions = model.get_actions() if hasattr(model, "get_actions") else None
    if actions:
        return DiscreteActionSampler(list(actions))
    return UnitCircleActionSampler()


def _run_params(world_spec: DocsWorld) -> Any:
    from POMDPPlanners.core.simulation import EnvironmentRunParams
    from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW

    # The belief is sampled here, in this process, before any episode seed is
    # set; without its own seed two regenerations start from different beliefs.
    seed = zlib.crc32(world_spec.slug.encode("utf-8"))
    random.seed(seed)
    np.random.seed(seed)
    world, model, belief = world_spec.build()
    policy = PFT_DPW(
        environment=model,
        discount_factor=model.discount_factor,
        depth=world_spec.depth,
        name=f"{POLICY_PREFIX}_{world_spec.slug}",
        action_sampler=_action_sampler(model),
        n_simulations=world_spec.n_simulations,
    )
    return EnvironmentRunParams(
        environment=world,
        belief=belief,
        policies=[policy],
        num_episodes=1,
        num_steps=world_spec.num_steps,
    )


def _compact(value: Any) -> Any:
    """Whole floats as integers, others to ``FLOAT_DECIMALS`` places.

    JavaScript has one number type, so ``1.0`` and ``1`` replay identically;
    the long tails of float64 were half of every trace's size.
    """
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer() and abs(value) < 2**53:
            return int(value)
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, list):
        return [_compact(v) for v in value]
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    return value


def _find_trace(results_dir: Path, env_name: str, slug: str, started: float) -> Optional[Path]:
    """This run's trace for one world, or ``None``.

    The results directory keeps every earlier run's MLflow artifacts, so only a
    trace written since this run started counts. Otherwise a failed export
    would silently republish an old episode.
    """
    policy = f"{POLICY_PREFIX}_{slug}"
    matches = [
        path
        for path in results_dir.glob(f"**/{env_name}/{policy}/**/trace_0.json")
        if path.stat().st_mtime >= started
    ]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--only", nargs="*", help="Slugs to regenerate (default: all).")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
        LocalSimulationsAPI,
    )

    chosen = [w for w in WORLDS if not args.only or w.slug in args.only]
    unknown = set(args.only or []) - {w.slug for w in WORLDS}
    if unknown:
        parser.error(f"unknown slugs: {sorted(unknown)}")

    params = [_run_params(w) for w in chosen]
    names = [p.environment.name for p in params]
    if len(set(names)) != len(names):
        raise SystemExit(f"environment names must be unique per run: {names}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    api = LocalSimulationsAPI(cache_dir_path=args.results_dir)
    api.run_multiple_environments_and_policies(
        environment_run_params=params,
        alpha=0.1,
        confidence_interval_level=0.95,
        experiment_name="docs_episode_traces",
        n_jobs=args.n_jobs,
        cache_dir_path=args.results_dir,
    )

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    rows: List[Dict[str, Any]] = []
    for spec, name in zip(chosen, names):
        source = _find_trace(args.results_dir, name, spec.slug, started)
        if source is None:
            failed.append(spec.slug)
            continue
        trace = json.loads(source.read_text(encoding="utf-8"))
        target = TRACES_DIR / f"{spec.slug}.json"
        compact = json.dumps(_compact(trace), separators=(",", ":"), allow_nan=False)
        target.write_text(compact + "\n", encoding="utf-8")
        rows.append(
            {
                "slug": spec.slug,
                "payload_kind": trace.get("payload_kind"),
                "steps": trace.get("num_steps"),
                "return": trace.get("discounted_return"),
                "terminal": trace.get("reach_terminal_state"),
                "bytes": target.stat().st_size,
            }
        )
        shutil.copy2(source, args.results_dir / f"{spec.slug}.trace_0.json")

    for row in rows:
        print(
            f"{row['slug']:<26} {row['payload_kind']:<28} steps={row['steps']:<4} "
            f"return={row['return']:.2f} terminal={row['terminal']} {row['bytes'] / 1024:.1f} KiB"
        )
    if failed:
        print(f"No trace written for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
