# SPDX-License-Identifier: MIT

"""Tests for environment visualization consistency and correctness.

This module tests that visualizers produce deterministic, reproducible outputs
when given identical episode histories. Uses golden file testing approach where
reference GIF files are stored and compared against new outputs.

Golden File Testing Workflow:
    1. First run: If golden file doesn't exist, generate and save it
    2. Subsequent runs: Compare new output against golden file using hash
    3. Update golden files: Delete old golden file and re-run test to regenerate

Every environment is one entry in :data:`GOLDEN_VISUALIZATIONS`, naming how
its episode is built and how it is rendered. The two tests at the bottom of
this file are parametrized over that registry, so adding an environment means
adding an entry rather than pasting two more test methods.

Directory Structure:
    One ``<name>_visualization.gif`` per registry entry, in
    ``POMDPPlanners/tests/test_environments/golden_visualizations/``. Read the
    registry or list the directory rather than a copy of the list here, which
    was eleven names out of date by the time it was replaced.
"""

import hashlib
import random
import shutil
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple, cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.snake_pomdp.snake_belief import SnakeBelief
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakeAction, SnakePOMDP
from POMDPPlanners.environments.snake_pomdp.visualizer.snake_visualizer import SnakeVisualizer
from POMDPPlanners.environments.battleship_pomdp.battleship_belief import BattleshipBelief
from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import BattleshipPOMDP
from POMDPPlanners.environments.battleship_pomdp.battleship_visualizer import (
    BattleshipVisualizer,
)
from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_visualizer import (
    CaptureTheFlagVisualizer,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (
    RockSampleVisualizer,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.visualizer.pacman_visualizer import PacManVisualizer
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.visualizer.light_dark_visualizer import (
    LightDarkPOMDPVisualizer,
)
from POMDPPlanners.environments.maze_pomdp import (
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeVisualizer,
)
from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    ChicheckInvadersPOMDP,
    ChicheckInvadersVisualizer,
    create_chicheck_invaders_belief,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
    FireCategory,
    FirefightingAction,
    MultiAgentFirefightingPOMDP,
    MultiAgentFirefightingVisualizer,
    WindDirection,
    WindStrength,
    create_firefighting_state,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridAction,
    OccupancyGridMappingBelief,
    OccupancyGridMappingPOMDP,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.visualizer.occupancy_grid_mapping_visualizer import (  # noqa: E501
    OccupancyGridMappingVisualizer,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import (
    PushPOMDPVisualizer,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualizer import (
    LaserTagVisualizer,
)
from POMDPPlanners.environments.laser_tag_pomdp import (
    _native as _laser_tag_native,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
)
from POMDPPlanners.environments.push_pomdp import _native as _push_native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
)
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp_visualizer import (
    ContinuousPushPOMDPVisualizer,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
    SafeAntVelocityPOMDP,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.visualizer import (
    SafeAntVelocityVisualizer,
)
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import (
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    TMazePOMDP,
    create_t_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    ACTION_OFFSETS,
    create_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualizer import MazeVisualizer
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    battleship_pinned_kwargs,
    chicheck_invaders_pinned_kwargs,
    continuous_laser_tag_pinned_kwargs,
    continuous_light_dark_pinned_kwargs,
    continuous_maze_pinned_kwargs,
    continuous_push_pinned_kwargs,
    discrete_maze_pinned_kwargs,
    laser_tag_pinned_kwargs,
    multiagent_firefighting_pinned_kwargs,
    occupancy_grid_mapping_pinned_kwargs,
    pacman_pinned_kwargs,
    push_pinned_kwargs,
    rock_sample_pinned_kwargs,
    safety_ant_velocity_pinned_kwargs,
    snake_pinned_kwargs,
    t_maze_pinned_kwargs,
)


# Golden files directory
GOLDEN_DIR = Path(__file__).parent / "golden_visualizations"


def _create_mock_belief(state: Any) -> WeightedParticleBelief:
    """Create a simple mock belief for visualization purposes.

    Args:
        state: Current state

    Returns:
        Mock belief centered on the current state
    """
    return WeightedParticleBelief(
        particles=[state],
        log_weights=np.array([1.0]),  # Non-zero log weight (unnormalized)
    )


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for comparison.

    Args:
        file_path: Path to file to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compare_or_create_golden_file(output_path: Path, golden_name: str, test_name: str) -> None:
    """Compare output file against golden file, or create golden file if missing.

    This function implements the golden file testing pattern:
    - If golden file exists: Compare hashes and fail if different
    - If golden file doesn't exist: Create it from output and warn user

    Args:
        output_path: Path to newly generated output file
        golden_name: Name of golden file (e.g., "rock_sample_visualization.gif")
        test_name: Name of test for informative messages

    Raises:
        AssertionError: If golden file exists but hash doesn't match
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / golden_name

    if not golden_path.exists():
        # Golden file doesn't exist - create it
        shutil.copy(output_path, golden_path)
        warnings.warn(
            f"\n{'='*70}\n"
            f"GOLDEN FILE CREATED: {golden_path}\n"
            f"{'='*70}\n"
            f"This is the first run of {test_name}.\n"
            f"A new golden reference file has been created at:\n"
            f"  {golden_path}\n\n"
            f"This file will be used for comparison in future test runs.\n"
            f"Please review the generated GIF to ensure it's correct.\n\n"
            f"To regenerate golden files:\n"
            f"  1. Delete the golden file: rm {golden_path}\n"
            f"  2. Re-run the test to create a new golden file\n"
            f"{'='*70}\n",
            UserWarning,
        )
        pytest.skip(f"Golden file created for {test_name}. Re-run test to validate.")
    else:
        # Golden file exists - compare hashes
        output_hash = compute_file_hash(output_path)
        golden_hash = compute_file_hash(golden_path)

        assert output_hash == golden_hash, (
            f"\n{'='*70}\n"
            f"VISUALIZATION OUTPUT CHANGED!\n"
            f"{'='*70}\n"
            f"Test: {test_name}\n"
            f"Golden file: {golden_path}\n"
            f"Output file: {output_path}\n\n"
            f"Expected hash: {golden_hash}\n"
            f"Actual hash:   {output_hash}\n\n"
            f"The visualization output has changed. This could mean:\n"
            f"  1. You intentionally changed visualization logic (expected)\n"
            f"  2. A bug was introduced causing different output (unexpected)\n"
            f"  3. Dependencies changed (matplotlib, pillow versions)\n\n"
            f"To update the golden file if change is intentional:\n"
            f"  1. Review the new output: {output_path}\n"
            f"  2. If correct, delete old golden: rm {golden_path}\n"
            f"  3. Re-run test to create new golden file\n"
            f"{'='*70}\n"
        )


def build_chicheck_invaders_env() -> ChicheckInvadersPOMDP:
    """Build the environment the Chicheck Invaders golden GIF is rendered for."""
    return ChicheckInvadersPOMDP(discount_factor=0.95, **chicheck_invaders_pinned_kwargs())


def create_deterministic_chicheck_invaders_episode(seed: int = 11) -> List[StepData]:
    """Create a deterministic Chicheck Invaders episode for the golden GIF.

    The action sequence is fixed and the belief attached to each step is a real
    :class:`ChicheckInvadersBelief` rather than a mock, for the reason the
    occupancy-grid fixture gives: the belief panel is the part of this
    visualization most likely to regress, and hashing a mock belief would leave
    it untested.

    Both RNGs are seeded, not just NumPy, for the reason the Battleship fixture
    documents: ``conftest`` seeds the stdlib ``random`` once at import, so
    seeding NumPy alone would leave the golden hash dependent on which tests ran
    before this one.

    Args:
        seed: Random seed pinning the flock, the dives, the sensor noise and the
            filter's resampling.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_chicheck_invaders_env()
    belief = create_chicheck_invaders_belief(env, n_particles=60)
    state = env.initial_state_dist().sample()[0]
    history: List[StepData] = []

    # Shoot, sidestep, shoot again: the frames show a beam flashing up the
    # ship's column on the steps that fired, the ship moving under the flock,
    # and the belief tightening as the sensors report. Indices follow
    # ChicheckInvadersAction: 0 stay, 1 left, 2 right, 3 fire.
    action_sequence = [1, 3, 0, 2, 3, 2, 0, 3, 1, 3, 0, 3]

    for action in action_sequence:
        if env.is_terminal(state):
            break
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=belief,
                info=env.step_info(state, action, next_state),
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
        state = next_state

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=env.step_info(state, None, None),
        )
    )
    return history


def build_multiagent_firefighting_env() -> MultiAgentFirefightingPOMDP:
    """Build the environment the multi-agent firefighting golden GIF is rendered for."""
    return MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs()
    )


def create_deterministic_multiagent_firefighting_episode(seed: int = 5) -> List[StepData]:
    """Create a deterministic firefighting episode for the golden GIF.

    The belief attached to each step is a real :class:`WeightedParticleBelief`
    rather than a mock, for the reason the occupancy-grid fixture documents:
    two of this renderer's three panels are belief projections -- the per-cell
    P(alight) heatmap and the histogram over the eight hidden wind values --
    and hashing a mock belief would leave both untested.

    Both RNGs are seeded, not just NumPy, for the reason the Battleship fixture
    documents: ``conftest`` seeds the stdlib ``random`` once at import, so
    seeding NumPy alone would leave the golden hash dependent on which tests
    ran before this one.

    Args:
        seed: Random seed pinning the wind, the ignition cell, the spread, the
            sensor noise and the filter's resampling.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_multiagent_firefighting_env()

    # The true world is stated rather than drawn. The reset distribution puts
    # the fire in a uniformly random cell, and on most seeds that is nowhere
    # near the robots' fixed tour -- the fire then burns itself out off-screen
    # and the golden file shows eight frames of nothing happening. Pinning it
    # two cells south-east of the robots, with an easterly wind carrying it
    # away from them, is what makes the frames show suppression, heat damage
    # and a belief narrowing onto a real front.
    fire = np.full((env.num_rows, env.num_cols), float(FireCategory.UNBURNT))
    fire[4, 4] = float(FireCategory.BURNING)
    fire[4, 5] = float(FireCategory.SMOLDERING)
    state = create_firefighting_state(
        env,
        robots=[(2, 2, env.max_tank, env.max_health), (2, 3, env.max_tank, env.max_health)],
        wind=(int(WindDirection.EAST), int(WindStrength.HIGH)),
        fire=fire,
    )
    # The belief starts knowing the fire and not the wind: four particles per
    # wind value (twelve, at ninety-six particles), all carrying the true map. That is a deliberate choice for a
    # *fixture*, not a claim about the filter. A bootstrap particle filter run
    # from the reset prior over whole 100-cell maps is degenerate here -- no
    # prior particle ever matches the observed front, every weight hits the
    # epsilon floor, and both belief panels render as noise, which would leave
    # the thing this GIF exists to check untested. Starting from the known fire
    # isolates the half of the belief the environment is actually about: the
    # histogram has to move off flat as the robots watch which way the fire
    # grows, and if the observation model or the update stops working it will
    # not.
    particles = [
        create_firefighting_state(
            env,
            robots=[
                (2, 2, env.max_tank, env.max_health),
                (2, 3, env.max_tank, env.max_health),
            ],
            wind=(index % len(WindDirection), (index // len(WindDirection)) % len(WindStrength)),
            fire=fire,
        )
        for index in range(96)
    ]
    belief = WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.full(len(particles), 1.0 / len(particles))),
        resampling=True,
    )

    # A fixed tour: the robots take up positions north and south of the front
    # and spray it from beside it, which is the intended play -- neither ever
    # stands on an alight cell, so neither takes heat damage. On the fifth
    # step both cover the burning cell at once, which is the cooperation case:
    # two independent attempts at soaking it rather than one. Robot 1's last
    # move is refused by the obstacle blob, so a blocked move is in frame too.
    def joint(first: FirefightingAction, second: FirefightingAction) -> int:
        return int(first) + 5 * int(second)

    action_sequence = [
        joint(FirefightingAction.SOUTH, FirefightingAction.SOUTH),
        joint(FirefightingAction.EAST, FirefightingAction.SOUTH),
        joint(FirefightingAction.EAST, FirefightingAction.SOUTH),
        joint(FirefightingAction.SUPPRESS, FirefightingAction.EAST),
        joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS),
        joint(FirefightingAction.EAST, FirefightingAction.EAST),
        joint(FirefightingAction.SUPPRESS, FirefightingAction.SUPPRESS),
    ]

    history: List[StepData] = []
    for action in action_sequence:
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=env.step_info(state, action, next_state),
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
        state = next_state

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=env.step_info(state, None, None),
        )
    )
    return history


def build_occupancy_grid_mapping_env() -> OccupancyGridMappingPOMDP:
    """Build the environment the occupancy-grid mapping golden GIF is rendered for."""
    return OccupancyGridMappingPOMDP(discount_factor=0.95, **occupancy_grid_mapping_pinned_kwargs())


def create_deterministic_occupancy_grid_mapping_episode(seed: int = 3) -> List[StepData]:
    """Create a deterministic occupancy-grid mapping episode for the golden GIF.

    The action sequence is fixed, and the belief attached to each step is a real
    :class:`OccupancyGridMappingBelief` rather than a mock: the belief panel is the
    part of this visualization most likely to regress, and hashing a mock belief
    would leave it untested. Both the resampling inside the filter and the
    sensor noise are random, so the seed is pinned.

    Both RNGs are seeded, not just NumPy, for the reason the Battleship fixture
    documents: ``conftest`` seeds the stdlib ``random`` once at import, so
    seeding NumPy alone would leave the golden hash dependent on which tests ran
    before this one.

    Args:
        seed: Random seed pinning the true map, the scans and the resampling.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_occupancy_grid_mapping_env()

    belief = OccupancyGridMappingBelief.initial(env, n_particles=24)
    state = env.initial_state_dist().sample()[0]
    history: List[StepData] = []

    # A fixed tour: drive out, turn, drive out again, so the frames show the
    # occupancy grid growing along a path rather than from one vantage point.
    action_sequence = [
        OccupancyGridAction.FORWARD,
        OccupancyGridAction.FORWARD,
        OccupancyGridAction.TURN_RIGHT,
        OccupancyGridAction.FORWARD,
        OccupancyGridAction.FORWARD,
        OccupancyGridAction.TURN_RIGHT,
        OccupancyGridAction.FORWARD,
        OccupancyGridAction.FORWARD,
    ]

    for action in action_sequence:
        next_state, observation, reward = env.sample_next_step(state, int(action))
        history.append(
            StepData(
                state=state,
                action=int(action),
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=env.step_info(state, int(action), next_state),
            )
        )
        belief = belief.update(action=int(action), observation=observation, pomdp=env)
        state = next_state

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=env.step_info(state, None, None),
        )
    )
    return history


def build_snake_env() -> SnakePOMDP:
    """Build the Snake environment the golden visualization is rendered for."""
    return SnakePOMDP(discount_factor=0.98, **snake_pinned_kwargs(target_length=6))


def create_deterministic_snake_episode(seed: int = 5) -> List[StepData]:
    """Create a deterministic Snake episode for the golden GIF.

    The action sequence is fixed, and the belief attached to each step is the
    real :class:`SnakeBelief` rather than a mock: the amber belief layer is the
    part of this visualization most likely to regress, and hashing a mock belief
    would leave it untested. The food draw, the sighting and the scent are all
    random, so both RNGs are seeded for the reason the Battleship fixture
    documents -- ``conftest`` seeds the stdlib ``random`` once at import, so
    seeding NumPy alone would leave the golden hash dependent on which tests ran
    before this one.

    Args:
        seed: Random seed pinning the food, the readings and the particle draws.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_snake_env()

    belief = SnakeBelief.from_environment(env, n_particles=32)
    state = env.initial_state_dist().sample()[0]
    history: List[StepData] = []

    # A fixed tour: run east, turn down, run south, turn back west. It sweeps
    # the window across a good part of the board, so the frames show the belief
    # both spreading under the scent and being cut back by the silent window.
    action_sequence = [
        SnakeAction.GO_STRAIGHT,
        SnakeAction.GO_STRAIGHT,
        SnakeAction.TURN_RIGHT,
        SnakeAction.GO_STRAIGHT,
        SnakeAction.GO_STRAIGHT,
        SnakeAction.TURN_RIGHT,
        SnakeAction.GO_STRAIGHT,
        SnakeAction.GO_STRAIGHT,
        SnakeAction.TURN_RIGHT,
        SnakeAction.GO_STRAIGHT,
    ]

    for action in action_sequence:
        if env.is_terminal(state):
            break
        next_state, observation, reward = env.sample_next_step(state, int(action))
        history.append(
            StepData(
                state=state,
                action=int(action),
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=env.step_info(state, int(action), next_state),
            )
        )
        belief = belief.update(action=int(action), observation=observation, pomdp=env)
        state = next_state

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=env.step_info(state, None, None),
        )
    )
    return history


def build_battleship_env() -> BattleshipPOMDP:
    """Build the Battleship environment the golden visualization is rendered for."""
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


def create_deterministic_battleship_episode(seed: int = 7) -> List[StepData]:
    """Create a deterministic Battleship episode for visualization testing.

    The probe sequence is fixed, and the belief attached to each step is the real
    :class:`BattleshipBelief` rather than a mock: the belief panel is the part of
    the visualization most likely to regress, so hashing a mock belief would
    leave it untested. The belief update is deterministic given the observations,
    but drawing its particles is not, so the seed is pinned.

    Both RNGs are seeded, not just NumPy. ``WeightedParticleBelief.sample`` draws
    the true board with the stdlib ``random``, which ``conftest`` seeds once at
    import rather than per test — so seeding NumPy alone would leave the fleet,
    the observations and therefore the golden hash dependent on which tests ran
    first. The repeated-render determinism test cannot catch that, because it
    renders one already-built history twice.

    Args:
        seed: Random seed pinning the initial board and the particle draws.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_battleship_env()

    belief = BattleshipBelief.from_environment(env, n_particles=32)
    state = belief.sample()
    history: List[StepData] = []

    # A fixed scan: two corners, then a diagonal sweep across the board.
    action_sequence = [0, 4, 6, 12, 18, 24, 7, 11, 13, 17]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=belief,
            )
        )
        belief = belief.update(action=action, observation=obs, pomdp=env)
        state = next_state
        if env.is_terminal(state):
            break

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
        )
    )
    return history


def create_deterministic_rock_sample_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic RockSample episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    np.random.seed(seed)
    env = RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(
            map_size=(5, 5),
            rock_positions=[(1, 1), (2, 3), (4, 2)],
            dangerous_areas=[(2, 2)],
            dangerous_area_radius=1.0,
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence
    action_sequence = [1, 1, 2, 0, 1, 3, 3, 4]  # Move and sample

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_pacman_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic PacMan episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from POMDPPlanners.environments.pacman_pomdp import _native as _pacman_native

    np.random.seed(seed)
    # Transition / observation sampling now runs on the C++ RNG (see
    # PacManTransitionCpp / PacManObservationCpp); seeding _native keeps
    # the episode deterministic end-to-end.
    _pacman_native.set_seed(seed)
    env = PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(7, 7),
            num_ghosts=2,
            initial_ghost_positions=None,
            ghost_strategies=None,
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence
    action_sequence = [0, 0, 1, 1, 2, 2, 3, 3]  # Move in pattern

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_light_dark_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic LightDark episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from POMDPPlanners.environments.light_dark_pomdp import _native as _ld_native

    np.random.seed(seed)
    # Transition / observation sampling runs on the C++ RNG now that the
    # continuous light-dark models inherit from the native extension;
    # seeding _native keeps the episode deterministic end-to-end.
    _ld_native.set_seed(seed)
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        **continuous_light_dark_pinned_kwargs(),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (continuous 2D actions)
    action_sequence = [
        np.array([0.5, 0.0]),
        np.array([-0.5, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 0.5]),
        np.array([0.5, -0.5]),
        np.array([-0.5, 0.5]),
        np.array([0.0, 0.0]),
    ]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_push_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic Push episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    np.random.seed(seed)
    env = PushPOMDP(
        discount_factor=0.95,
        **push_pinned_kwargs(
            grid_size=8,
            transition_error_prob=0.0,  # Explicitly set for deterministic behavior
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (discrete actions: 'up', 'down', 'right', 'left')
    # Actions are strings, not arrays
    action_sequence = ["right", "right", "down", "down", "left", "left", "up", "up"]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_laser_tag_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic LaserTag episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    np.random.seed(seed)
    env = LaserTagPOMDP(
        discount_factor=0.95,
        **laser_tag_pinned_kwargs(
            transition_error_prob=0.0,  # Explicitly set for deterministic behavior
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (5 actions total: 0-4)
    action_sequence = [0, 1, 2, 3, 4, 0, 1, 2]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_safety_ant_velocity_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic SafeAntVelocity episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from POMDPPlanners.environments.safety_ant_velocity_pomdp import _native as _sa_native

    np.random.seed(seed)
    # Transition / observation sampling now runs on the C++ RNG (see
    # SafeAntVelocityTransitionCpp / SafeAntVelocityObservationCpp); seeding
    # _native keeps the episode deterministic end-to-end.
    _sa_native.set_seed(seed)
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        **safety_ant_velocity_pinned_kwargs(),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (4 discrete actions: 0-3)
    action_sequence = [0, 1, 2, 3, 0, 1, 2, 3]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_continuous_laser_tag_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic Continuous LaserTag episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    np.random.seed(seed)
    # Continuous LaserTag transitions now execute in C++ via _native; seed
    # that RNG too so the episode is fully deterministic. numpy's RNG is
    # still used for initial state rejection sampling.
    _laser_tag_native.set_seed(seed)
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        **continuous_laser_tag_pinned_kwargs(
            robot_transition_cov_matrix=np.eye(2) * 0.01,
            opponent_transition_cov_matrix=np.eye(2) * 0.01,
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (continuous 3D: [dx, dy, tag_flag])
    action_sequence = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([-0.5, 0.5, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),  # tag attempt
    ]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_continuous_push_episode(seed: int = 42) -> List[StepData]:
    """Create deterministic Continuous Push episode for testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of StepData objects representing episode history
    """
    np.random.seed(seed)
    # Continuous Push transition/observation sampling runs through the
    # native C++ module's std::mt19937_64, so seeding numpy alone no
    # longer fixes the sample sequence. Seed both to keep the golden
    # visualization reproducible.
    _push_native.set_seed(seed)
    env = ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            grid_size=10,
            obstacles=[(3.0, 3.0, 0.5), (6.0, 6.0, 0.5)],
            state_transition_cov_matrix=np.eye(2) * 0.01,
        ),
    )

    state = env.initial_state_dist().sample()[0]
    history = []

    # Execute fixed action sequence (continuous 2D vectors)
    action_sequence = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([0.5, 0.5]),
    ]

    for action in action_sequence:
        next_state, obs, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=obs,
                reward=reward,
                belief=_create_mock_belief(state),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    # Add terminal state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_mock_belief(state),
        )
    )

    return history


def create_deterministic_t_maze_episode() -> List[StepData]:
    """Create a deterministic T-Maze episode with a genuinely spread belief.

    The action sequence walks up the stem, past the cue cell, to the junction and
    then into the left arm, so the frames cover every phase the visualizer draws:
    cue unseen, cue emitting, cue consumed, and a terminal endpoint.

    The belief is built by hand rather than by running a filter, and it is
    deliberately *not* a point mass: it carries one particle per goal side, with
    the weights swinging to 0.9 / 0.1 on the step that reads the cue. A point-mass
    belief would render the same picture whether the goal-side view worked or not.

    The fixture is also kept physically consistent with the environment it claims
    to come from, because the picture it pins is the one a human reviews. Its
    particles sit on the agent's own cell rather than staying at the start (position
    is observable here, so a filter's particles track the agent), and the step that
    enters the cue cell reports ``left_cue`` rather than ``empty`` — that reading is
    what justifies the belief swinging to 0.9, and an episode that swung on an
    ``empty`` could not have happened.

    Returns:
        The episode's step records, terminal bookkeeping step included.
    """
    env = TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs())
    goal_side = GOAL_LEFT
    state = create_t_maze_state(env.start_cell, goal_side)
    actions = ["up", "up", "up", "up", "left"]
    # Weight on the true (left) side per step: flat until the cue is read on the
    # first move, then held through the corridor, as a correct filter would.
    left_weights = [0.5, 0.9, 0.9, 0.9, 0.9]
    # The first action steps onto the cue cell, so that step — and only that step —
    # returns a cue. Every later step in the corridor is silent.
    observations = [OBSERVATION_LEFT_CUE] + [OBSERVATION_EMPTY] * (len(actions) - 1)

    history: List[StepData] = []
    for action, left_weight, observation in zip(actions, left_weights, observations):
        next_state = env.sample_next_state(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=env.reward(state, action, next_state),
                belief=_create_t_maze_belief(state, left_weight),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_create_t_maze_belief(state, left_weights[-1]),
        )
    )
    return history


def _create_t_maze_belief(state: np.ndarray, left_weight: float) -> WeightedParticleBelief:
    """A two-particle belief on ``state``'s own cell, ``left_weight`` on the left goal.

    Both particles carry the agent's position and cue phase because position is
    observable in this maze: a filter's particles agree about where the agent is and
    disagree only about the goal side.
    """
    position = (int(state[0]), int(state[1]))
    cue_phase = float(state[3])
    return WeightedParticleBelief(
        particles=[
            create_t_maze_state(position, GOAL_LEFT, cue_phase),
            create_t_maze_state(position, GOAL_RIGHT, cue_phase),
        ],
        log_weights=np.log(np.array([left_weight, 1.0 - left_weight])),
    )


def create_deterministic_maze_episode(env) -> List[StepData]:
    """Walk a generated Maze to its left goal with a two-particle goal belief."""
    parents = {env.start_cell: None}
    queue = deque([env.start_cell])
    while queue:
        cell = queue.popleft()
        if cell == env.left_goal_cell:
            break
        for neighbour in env.geometry.neighbours(cell):
            if neighbour not in parents:
                parents[neighbour] = cell
                queue.append(neighbour)
    cells = []
    cell = env.left_goal_cell
    while cell is not None:
        cells.append(cell)
        cell = parents[cell]
    cells.reverse()

    state = create_maze_state(env.start_cell, GOAL_LEFT)
    history: List[StepData] = []
    for index, (first, second) in enumerate(zip(cells, cells[1:])):
        offset = (second[0] - first[0], second[1] - first[1])
        if isinstance(env, DiscreteMazePOMDP):
            action = next(name for name, delta in ACTION_OFFSETS.items() if delta == offset)
        else:
            action = np.asarray(offset, dtype=np.float64)
        next_state = env.sample_next_state(state, action)
        observation = OBSERVATION_LEFT_CUE if index == 0 else OBSERVATION_EMPTY
        weight = 0.5 if index == 0 else 0.9
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=env.reward(state, action, next_state),
                belief=WeightedParticleBelief(
                    particles=[
                        create_maze_state(state[:2], GOAL_LEFT, state[3]),
                        create_maze_state(state[:2], GOAL_RIGHT, state[3]),
                    ],
                    log_weights=np.log(np.array([weight, 1.0 - weight])),
                ),
            )
        )
        state = next_state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=WeightedParticleBelief(
                particles=[
                    create_maze_state(state[:2], GOAL_LEFT, state[3]),
                    create_maze_state(state[:2], GOAL_RIGHT, state[3]),
                ],
                log_weights=np.log(np.array([0.9, 0.1])),
            ),
        )
    )
    return history


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "visualizations"
    output_dir.mkdir()
    return output_dir


def build_capture_the_flag_env() -> CaptureTheFlagPOMDP:
    """Build the CaptureTheFlag environment the golden episode is rendered from."""
    return CaptureTheFlagPOMDP(discount_factor=0.95)


def create_deterministic_capture_the_flag_episode(seed: int = 0) -> List[StepData]:
    """Roll a short CaptureTheFlag episode with a real weighted particle belief.

    The belief matters here: the visualizer draws the marginals over the red
    players and the red flag cell, so a history carrying placeholder beliefs
    would exercise none of the overlay the golden file is meant to pin.

    Args:
        seed: Seed applied to both random streams before rolling.

    Returns:
        The recorded steps.
    """
    env = build_capture_the_flag_env()
    random.seed(seed)
    np.random.seed(seed)
    # cast: initial_state_dist is typed as the abstract Distribution, whose
    # support is not part of that interface. Pinning one candidate keeps the
    # golden episode deterministic.
    state = cast(DiscreteDistribution, env.initial_state_dist()).values[2]
    particles = [env.initial_state_dist().sample()[0] for _ in range(48)]
    history: List[StepData] = []
    for _ in range(10):
        action = int(np.random.randint(0, len(env.get_actions())))
        next_state = env.sample_next_state(state, action)
        observation = env.sample_observation(next_state, action)
        particles = [env.sample_next_state(particle, action) for particle in particles]
        log_weights = np.array(
            [env.observation_log_probability(p, action, [observation])[0] for p in particles]
        )
        log_weights = np.where(np.isfinite(log_weights), log_weights, -1e9)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=env.reward(state, action, next_state),
                belief=WeightedParticleBelief(
                    particles=particles, log_weights=log_weights, resampling=False
                ),
            )
        )
        state = next_state
        if env.is_terminal(state):
            break
    return history


# What a visualizer's draw call looks like once its environment is built.
Renderer = Callable[[List[StepData], Path], None]


@dataclass(frozen=True)
class GoldenVisualization:
    """One environment's entry in the golden-visualization suite.

    Every environment used to carry its own copy of two near-identical test
    methods, differing only in which episode factory and visualizer they named
    and in four strings. Adding an environment meant pasting both, at the same
    place every other author pasted theirs, which is what made this file
    conflict on every pair of concurrent environment branches.

    An entry names what is genuinely per-environment -- how the episode is
    built and how it is rendered -- and the two tests below are parametrized
    over the registry.

    Attributes:
        name: Slug used for the parametrization id and to derive the golden
            file name, which every environment here follows.
        build_history: Builds the episode to render. Called inside the test, so
            an environment constructed here is constructed at the same point in
            the global RNG stream as it was when each test built its own.
        build_renderer: Builds the environment and its visualizer and returns
            the call that draws a history to a path. It returns a renderer
            rather than rendering, so the determinism test can render twice
            through *one* visualizer instance and catch state a first render
            leaves behind on it -- cached artists, accumulated figure state --
            which is what each per-environment test used to do. Environments do
            not agree on the method name (``create_visualization``,
            ``cache_visualization``, ``create_animation`` and ``render_episode``
            are all in use), so the call is named per environment.
        check_repeated_render: Whether the determinism suite also renders this
            environment twice and compares the bytes. That check is what covers
            renderers outside the Docker image, and it is worth its runtime on
            the ones whose panels average over a particle collection.
    """

    name: str
    build_history: Callable[[], List[StepData]]
    build_renderer: Callable[[], Renderer]
    check_repeated_render: bool = False

    @property
    def golden_file(self) -> str:
        """Name of this environment's golden GIF inside :data:`GOLDEN_DIR`."""
        return f"{self.name}_visualization.gif"


def _renderer_battleship() -> Renderer:
    """Build the Battleship renderer."""
    return BattleshipVisualizer(build_battleship_env()).create_visualization


def _renderer_capture_the_flag() -> Renderer:
    """Build the CaptureTheFlag renderer."""
    return CaptureTheFlagVisualizer(build_capture_the_flag_env()).render_episode


def _renderer_chicheck_invaders() -> Renderer:
    """Build the Chicheck Invaders renderer."""
    return ChicheckInvadersVisualizer(build_chicheck_invaders_env()).create_visualization


def _renderer_continuous_laser_tag() -> Renderer:
    """Build the continuous LaserTag renderer."""
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        **continuous_laser_tag_pinned_kwargs(
            robot_transition_cov_matrix=np.eye(2) * 0.01,
            opponent_transition_cov_matrix=np.eye(2) * 0.01,
        ),
    )
    visualizer = ContinuousLaserTagVisualizer(
        grid_size=env.grid_size,
        walls=env.walls,
        robot_radius=env.robot_radius,
        opponent_radius=env.opponent_radius,
        dangerous_areas=env.dangerous_areas,
        dangerous_area_radius=env.dangerous_area_radius,
    )
    return visualizer.create_visualization


def _renderer_continuous_push() -> Renderer:
    """Build the continuous Push renderer."""
    env = ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            grid_size=10,
            obstacles=[(3.0, 3.0, 0.5), (6.0, 6.0, 0.5)],
            state_transition_cov_matrix=np.eye(2) * 0.01,
        ),
    )
    return ContinuousPushPOMDPVisualizer(env).create_visualization


def _renderer_laser_tag() -> Renderer:
    """Build the discrete LaserTag renderer."""
    env = LaserTagPOMDP(
        discount_factor=0.95,
        **laser_tag_pinned_kwargs(
            transition_error_prob=0.0,  # Explicitly set for deterministic behavior
        ),
    )
    visualizer = LaserTagVisualizer(
        floor_shape=env.floor_shape,
        walls=env.walls,
        dangerous_areas=list(env.dangerous_areas),
        dangerous_area_radius=env.dangerous_area_radius,
    )
    return visualizer.create_visualization


def _renderer_light_dark() -> Renderer:
    """Build the continuous Light-Dark renderer."""
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        **continuous_light_dark_pinned_kwargs(),
    )
    return LightDarkPOMDPVisualizer(env).cache_visualization


def _renderer_multiagent_firefighting() -> Renderer:
    """Build the multi-agent firefighting renderer."""
    visualizer = MultiAgentFirefightingVisualizer(build_multiagent_firefighting_env())
    return visualizer.create_visualization


def _renderer_occupancy_grid_mapping() -> Renderer:
    """Build the occupancy-grid mapping renderer."""
    visualizer = OccupancyGridMappingVisualizer(build_occupancy_grid_mapping_env())
    return visualizer.create_visualization


def _renderer_pacman() -> Renderer:
    """Build the PacMan renderer."""
    env = PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(7, 7),
            num_ghosts=2,
            initial_ghost_positions=None,
            ghost_strategies=None,
        ),
    )
    return PacManVisualizer(env).cache_visualization


def _renderer_push() -> Renderer:
    """Build the discrete Push renderer."""
    env = PushPOMDP(
        discount_factor=0.95,
        **push_pinned_kwargs(
            grid_size=8,
            transition_error_prob=0.0,  # Explicitly set for deterministic behavior
        ),
    )
    return PushPOMDPVisualizer(env).create_visualization


def _renderer_rock_sample() -> Renderer:
    """Build the RockSample renderer."""
    env = RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(
            map_size=(5, 5),
            rock_positions=[(1, 1), (2, 3), (4, 2)],
            dangerous_areas=[(2, 2)],
            dangerous_area_radius=1.0,
        ),
    )
    return RockSampleVisualizer(env).create_visualization


def _renderer_safety_ant_velocity() -> Renderer:
    """Build the SafeAntVelocity renderer."""
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        **safety_ant_velocity_pinned_kwargs(),
    )
    return SafeAntVelocityVisualizer(env).create_animation


def _renderer_snake() -> Renderer:
    """Build the Snake renderer."""
    return SnakeVisualizer(build_snake_env()).create_visualization


def _renderer_t_maze() -> Renderer:
    """Build the T-Maze renderer."""
    env = TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs())
    return MazeVisualizer(env).create_visualization


# Both public Maze variants generate their geometry from a pinned ``maze_seed``,
# so the environment the history is walked on and the one the visualizer draws
# have to be the same object for the golden file to mean anything.
_DISCRETE_MAZE_ENV = DiscreteMazePOMDP(discount_factor=0.95, **discrete_maze_pinned_kwargs())
_CONTINUOUS_MAZE_ENV = ContinuousMazePOMDP(discount_factor=0.95, **continuous_maze_pinned_kwargs())


def _renderer_maze(env: Any) -> Callable[[], Renderer]:
    """Build a renderer factory for one Maze variant.

    Args:
        env: The Maze environment whose geometry the history was walked on.

    Returns:
        A factory returning a renderer that draws a history of ``env``.
    """

    def build() -> Renderer:
        return MazeVisualizer(env).create_visualization

    return build


# Alphabetical by name, for the reason test_registration_lists_are_sorted.py
# gives: two environment branches then insert at different lines.
GOLDEN_VISUALIZATIONS: Tuple[GoldenVisualization, ...] = (
    GoldenVisualization(
        name="battleship",
        build_history=lambda: create_deterministic_battleship_episode(seed=7),
        build_renderer=_renderer_battleship,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="capture_the_flag",
        build_history=lambda: create_deterministic_capture_the_flag_episode(seed=0),
        build_renderer=_renderer_capture_the_flag,
    ),
    GoldenVisualization(
        name="chicheck_invaders",
        build_history=lambda: create_deterministic_chicheck_invaders_episode(seed=11),
        build_renderer=_renderer_chicheck_invaders,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="continuous_laser_tag",
        build_history=lambda: create_deterministic_continuous_laser_tag_episode(seed=42),
        build_renderer=_renderer_continuous_laser_tag,
    ),
    GoldenVisualization(
        name="continuous_maze",
        build_history=lambda: create_deterministic_maze_episode(_CONTINUOUS_MAZE_ENV),
        build_renderer=_renderer_maze(_CONTINUOUS_MAZE_ENV),
    ),
    GoldenVisualization(
        name="continuous_push",
        build_history=lambda: create_deterministic_continuous_push_episode(seed=42),
        build_renderer=_renderer_continuous_push,
    ),
    GoldenVisualization(
        name="discrete_maze",
        build_history=lambda: create_deterministic_maze_episode(_DISCRETE_MAZE_ENV),
        build_renderer=_renderer_maze(_DISCRETE_MAZE_ENV),
    ),
    GoldenVisualization(
        name="laser_tag",
        build_history=lambda: create_deterministic_laser_tag_episode(seed=42),
        build_renderer=_renderer_laser_tag,
    ),
    GoldenVisualization(
        name="light_dark",
        build_history=lambda: create_deterministic_light_dark_episode(seed=42),
        build_renderer=_renderer_light_dark,
    ),
    GoldenVisualization(
        name="multiagent_firefighting",
        build_history=lambda: create_deterministic_multiagent_firefighting_episode(seed=5),
        build_renderer=_renderer_multiagent_firefighting,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="occupancy_grid_mapping",
        build_history=lambda: create_deterministic_occupancy_grid_mapping_episode(seed=3),
        build_renderer=_renderer_occupancy_grid_mapping,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="pacman",
        build_history=lambda: create_deterministic_pacman_episode(seed=42),
        build_renderer=_renderer_pacman,
    ),
    GoldenVisualization(
        name="push",
        build_history=lambda: create_deterministic_push_episode(seed=42),
        build_renderer=_renderer_push,
    ),
    GoldenVisualization(
        name="rock_sample",
        build_history=lambda: create_deterministic_rock_sample_episode(seed=42),
        build_renderer=_renderer_rock_sample,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="safety_ant_velocity",
        build_history=lambda: create_deterministic_safety_ant_velocity_episode(seed=42),
        build_renderer=_renderer_safety_ant_velocity,
    ),
    GoldenVisualization(
        name="snake",
        build_history=lambda: create_deterministic_snake_episode(seed=5),
        build_renderer=_renderer_snake,
        check_repeated_render=True,
    ),
    GoldenVisualization(
        name="t_maze",
        build_history=create_deterministic_t_maze_episode,
        build_renderer=_renderer_t_maze,
    ),
)

REPEATED_RENDER_VISUALIZATIONS: Tuple[GoldenVisualization, ...] = tuple(
    spec for spec in GOLDEN_VISUALIZATIONS if spec.check_repeated_render
)


def test_golden_visualization_registry_is_alphabetical():
    """Test that the golden-visualization registry is in alphabetical order.

    Purpose: This file is where the last three environment branches conflicted,
        because each appended its entry at the same line

    Given: :data:`GOLDEN_VISUALIZATIONS`
    When: The entry names are read in declaration order
    Then: They are already sorted

    Test type: unit
    """
    names = [spec.name for spec in GOLDEN_VISUALIZATIONS]
    assert names == sorted(names), (
        "GOLDEN_VISUALIZATIONS must be alphabetical by name so two environment "
        f"branches insert at different lines, got {names}"
    )


@pytest.mark.skipif(
    not Path("/.dockerenv").exists(),
    reason=(
        "Golden visualization hashes are pinned to the matplotlib / PIL "
        "versions used in the project's Docker CI image. On bare runners "
        "with different font/rendering stacks the byte-identical hash "
        "check is expected to differ, so these consistency checks only "
        "run inside Docker."
    ),
)
class TestVisualizationConsistency:
    """Test suite for visualization determinism and consistency.

    These tests use golden file testing to ensure visualizations remain
    consistent across code changes. On first run, golden files are created.
    On subsequent runs, new outputs are compared against golden files.
    """

    @pytest.mark.parametrize(
        "spec", GOLDEN_VISUALIZATIONS, ids=[spec.name for spec in GOLDEN_VISUALIZATIONS]
    )
    def test_visualization_consistency(self, temp_output_dir, spec):
        """Test that an environment's visualization produces consistent output.

        Purpose: Validates that every registered visualizer is deterministic,
            including the belief panels that average over a particle collection,
            which is where an unordered iteration or a stray draw would leak in

        Given: A deterministic episode built by the environment's own factory
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        history = spec.build_history()
        render = spec.build_renderer()
        output_path = temp_output_dir / f"{spec.name}_test.gif"
        render(history, output_path)

        compare_or_create_golden_file(
            output_path,
            spec.golden_file,
            f"test_visualization_consistency[{spec.name}]",
        )


class TestVisualizationDeterminism:
    """Test that visualizations are deterministic when re-run with same inputs."""

    @pytest.mark.parametrize(
        "spec",
        REPEATED_RENDER_VISUALIZATIONS,
        ids=[spec.name for spec in REPEATED_RENDER_VISUALIZATIONS],
    )
    def test_repeated_visualization_identical(self, temp_output_dir, spec):
        """Test that repeated visualizations are byte-for-byte identical.

        Purpose: Validates absolute determinism of a renderer, which the
            golden-hash check cannot cover outside the project's Docker image.
            The registered environments are the ones whose panels are weighted
            means over a particle collection, which is exactly where an
            unordered iteration or a fresh random draw would show up.

        Given: One episode rendered twice from the same history
        When: Both renders use identical inputs
        Then: Output files have identical SHA256 hashes

        Test type: unit
        """
        history = spec.build_history()
        render = spec.build_renderer()
        first_path = temp_output_dir / f"{spec.name}_first.gif"
        second_path = temp_output_dir / f"{spec.name}_second.gif"
        render(history, first_path)
        render(history, second_path)

        first_hash = compute_file_hash(first_path)
        second_hash = compute_file_hash(second_path)
        assert first_hash == second_hash, (
            f"{spec.name} visualization is not deterministic.\n"
            f"Hash 1: {first_hash}\n"
            f"Hash 2: {second_hash}"
        )
