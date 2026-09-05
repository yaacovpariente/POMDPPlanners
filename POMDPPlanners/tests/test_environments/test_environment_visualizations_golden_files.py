# SPDX-License-Identifier: MIT

"""Tests for environment visualization consistency and correctness.

This module tests that visualizers produce deterministic, reproducible outputs
when given identical episode histories. Uses golden file testing approach where
reference GIF files are stored and compared against new outputs.

Golden File Testing Workflow:
    1. First run: If golden file doesn't exist, generate and save it
    2. Subsequent runs: Compare new output against golden file using hash
    3. Update golden files: Delete old golden file and re-run test to regenerate

Directory Structure:
    POMDPPlanners/tests/test_environments/golden_visualizations/
        ├── rock_sample_visualization.gif
        ├── pacman_visualization.gif
        ├── light_dark_visualization.gif
        ├── push_visualization.gif
        ├── laser_tag_visualization.gif
        ├── continuous_laser_tag_visualization.gif
        ├── continuous_push_visualization.gif
        ├── discrete_maze_visualization.gif
        ├── continuous_maze_visualization.gif
        ├── t_maze_visualization.gif  # compatibility layout
        └── safety_ant_velocity_visualization.gif
"""

import hashlib
import shutil
import warnings
from collections import deque
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (
    RockSampleVisualizer,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_visualizer import PacManVisualizer
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_visualizer import (
    LightDarkPOMDPVisualizer,
)
from POMDPPlanners.environments.maze_pomdp import (
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeVisualizer,
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
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualizer import (
    SafeAntVelocityVisualizer,
)
from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import (
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    TMazePOMDP,
    create_t_maze_state,
)
from POMDPPlanners.environments.t_maze_pomdp.maze_pomdp import (
    ACTION_OFFSETS,
    create_maze_state,
)
from POMDPPlanners.environments.t_maze_pomdp.t_maze_visualizer import TMazeVisualizer
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_maze_pinned_kwargs,
    continuous_laser_tag_pinned_kwargs,
    continuous_light_dark_pinned_kwargs,
    continuous_push_pinned_kwargs,
    discrete_maze_pinned_kwargs,
    laser_tag_pinned_kwargs,
    pacman_pinned_kwargs,
    push_pinned_kwargs,
    rock_sample_pinned_kwargs,
    safety_ant_velocity_pinned_kwargs,
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

    def test_rock_sample_visualization_consistency(self, temp_output_dir):
        """Test RockSample visualization produces consistent output.

        Purpose: Validates that RockSample visualizations are deterministic

        Given: A deterministic RockSample episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_rock_sample_episode(seed=42)

        # Create environment and visualizer
        env = RockSamplePOMDP(
            discount_factor=0.95,
            **rock_sample_pinned_kwargs(
                map_size=(5, 5),
                rock_positions=[(1, 1), (2, 3), (4, 2)],
                dangerous_areas=[(2, 2)],
                dangerous_area_radius=1.0,
            ),
        )
        visualizer = RockSampleVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "rock_sample_test.gif"
        visualizer.create_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "rock_sample_visualization.gif",
            "test_rock_sample_visualization_consistency",
        )

    def test_pacman_visualization_consistency(self, temp_output_dir):
        """Test PacMan visualization produces consistent output.

        Purpose: Validates that PacMan visualizations are deterministic

        Given: A deterministic PacMan episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_pacman_episode(seed=42)

        # Create environment and visualizer
        env = PacManPOMDP(
            discount_factor=0.95,
            **pacman_pinned_kwargs(
                maze_size=(7, 7),
                num_ghosts=2,
                initial_ghost_positions=None,
                ghost_strategies=None,
            ),
        )
        visualizer = PacManVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "pacman_test.gif"
        visualizer.cache_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "pacman_visualization.gif",
            "test_pacman_visualization_consistency",
        )

    def test_light_dark_visualization_consistency(self, temp_output_dir):
        """Test LightDark visualization produces consistent output.

        Purpose: Validates that LightDark visualizations are deterministic

        Given: A deterministic LightDark episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_light_dark_episode(seed=42)

        # Create environment and visualizer
        env = ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            **continuous_light_dark_pinned_kwargs(),
        )
        visualizer = LightDarkPOMDPVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "light_dark_test.gif"
        visualizer.cache_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "light_dark_visualization.gif",
            "test_light_dark_visualization_consistency",
        )

    def test_push_visualization_consistency(self, temp_output_dir):
        """Test Push visualization produces consistent output.

        Purpose: Validates that Push visualizations are deterministic

        Given: A deterministic Push episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_push_episode(seed=42)

        # Create environment and visualizer
        env = PushPOMDP(
            discount_factor=0.95,
            **push_pinned_kwargs(
                grid_size=8,
                transition_error_prob=0.0,  # Explicitly set for deterministic behavior
            ),
        )
        visualizer = PushPOMDPVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "push_test.gif"
        visualizer.create_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "push_visualization.gif",
            "test_push_visualization_consistency",
        )

    def test_laser_tag_visualization_consistency(self, temp_output_dir):
        """Test LaserTag visualization produces consistent output.

        Purpose: Validates that LaserTag visualizations are deterministic

        Given: A deterministic LaserTag episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_laser_tag_episode(seed=42)

        # Create environment and visualizer
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

        # Generate visualization
        output_path = temp_output_dir / "laser_tag_test.gif"
        visualizer.create_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "laser_tag_visualization.gif",
            "test_laser_tag_visualization_consistency",
        )

    def test_continuous_laser_tag_visualization_consistency(self, temp_output_dir):
        """Test Continuous LaserTag visualization produces consistent output.

        Purpose: Validates that Continuous LaserTag visualizations are deterministic

        Given: A deterministic Continuous LaserTag episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_continuous_laser_tag_episode(seed=42)

        # Create environment and visualizer
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

        # Generate visualization
        output_path = temp_output_dir / "continuous_laser_tag_test.gif"
        visualizer.create_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "continuous_laser_tag_visualization.gif",
            "test_continuous_laser_tag_visualization_consistency",
        )

    def test_continuous_push_visualization_consistency(self, temp_output_dir):
        """Test Continuous Push visualization produces consistent output.

        Purpose: Validates that Continuous Push visualizations are deterministic

        Given: A deterministic Continuous Push episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_continuous_push_episode(seed=42)

        # Create environment and visualizer
        env = ContinuousPushPOMDP(
            discount_factor=0.99,
            **continuous_push_pinned_kwargs(
                grid_size=10,
                obstacles=[(3.0, 3.0, 0.5), (6.0, 6.0, 0.5)],
                state_transition_cov_matrix=np.eye(2) * 0.01,
            ),
        )
        visualizer = ContinuousPushPOMDPVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "continuous_push_test.gif"
        visualizer.create_visualization(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "continuous_push_visualization.gif",
            "test_continuous_push_visualization_consistency",
        )

    def test_safety_ant_velocity_visualization_consistency(self, temp_output_dir):
        """Test SafeAntVelocity visualization produces consistent output.

        Purpose: Validates that SafeAntVelocity visualizations are deterministic

        Given: A deterministic SafeAntVelocity episode with fixed seed
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        # Create deterministic episode
        history = create_deterministic_safety_ant_velocity_episode(seed=42)

        # Create environment and visualizer
        env = SafeAntVelocityPOMDP(
            discount_factor=0.95,
            **safety_ant_velocity_pinned_kwargs(),
        )
        visualizer = SafeAntVelocityVisualizer(env)

        # Generate visualization
        output_path = temp_output_dir / "safety_ant_velocity_test.gif"
        visualizer.create_animation(history, output_path)

        # Compare against golden file
        compare_or_create_golden_file(
            output_path,
            "safety_ant_velocity_visualization.gif",
            "test_safety_ant_velocity_visualization_consistency",
        )


    def test_t_maze_visualization_consistency(self, temp_output_dir):
        """Test T-Maze visualization produces consistent output.

        Purpose: Validates that T-Maze visualizations are deterministic, including
            the belief overlay, which is the part a human reads to check the
            observation model did anything

        Given: A deterministic T-Maze episode with a hand-built spread belief
        When: Visualization is created from the episode
        Then: Output matches golden file hash (or creates golden if missing)

        Test type: integration
        """
        history = create_deterministic_t_maze_episode()

        env = TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs())
        visualizer = TMazeVisualizer(env)

        output_path = temp_output_dir / "t_maze_test.gif"
        visualizer.create_visualization(history, output_path)

        compare_or_create_golden_file(
            output_path,
            "t_maze_visualization.gif",
            "test_t_maze_visualization_consistency",
        )

    @pytest.mark.parametrize(
        "env,golden_name",
        [
            (
                DiscreteMazePOMDP(discount_factor=0.95, **discrete_maze_pinned_kwargs()),
                "discrete_maze_visualization.gif",
            ),
            (
                ContinuousMazePOMDP(
                    discount_factor=0.95, **continuous_maze_pinned_kwargs()
                ),
                "continuous_maze_visualization.gif",
            ),
        ],
        ids=["discrete_maze", "continuous_maze"],
    )
    def test_maze_visualization_consistency(self, temp_output_dir, env, golden_name):
        """Both public Maze variants render deterministic generated geometry."""
        history = create_deterministic_maze_episode(env)
        output_path = temp_output_dir / golden_name
        MazeVisualizer(env).create_visualization(history, output_path)
        compare_or_create_golden_file(
            output_path,
            golden_name,
            f"test_{golden_name.removesuffix('.gif')}_consistency",
        )


class TestVisualizationDeterminism:
    """Test that visualizations are deterministic when re-run with same inputs."""

    def test_rock_sample_repeated_visualization_identical(self, temp_output_dir):
        """Test that repeated RockSample visualizations are byte-for-byte identical.

        Purpose: Validates absolute determinism of visualization generation

        Given: A deterministic episode visualized twice
        When: Both visualizations use identical inputs
        Then: Output files have identical SHA256 hashes

        Test type: unit
        """
        # Create deterministic episode
        history = create_deterministic_rock_sample_episode(seed=42)

        # Create environment and visualizer
        env = RockSamplePOMDP(
            discount_factor=0.95,
            **rock_sample_pinned_kwargs(
                map_size=(5, 5),
                rock_positions=[(1, 1), (2, 3), (4, 2)],
                dangerous_areas=[(2, 2)],
                dangerous_area_radius=1.0,
            ),
        )
        visualizer = RockSampleVisualizer(env)

        # Generate two visualizations
        output_path_1 = temp_output_dir / "viz_1.gif"
        output_path_2 = temp_output_dir / "viz_2.gif"

        visualizer.create_visualization(history, output_path_1)
        visualizer.create_visualization(history, output_path_2)

        # Compare hashes
        hash_1 = compute_file_hash(output_path_1)
        hash_2 = compute_file_hash(output_path_2)

        assert hash_1 == hash_2, (
            f"Repeated visualizations should be identical.\n"
            f"Hash 1: {hash_1}\n"
            f"Hash 2: {hash_2}"
        )
