# SPDX-License-Identifier: MIT

"""The Snake renderer.

The golden-hash test pins the rendered bytes, but only inside the project's
Docker image. These tests cover what it cannot: that the renderer refuses
malformed input, that it reads the belief rather than the true state, and that
it renders an episode that ended in each of the four ways.
"""

import random

import numpy as np
import pytest
from PIL import Image

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.snake_pomdp.snake_belief import SnakeBelief
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    SnakeAction,
    SnakePOMDP,
    create_snake_state,
)
from POMDPPlanners.environments.snake_pomdp.snake_visualizer import SnakeVisualizer


def build_env(**overrides):
    """Return a small Snake environment, with overrides applied."""
    kwargs = {"grid_size": 7, "target_length": 6, "starvation_limit": 8, "discount_factor": 0.98}
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs)


def belief_matching(env, state, n_particles=16):
    """Uniform belief over the food, sharing the body the episode starts from.

    ``SnakeBelief.from_environment`` builds the belief for the *initial* state,
    whose body sits at the centre of the grid. Several tests here start from a
    hand-placed body instead, and pairing that with the initial belief makes a
    belief that was never tracking the episode: the terminal reading then says
    the episode ended while the belief holds no food cell that could have ended
    it, which the update rightly refuses.
    """
    body = env.body(state)
    free = env.free_cells(body)
    probabilities = np.zeros(env.num_cells, dtype=np.float64)
    probabilities[free] = 1.0 / free.size
    cells = np.random.choice(env.num_cells, size=n_particles, p=probabilities)
    particles = np.asarray(
        [
            create_snake_state(
                body=body,
                food=(int(cell) // env.grid_size, int(cell) % env.grid_size),
                steps_since_food=env.steps_since_food(state),
                target_length=env.target_length,
            )
            for cell in cells
        ],
        dtype=np.float64,
    )
    return SnakeBelief(
        particles=particles,
        log_weights=np.full(n_particles, -float(np.log(n_particles))),
        food_probabilities=probabilities,
    )


def episode(env, actions, state, belief=None):
    """Record a fixed action sequence the way the episode runner does."""
    belief = belief_matching(env, state) if belief is None else belief
    steps = []
    for action in actions:
        if env.is_terminal(state):
            break
        next_state, observation, reward = env.sample_next_step(state, int(action))
        steps.append(
            StepData(
                state=state,
                action=int(action),
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=belief,
                info=env.step_info(state, int(action), next_state),
            )
        )
        belief = belief.update(int(action), observation, env)
        state = next_state
    if env.is_terminal(state):
        steps.append(
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
    return steps


def test_one_frame_is_rendered_per_recorded_step(tmp_path):
    """The GIF has exactly as many frames as the history has steps.

    Test type: integration
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)] * 3, env.initial_state_dist().sample()[0])
    output = tmp_path / "snake.gif"
    SnakeVisualizer(env).create_visualization(history, output)
    with Image.open(output) as handle:
        assert getattr(handle, "n_frames", 1) == len(history)


@pytest.mark.parametrize(
    "body, food, counter, kwargs, actions",
    [
        # Wall hit.
        ([(0, 3), (1, 3), (2, 3)], (5, 5), 0, {}, [SnakeAction.GO_STRAIGHT]),
        # Self hit.
        (
            [(2, 3), (2, 2), (3, 2), (3, 3), (4, 3)],
            (0, 0),
            0,
            {},
            [SnakeAction.TURN_RIGHT],
        ),
        # Starvation.
        (
            [(3, 3), (3, 2), (3, 1)],
            (0, 6),
            3,
            {"starvation_limit": 4},
            [SnakeAction.GO_STRAIGHT],
        ),
        # Win.
        ([(3, 3), (3, 2), (3, 1)], (3, 4), 0, {"target_length": 4}, [SnakeAction.GO_STRAIGHT]),
    ],
)
def test_every_ending_renders(tmp_path, body, food, counter, kwargs, actions):
    """Each of the four terminal states has its own marker and caption.

    Purpose: The death cross, the starvation hourglass and the win caption are
        drawn from a state the ordinary frames never carry -- a head off the
        grid, or a body with no food. Rendering only running frames would leave
        all four paths untested.

    Test type: integration
    """
    env = build_env(**kwargs)
    random.seed(0)
    np.random.seed(0)
    state = create_snake_state(
        body=body, food=food, steps_since_food=counter, target_length=env.target_length
    )
    history = episode(env, actions, state)
    output = tmp_path / "snake.gif"
    SnakeVisualizer(env).create_visualization(history, output)
    assert output.stat().st_size > 0
    with Image.open(output) as handle:
        # ``getattr`` rather than a plain attribute read: Pillow only declares
        # ``n_frames`` on the multi-frame subclasses, so the CI image's older
        # stubs reject the direct access on the ``ImageFile`` the opener is
        # typed as. Seeking to the last frame and loading it is what actually
        # decodes every frame, which is how a half-written GIF is caught.
        handle.seek(getattr(handle, "n_frames", 1) - 1)
        handle.load()


def test_the_belief_layer_comes_from_the_belief_and_not_the_true_state(tmp_path):
    """Two episodes with the same states and different beliefs render differently.

    Purpose: The belief layer is the whole reason this visualization exists. A
        renderer that drew the true food position as the "belief" would look
        entirely plausible and would make every frame useless for review.

    Test type: integration
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)] * 2, state)

    visualizer = SnakeVisualizer(env)
    first = tmp_path / "first.gif"
    second = tmp_path / "second.gif"
    visualizer.create_visualization(history, first)

    body = env.body(state)
    skewed = np.zeros(env.num_cells)
    free = env.free_cells(body)
    skewed[free[0]] = 1.0
    replaced = [
        StepData(
            state=step.state,
            action=step.action,
            next_state=step.next_state,
            observation=step.observation,
            reward=step.reward,
            belief=SnakeBelief(
                particles=step.belief.particles,
                log_weights=step.belief.log_weights,
                food_probabilities=skewed,
            ),
            info=step.info,
        )
        for step in history
    ]
    visualizer.create_visualization(replaced, second)
    assert first.read_bytes() != second.read_bytes()


def test_a_generic_particle_belief_still_renders(tmp_path):
    """The renderer falls back to the particle histogram, not to nothing.

    Test type: integration
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    particles = np.asarray(env.initial_state_dist().sample(12), dtype=np.float64)
    generic = WeightedParticleBelief(
        particles=particles, log_weights=np.full(12, -float(np.log(12)))
    )
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)] * 2, state, belief=generic)
    output = tmp_path / "snake.gif"
    SnakeVisualizer(env).create_visualization(history, output)
    grid = SnakeVisualizer(env)._belief_grid(generic)  # pylint: disable=protected-access
    assert grid.sum() == pytest.approx(1.0)


def test_the_environment_owns_the_output_filename(tmp_path):
    """``cache_visualization`` takes a directory and an index, not a path.

    Test type: integration
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)] * 2, env.initial_state_dist().sample()[0])
    env.cache_visualization(history, tmp_path, 3)
    assert (tmp_path / "snake_board_3.gif").is_file()


@pytest.mark.parametrize(
    "history, path, error",
    [
        ("not a list", "snake.gif", TypeError),
        ([], "snake.gif", ValueError),
        (["not a StepData"], "snake.gif", TypeError),
    ],
)
def test_malformed_input_is_refused(tmp_path, history, path, error):
    """Bad input fails loudly rather than writing a broken GIF.

    Test type: unit
    """
    visualizer = SnakeVisualizer(build_env())
    with pytest.raises(error):
        visualizer.create_visualization(history, tmp_path / path)


def test_a_non_gif_destination_is_refused(tmp_path):
    """The renderer writes GIFs and says so.

    Test type: unit
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)], env.initial_state_dist().sample()[0])
    with pytest.raises(ValueError, match=".gif"):
        SnakeVisualizer(env).create_visualization(history, tmp_path / "snake.png")
    with pytest.raises(TypeError):
        SnakeVisualizer(env).create_visualization(history, str(tmp_path / "snake.gif"))  # type: ignore[arg-type]


def test_rendering_the_same_history_twice_is_byte_identical(tmp_path):
    """Nothing in the renderer depends on iteration order or a fresh draw.

    Test type: unit
    """
    env = build_env()
    random.seed(0)
    np.random.seed(0)
    history = episode(env, [int(SnakeAction.GO_STRAIGHT)] * 3, env.initial_state_dist().sample()[0])
    visualizer = SnakeVisualizer(env)
    first = tmp_path / "first.gif"
    second = tmp_path / "second.gif"
    visualizer.create_visualization(history, first)
    np.random.seed(99)
    random.seed(99)
    visualizer.create_visualization(history, second)
    assert first.read_bytes() == second.read_bytes()
