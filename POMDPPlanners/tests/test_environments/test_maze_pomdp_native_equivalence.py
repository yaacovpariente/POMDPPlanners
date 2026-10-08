# SPDX-License-Identifier: MIT

"""Native (C++) equivalence tests for the maze family.

``DiscreteMazePOMDP``, ``ContinuousMazePOMDP`` and ``TMazePOMDP`` answer their
public environment API from one ``MazeModelCpp`` each. Every environment keeps
its Python implementation of the model as private methods (``_successor``,
``_reward_from_successor``, ``_observation_probs``, ``_py_is_terminal``,
``_execute``); these tests compare the native public API against them.

* Transitions, terminal tests, rewards and wall collisions are deterministic,
  so they are compared with exact equality, byte for byte, on many random
  states and actions. The continuous cases include points on cell boundaries,
  moves through wall corners, and displacements longer than ``max_step_size``,
  where a one-ulp difference would flip a cell test.
* Observation log-probabilities are compared exactly against ``np.log`` of the
  reference probabilities.
* Observation draws come from the native RNG, so they are checked against the
  cue accuracy with 200k draws, and for reproducibility under ``set_seed``.
* The native rollout is compared exactly against ``python_random_rollout``
  replaying the same actions.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access

import pickle
from typing import Any, Callable, Iterator, List, Sequence

import numpy as np
import pytest

from POMDPPlanners.environments.maze_pomdp import _native
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    ACTIONS,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATIONS,
    BaseMazePOMDP,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
)
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import TMazePOMDP
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_maze_pinned_kwargs,
    discrete_maze_pinned_kwargs,
    t_maze_pinned_kwargs,
)

_PHASES = (CUE_UNSEEN, CUE_EMITTING, CUE_CONSUMED)
_SIDES = (GOAL_LEFT, GOAL_RIGHT)


def _discrete_envs() -> List[DiscreteMazePOMDP]:
    return [
        DiscreteMazePOMDP(discount_factor=0.95, **discrete_maze_pinned_kwargs()),
        DiscreteMazePOMDP(
            discount_factor=0.95,
            **discrete_maze_pinned_kwargs(maze_width=11, maze_height=13, maze_seed=4),
        ),
        DiscreteMazePOMDP(
            discount_factor=0.95,
            **discrete_maze_pinned_kwargs(
                maze_width=15, maze_height=17, maze_seed=9, loop_fraction=0.0, cue_accuracy=1.0
            ),
        ),
    ]


def _t_maze_envs() -> List[TMazePOMDP]:
    return [
        TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs()),
        TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs(stem_length=7, arm_length=3)),
        TMazePOMDP(
            discount_factor=0.95,
            **t_maze_pinned_kwargs(stem_length=2, arm_length=1, cue_accuracy=0.5),
        ),
    ]


def _continuous_envs() -> List[ContinuousMazePOMDP]:
    return [
        ContinuousMazePOMDP(
            discount_factor=0.95,
            **continuous_maze_pinned_kwargs(
                maze_width=11, maze_height=13, maze_seed=7, max_step_size=step
            ),
        )
        for step in (0.75, 1.0, 2.0, 3.5)
    ]


def _all_envs() -> List[Any]:
    return [*_discrete_envs(), *_t_maze_envs(), *_continuous_envs()]


def _env_id(env: Any) -> str:
    return f"{type(env).__name__}-{env.config_id[:8]}"


def _state(x: float, y: float, side: float, phase: float) -> np.ndarray:
    return np.array([x, y, side, phase], dtype=np.float64)


def _cell_states(cells: Any) -> Iterator[np.ndarray]:
    """Every walkable cell under every goal side and cue phase."""
    for x, y in sorted(cells):
        for side in _SIDES:
            for phase in _PHASES:
                yield _state(float(x), float(y), side, phase)


def _continuous_states(env: ContinuousMazePOMDP, rng: np.random.Generator) -> List[np.ndarray]:
    """Real positions inside walkable cells, on their edges, and on their corners."""
    states = []
    for x, y in sorted(env.walkable_cells):
        offsets = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (-0.5, -0.5)]
        offsets += [tuple(rng.uniform(-0.5, 0.5, size=2)) for _ in range(3)]
        for dx, dy in offsets:
            states.append(_state(x + dx, y + dy, rng.choice(_SIDES), rng.choice(_PHASES)))
    return states


def _continuous_actions(env: ContinuousMazePOMDP, rng: np.random.Generator) -> List[np.ndarray]:
    """Axis moves, diagonals through corner points, zero, and over-long vectors."""
    step = env.max_step_size
    actions = [
        np.array([0.0, 0.0]),
        np.array([0.0, step]),
        np.array([step, 0.0]),
        np.array([-step, 0.0]),
        np.array([0.0, -0.5]),
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.array([1.0, 1.0]),
        np.array([3.0, 4.0]),
        np.array([-7.0, 2.0]),
    ]
    angles = rng.uniform(0.0, 2.0 * np.pi, size=12)
    radii = rng.uniform(0.0, 2.0 * step, size=12)
    actions += [np.array([r * np.cos(a), r * np.sin(a)]) for r, a in zip(radii, angles)]
    return actions


def _cases(env: Any, rng: np.random.Generator) -> Iterator[Any]:
    """(state, action) pairs covering the environment's movement model."""
    if isinstance(env, ContinuousMazePOMDP):
        states = _continuous_states(env, rng)
        actions = _continuous_actions(env, rng)
        for index, state in enumerate(states):
            # Every state against a rotating subset keeps the run short while
            # every action meets states on edges, corners and interiors.
            for action in actions[index % 3 :: 3]:
                yield state, action
        return
    cells = env.valid_cells if isinstance(env, TMazePOMDP) else env.walkable_cells
    for state in _cell_states(cells):
        for action in ACTIONS:
            yield state, action


def _assert_bitwise_equal(native: np.ndarray, reference: np.ndarray, context: str) -> None:
    native = np.asarray(native, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    assert native.shape == reference.shape, f"{context}: shape {native.shape} != {reference.shape}"
    assert native.tobytes() == reference.tobytes(), f"{context}: {native!r} != {reference!r}"


@pytest.fixture(name="rng")
def _rng() -> np.random.Generator:
    return np.random.default_rng(20261007)


class TestDeterministicParts:
    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    def test_successor_terminal_and_reward_match_python_bitwise(self, env, rng):
        n_cases = 0
        for state, action in _cases(env, rng):
            context = f"state={state.tolist()} action={action!r}"
            expected = env._successor(state, action)
            _assert_bitwise_equal(env.sample_next_state(state, action), expected, context)
            assert env.is_terminal(state) == env._py_is_terminal(state), context
            assert env.is_terminal(expected) == env._py_is_terminal(expected), context

            expected_reward = env._reward_from_successor(state, expected)
            assert env.reward(state, action) == expected_reward, context
            assert env.reward(state, action, next_state=expected) == expected_reward, context

            next_state, _, reward = env.sample_next_step(state, action)
            _assert_bitwise_equal(next_state, expected, context)
            assert reward == expected_reward, context

            if isinstance(env, BaseMazePOMDP) and not env._py_is_terminal(state):
                info = env.step_info(state, action, expected)
                assert info["wall_collision"] == float(env._execute(state, action).blocked)
            n_cases += 1
        assert n_cases > 100

    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    def test_batch_paths_match_single_paths(self, env, rng):
        cases = list(_cases(env, rng))
        by_action: dict = {}
        for state, action in cases:
            key = env.hash_action(action)
            by_action.setdefault(key, (action, []))[1].append(state)
        for action, states in by_action.values():
            states_arr = np.stack(states)
            expected = np.stack([env._successor(s, action) for s in states])
            batch = env.sample_next_state_batch(states_arr, action)
            assert batch.dtype == np.float64
            _assert_bitwise_equal(batch, expected, f"batch action={action!r}")

            expected_rewards = np.array(
                [env._reward_from_successor(s, n) for s, n in zip(states, expected)]
            )
            _assert_bitwise_equal(env.reward_batch(states_arr, action), expected_rewards, "rb")
            _assert_bitwise_equal(
                env.reward_batch(states_arr, action, next_states=expected), expected_rewards, "rb2"
            )

    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    def test_transition_log_probability_is_an_indicator(self, env, rng):
        state, action = next(_cases(env, rng))
        successor = env._successor(state, action)
        perturbed = successor.copy()
        perturbed[3] = 7.0
        out = env.transition_log_probability(state, action, np.stack([successor, perturbed]))
        assert out.tolist() == [0.0, -np.inf]

    def test_unknown_discrete_action_raises_only_when_it_would_move(self):
        for env in (_discrete_envs()[0], _t_maze_envs()[0]):
            start = env.initial_state_dist().sample()[0]
            with pytest.raises(KeyError):
                env.sample_next_state(start, "jump")
            goal = env.left_goal_cell if isinstance(env, BaseMazePOMDP) else env.left_endpoint
            terminal = _state(goal[0], goal[1], GOAL_LEFT, CUE_CONSUMED)
            _assert_bitwise_equal(env.sample_next_state(terminal, "jump"), terminal, "absorbing")

    def test_discrete_maze_rejects_a_state_off_the_walkable_cells(self):
        env = _discrete_envs()[0]
        with pytest.raises(KeyError):
            env.sample_next_state(_state(0.0, 0.0, GOAL_LEFT, CUE_UNSEEN), "up")

    def test_continuous_rejects_malformed_actions_like_clip_action(self):
        env = _continuous_envs()[0]
        start = env.initial_state_dist().sample()[0]
        for invalid in (np.array([1.0]), np.array([1.0, 2.0, 3.0]), np.array([np.nan, 0.0])):
            with pytest.raises(ValueError):
                env.clip_action(invalid)
            with pytest.raises(ValueError):
                env.sample_next_state(start, invalid)


class TestObservationModel:
    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    def test_log_probabilities_match_python_exactly(self, env):
        labels = [*OBSERVATIONS, "wall"]
        states = [_state(0.0, 1.0, side, phase) for side in _SIDES for phase in _PHASES]
        for state in states:
            probs = env._observation_probs(state)
            with np.errstate(divide="ignore"):
                expected = [float(np.log(p)) if p > 0.0 else -np.inf for p in probs] + [-np.inf]
            native = env.observation_log_probability(state, None, labels)
            assert native.tolist() == expected
            assert env.observation_log_probability(state, None, "empty").tolist() == [expected[2]]
        states_arr = np.stack(states)
        for index, label in enumerate(labels):
            per_state = env.observation_log_probability_per_state(states_arr, None, label)
            single = [env.observation_log_probability(s, None, [label])[0] for s in states]
            assert per_state.tolist() == single, label
            assert index < 3 or np.all(per_state == -np.inf)

    @pytest.mark.parametrize("env", [_discrete_envs()[0], _t_maze_envs()[0]], ids=_env_id)
    def test_cue_frequency_matches_accuracy(self, env):
        _native.set_seed(123)
        n_draws = 200_000
        emitting_left = _state(0.0, 1.0, GOAL_LEFT, CUE_EMITTING)
        draws = env.sample_observation(emitting_left, None, n_samples=n_draws)
        assert set(draws) <= {"left_cue", "right_cue"}
        assert abs(draws.count("left_cue") / n_draws - env.cue_accuracy) < 5e-3

        emitting_right = _state(0.0, 1.0, GOAL_RIGHT, CUE_EMITTING)
        draws = env.sample_observation(emitting_right, None, n_samples=n_draws)
        assert abs(draws.count("right_cue") / n_draws - env.cue_accuracy) < 5e-3

        for phase in (CUE_UNSEEN, CUE_CONSUMED):
            quiet = _state(0.0, 1.0, GOAL_LEFT, phase)
            assert set(env.sample_observation(quiet, None, n_samples=1000)) == {"empty"}

    def test_observation_draws_reproduce_under_set_seed(self):
        env = _discrete_envs()[0]
        emitting = _state(0.0, 1.0, GOAL_LEFT, CUE_EMITTING)
        _native.set_seed(7)
        first = [env.sample_observation(emitting, None) for _ in range(200)]
        _native.set_seed(7)
        second = [env.sample_observation(emitting, None) for _ in range(200)]
        assert first == second
        assert all(label in OBSERVATIONS for label in first)


class _ReplaySampler(ActionSampler):
    """Hands back a fixed action list, one per ``sample()`` call."""

    def __init__(self, actions: Sequence[Any]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def sample(self, belief_node: Any = None) -> Any:
        del belief_node
        action = self.actions[self.calls]
        self.calls += 1
        return action


class TestRollout:
    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    @pytest.mark.parametrize("depth", [0, 3])
    def test_native_rollout_equals_python_rollout_on_the_same_actions(self, env, depth, rng):
        max_depth = 30
        draw: Callable[[], Any]
        if isinstance(env, ContinuousMazePOMDP):
            step = env.max_step_size

            def draw() -> Any:
                return rng.uniform(-1.5 * step, 1.5 * step, size=2)

        else:

            def draw() -> Any:
                return ACTIONS[int(rng.integers(len(ACTIONS)))]

        start = env.initial_state_dist().sample()[0]
        for _ in range(20):
            actions = [draw() for _ in range(max_depth)]
            native = env.simulate_random_rollout(
                state=start,
                action_sampler=_ReplaySampler(actions),
                max_depth=max_depth,
                discount_factor=0.95,
                depth=depth,
            )
            expected = python_random_rollout(
                state=start,
                depth=depth,
                action_sampler=_ReplaySampler(actions),
                environment=env,
                discount_factor=0.95,
                max_depth=max_depth,
            )
            assert native == expected

    def test_rollout_from_a_terminal_state_is_zero(self):
        env = _discrete_envs()[0]
        goal = env.left_goal_cell
        terminal = _state(goal[0], goal[1], GOAL_LEFT, CUE_CONSUMED)
        sampler = _ReplaySampler(["up"] * 10)
        assert env.simulate_random_rollout(terminal, sampler, 10, 0.95) == 0.0
        assert sampler.calls == 0


class TestModelLifecycle:
    @pytest.mark.parametrize("env", _all_envs(), ids=_env_id)
    def test_pickle_rebuilds_the_native_model_and_keeps_identity(self, env, rng):
        config_id = env.config_id
        state, action = next(_cases(env, rng))
        before = env.sample_next_state(state, action)
        restored = pickle.loads(pickle.dumps(env))
        assert restored.config_id == config_id == env.config_id
        assert restored == env
        _assert_bitwise_equal(restored.sample_next_state(state, action), before, "pickled")
        assert "_model_cpp" not in env.__getstate__()
