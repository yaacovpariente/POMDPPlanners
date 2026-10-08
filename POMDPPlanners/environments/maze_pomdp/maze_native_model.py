# SPDX-License-Identifier: MIT

"""The environment API of the maze family, answered by the C++ ``MazeModelCpp``.

``DiscreteMazePOMDP``, ``ContinuousMazePOMDP`` and ``TMazePOMDP`` spend most of a
tree search in a handful of calls: the transition, the observation draw, the
reward and the terminal test. :class:`NativeMazeMixin` routes those calls to one
``MazeModelCpp`` per environment, built in ``maze_pomdp/_cpp/maze.cpp``.

Each environment keeps its Python implementation of the same model as private
methods (``_successor``, ``_reward_from_successor``, ``_observation_probs``,
``_py_is_terminal``). Those are the reference the native-equivalence tests
compare against; the public methods here do not call them.

Observation draws come from the native module's RNG, seeded with
``POMDPPlanners.environments.maze_pomdp._native.set_seed``, not from
``np.random``. That matches every other native port, and it means a seeded run
draws different cue readings than the pure-Python version did, from the same
distribution.
"""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from POMDPPlanners.environments.maze_pomdp import _native

# Index of each observation label in OBSERVATIONS, the order the kernel uses.
_OBSERVATION_CODES: Dict[str, int] = {"left_cue": 0, "right_cue": 1, "empty": 2}

# Kernel mode codes, one per movement model; see the top of maze.cpp.
NATIVE_MODE_DISCRETE_MAZE = 0
NATIVE_MODE_T_MAZE = 1
NATIVE_MODE_CONTINUOUS_MAZE = 2


class NativeMazeMixin:
    """Answer the hot environment calls with the environment's ``MazeModelCpp``.

    A subclass provides :meth:`_build_native_model`, which returns the model for
    its geometry, and :meth:`_native_action`, which turns an action into what
    the model takes: an index into ``ACTIONS`` for the discrete variants, the
    action itself for the continuous one.

    The model is a private attribute, so it stays out of ``config_id`` and
    ``to_dict``. It is a pybind11 object and cannot be pickled, so
    ``__getstate__`` drops it and ``__setstate__`` builds it again.
    """

    _model_cpp: Any

    def _build_native_model(self) -> Any:
        raise NotImplementedError

    def _native_action(self, action: Any) -> Any:
        raise NotImplementedError

    def _native_rollout_actions(self, actions: Sequence[Any]) -> np.ndarray:
        """Pack actions drawn for a rollout into the array the model takes."""
        raise NotImplementedError

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_model_cpp", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._model_cpp = self._build_native_model()

    # Dynamics
    def is_terminal(self, state: Any) -> bool:
        """Whether ``state``'s position lies in a goal cell."""
        return bool(self._model_cpp.is_terminal(state))

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample successors. Transitions are deterministic, so all samples agree."""
        successor = self._model_cpp.sample_next_state(state, self._native_action(action))
        if n_samples == 1:
            return successor
        return np.repeat(successor[np.newaxis, :], n_samples, axis=0)

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        """Successors of every row of ``states``, as an ``(N, 4)`` float64 array."""
        return self._model_cpp.batch_sample(states, self._native_action(action))

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log ``T(s' | s, a)``: 0 for the one successor, ``-inf`` everywhere else."""
        successor = self._model_cpp.sample_next_state(state, self._native_action(action))
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        matches = np.all(np.isclose(candidates, successor[np.newaxis, :]), axis=1)
        return np.where(matches, 0.0, -np.inf)

    # Observation model
    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Draw an observation label from ``P(o | s')``. ``action`` does not enter it."""
        del action
        return self._model_cpp.sample_observation(next_state, n_samples)

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` for each observation in ``observations``."""
        del action
        values = [observations] if isinstance(observations, str) else list(observations)
        codes = np.fromiter(
            (_OBSERVATION_CODES.get(value, -1) for value in values),
            dtype=np.int32,
            count=len(values),
        )
        return self._model_cpp.observation_log_probability(next_state, codes)

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` of one observation against every row of ``next_states``."""
        del action
        code = _OBSERVATION_CODES.get(observation, -1) if isinstance(observation, str) else -1
        return self._model_cpp.batch_log_likelihood(next_states, code)

    # Reward
    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        """Immediate reward for ``(state, action)``.

        Transitions are deterministic, so ``next_state`` carries no information the
        model cannot recompute; when supplied it is scored as given.
        """
        return float(self._model_cpp.reward(state, self._native_action(action), next_state))

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Rewards for many states under one action."""
        return self._model_cpp.reward_batch(states, self._native_action(action), next_states)

    def sample_next_step(self, state: Any, action: Any) -> Any:
        """``(next_state, observation, reward)`` from one call into the model."""
        return self._model_cpp.sample_next_step(state, self._native_action(action))

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Discounted return of a random rollout, stepped in C++.

        Draws ``max_depth - depth`` actions from ``action_sampler`` up front, so
        the rollout follows the caller's sampler and not a uniform policy of its
        own. When the rollout reaches a goal early the unused draws are
        discarded, so the sampler is called more often than the step-by-step
        Python rollout calls it; the actions the rollout uses are drawn from the
        same sampler in the same order.

        Args:
            state: The state the rollout starts from.
            action_sampler: Object whose ``sample()`` returns one action.
            max_depth: Depth at which the rollout stops.
            discount_factor: Per-step discount factor.
            depth: Depth already used by the search tree. Defaults to 0.

        Returns:
            ``r_0 + discount_factor * (r_1 + discount_factor * (...))``, summed in
            the order the recursive Python rollout sums it.
        """
        steps_left = max_depth - depth
        if steps_left <= 0 or self.is_terminal(state):
            return 0.0
        drawn = [action_sampler.sample() for _ in range(steps_left)]
        return float(
            self._model_cpp.simulate_rollout(
                state,
                self._native_rollout_actions(drawn),
                max_depth,
                depth,
                float(discount_factor),
            )
        )


def native_log_cue_terms(cue_accuracy: float) -> Dict[str, float]:
    """``log(cue_accuracy)`` and ``log(1 - cue_accuracy)``, computed by numpy.

    Passed to the kernel rather than computed there with ``std::log``, so the
    log-likelihoods equal the Python reference's ``np.log`` values bit for bit.
    """
    with np.errstate(divide="ignore"):
        return {
            "log_cue_accuracy": float(np.log(cue_accuracy)),
            "log_cue_error": float(np.log(1.0 - cue_accuracy)),
        }


def build_native_model(
    mode: int,
    walkable: np.ndarray,
    origin: Sequence[int],
    cue_cell: Sequence[int],
    left_goal_cell: Sequence[int],
    right_goal_cell: Sequence[int],
    cue_accuracy: float,
    goal_reward: float,
    wrong_goal_penalty: float,
    step_penalty: float,
    max_step_size: float,
    observation_labels: Sequence[str],
) -> Any:
    """Build one ``MazeModelCpp``.

    Args:
        mode: One of the ``NATIVE_MODE_*`` codes.
        walkable: ``(nx, ny)`` array, nonzero where cell ``(origin + (i, j))`` is
            walkable. Cells outside it are walls.
        origin: Grid coordinates of ``walkable[0, 0]``.
        cue_cell: The cue cell.
        left_goal_cell: The goal paid for ``GOAL_LEFT``.
        right_goal_cell: The goal paid for ``GOAL_RIGHT``.
        cue_accuracy: Probability the cue names the true side.
        goal_reward: Reward for the correct goal.
        wrong_goal_penalty: Magnitude of the wrong-goal penalty.
        step_penalty: Magnitude of the per-step cost.
        max_step_size: Longest displacement per action; read in the continuous
            mode only.
        observation_labels: The three labels, in ``OBSERVATIONS`` order. The
            model returns these objects, so a sampled label is the same string
            object the module constant holds.

    Returns:
        The model.
    """
    return _native.MazeModelCpp(
        mode=mode,
        walkable=np.ascontiguousarray(walkable, dtype=np.uint8),
        origin_x=int(origin[0]),
        origin_y=int(origin[1]),
        cue_cell=(int(cue_cell[0]), int(cue_cell[1])),
        left_goal_cell=(int(left_goal_cell[0]), int(left_goal_cell[1])),
        right_goal_cell=(int(right_goal_cell[0]), int(right_goal_cell[1])),
        cue_accuracy=float(cue_accuracy),
        goal_reward=float(goal_reward),
        wrong_goal_penalty=float(wrong_goal_penalty),
        step_penalty=float(step_penalty),
        max_step_size=float(max_step_size),
        observation_labels=tuple(observation_labels),
        **native_log_cue_terms(cue_accuracy),
    )


def walkable_grid(cells: Any) -> Any:
    """Pack a collection of ``(x, y)`` cells into ``(walkable, origin)``."""
    xs = [int(cell[0]) for cell in cells]
    ys = [int(cell[1]) for cell in cells]
    origin = (min(xs), min(ys))
    grid = np.zeros((max(xs) - origin[0] + 1, max(ys) - origin[1] + 1), dtype=np.uint8)
    for x, y in zip(xs, ys):
        grid[x - origin[0], y - origin[1]] = 1
    return grid, origin


__all__ = [
    "NATIVE_MODE_CONTINUOUS_MAZE",
    "NATIVE_MODE_DISCRETE_MAZE",
    "NATIVE_MODE_T_MAZE",
    "NativeMazeMixin",
    "build_native_model",
    "walkable_grid",
]
