"""Type stubs for the native C++ maze extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/maze.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by the observation sampler.

    Maze transitions are deterministic; only the cue reading draws from this RNG.
    """
    ...

class MazeModelCpp:
    """Transition, observation, reward and rollout model of one maze environment.

    ``mode`` selects the movement model: ``0`` DiscreteMazePOMDP (one-cell moves,
    cell = round-half-even of the position), ``1`` TMazePOMDP (one-cell moves,
    cell = position truncated toward zero), ``2`` ContinuousMazePOMDP (clipped
    displacement, walked cell by cell along the swept segment). Discrete actions
    are indices into ``ACTIONS``; continuous actions are 2-vectors.
    """

    mode: int

    def __init__(  # pylint: disable=too-many-arguments
        self,
        mode: int,
        walkable: NDArray[np.uint8],
        origin_x: int,
        origin_y: int,
        cue_cell: Tuple[int, int],
        left_goal_cell: Tuple[int, int],
        right_goal_cell: Tuple[int, int],
        cue_accuracy: float,
        log_cue_accuracy: float,
        log_cue_error: float,
        goal_reward: float,
        wrong_goal_penalty: float,
        step_penalty: float,
        max_step_size: float,
        observation_labels: Tuple[str, ...],
    ) -> None: ...
    def is_terminal(self, state: Union[Sequence[float], NDArray[np.floating]]) -> bool: ...
    def sample_next_state(
        self, state: Union[Sequence[float], NDArray[np.floating]], action: Any
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, states: Any, action: Any) -> NDArray[np.float64]: ...
    def sample_observation(
        self, next_state: Union[Sequence[float], NDArray[np.floating]], n_samples: int = 1
    ) -> Union[str, List[str]]: ...
    def observation_log_probability(
        self, next_state: Union[Sequence[float], NDArray[np.floating]], codes: NDArray[np.int32]
    ) -> NDArray[np.float64]: ...
    def batch_log_likelihood(self, next_states: Any, observation: int) -> NDArray[np.float64]: ...
    def reward(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: Any,
        next_state: Optional[Any] = None,
    ) -> float: ...
    def reward_batch(
        self, states: Any, action: Any, next_states: Optional[Any] = None
    ) -> NDArray[np.float64]: ...
    def sample_next_step(
        self, state: Union[Sequence[float], NDArray[np.floating]], action: Any
    ) -> Tuple[NDArray[np.float64], str, float]: ...
    def blocked(self, state: Union[Sequence[float], NDArray[np.floating]], action: Any) -> bool: ...
    def simulate_rollout(
        self,
        initial_state: Union[Sequence[float], NDArray[np.floating]],
        actions: NDArray[Any],
        max_depth: int,
        start_depth: int,
        discount_factor: float,
    ) -> float: ...
