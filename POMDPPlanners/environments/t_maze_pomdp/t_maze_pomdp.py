# SPDX-License-Identifier: MIT

"""T-Maze POMDP: a memory task with a single-use cue and a delayed reward.

The agent starts at the bottom of a symmetric T-shaped corridor. One move up sits
the **cue cell**, which reveals — once, noisily — which arm the goal is in. The
junction is several moves further up, so the agent must *carry* that reading
through a stretch of corridor where every observation is identical before it can
act on it. That gap is the whole point of the environment: a planner that does not
track a belief has nothing left to turn on when it reaches the junction.

Task completion means **entering the correct arm endpoint**. Entering the wrong one
also ends the episode, and the two are reported separately (``task_completion_rate``
against ``wrong_endpoint_rate``) because a planner that guesses wrong and one that
never reaches the junction fail for opposite reasons.

State
    ``[x, y, goal_side, cue_phase]`` as a float64 array.

    * ``x`` — column, 0 on the stem, negative into the left arm, positive into the
      right arm.
    * ``y`` — row, 0 at the start cell, ``stem_length`` at the junction.
    * ``goal_side`` — ``GOAL_LEFT`` (0.0) or ``GOAL_RIGHT`` (1.0). Drawn uniformly
      once per episode and never changed by any transition.
    * ``cue_phase`` — ``CUE_UNSEEN`` (0.0) -> ``CUE_EMITTING`` (1.0) -> ``CUE_CONSUMED``
      (2.0). Carrying the delivery phase *in the state* rather than as a flag on the
      environment object is what keeps the problem Markov and keeps the environment
      usable as a planner's generative model: a planner resamples transitions from
      arbitrary states out of order, so an episode flag living on ``self`` would be
      read and written by the search as well as by the world.

Actions
    ``"up"``, ``"down"``, ``"left"``, ``"right"``. Movement is deterministic. A move
    into a wall leaves the position unchanged with probability 1 — the cue phase
    still advances, so bumping into a wall at the cue cell consumes the cue exactly
    as any other action does.

Observations
    ``"left_cue"``, ``"right_cue"``, ``"empty"``. There is deliberately **no** wall
    observation: a "wall" reading would be ambiguous here (three of the four actions
    bump into a wall almost everywhere on the stem) and would leak position
    information the task is not about. The observation is a function of the *next*
    state only:

    * ``cue_phase == CUE_EMITTING`` — ``"left_cue"`` / ``"right_cue"``, matching the
      true goal side with probability ``cue_accuracy`` and contradicting it with
      ``1 - cue_accuracy``.
    * anything else — ``"empty"`` with probability 1.

    The cue is therefore **single-use**: it emits on the step that first enters the
    cue cell and is consumed by the next action, whatever that action is. A revisit
    never reveals a second reading. Under a uniform prior over the goal side, one
    ``"left_cue"`` at the default accuracy moves the belief to 0.9 / 0.1, and every
    subsequent ``"empty"`` in the corridor leaves it exactly there.

Reward
    ``+goal_reward`` for entering the correct endpoint, ``-wrong_goal_penalty`` for
    entering the wrong one, ``-step_penalty`` for every other action including a
    wall collision, and ``0`` for any action taken from an (absorbing) terminal
    state. The terminal payout *replaces* the step penalty rather than stacking with
    it, so the best achievable episode return is exactly the goal reward discounted
    by the number of steps taken to get there.

Classes:
    TMazePOMDP: The environment.
    TMazeStepChannel: Per-step measurement channel names.
    TMazeMetric: Episode-level metric names.

Example:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> env = TMazePOMDP(discount_factor=0.95)
    >>> state = env.initial_state_dist().sample()[0]
    >>> env.is_terminal(state)
    False
    >>> next_state, observation, reward = env.sample_next_step(state, "up")
    >>> observation in ("left_cue", "right_cue")
    True
    >>> float(reward)
    -1.0
"""

from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction, StepInfoMetric

# State slot indices.
STATE_X = 0
STATE_Y = 1
STATE_GOAL = 2
STATE_CUE_PHASE = 3
STATE_WIDTH = 4

# Goal-side encodings.
GOAL_LEFT = 0.0
GOAL_RIGHT = 1.0

# Cue delivery phases. The cue is emitted on entering the cue cell and consumed by
# the following action, so exactly one observation can ever carry it.
CUE_UNSEEN = 0.0
CUE_EMITTING = 1.0
CUE_CONSUMED = 2.0

# Observation alphabet. No wall observation on purpose; see the module docstring.
OBSERVATION_LEFT_CUE = "left_cue"
OBSERVATION_RIGHT_CUE = "right_cue"
OBSERVATION_EMPTY = "empty"
OBSERVATIONS: Tuple[str, ...] = (
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    OBSERVATION_EMPTY,
)

# Actions and their (dx, dy) offsets. ``y`` grows upward from the start cell.
ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_OFFSETS: Dict[str, Tuple[int, int]] = {
    ACTION_UP: (0, 1),
    ACTION_DOWN: (0, -1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
ACTIONS: Tuple[str, ...] = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)

# The cue sits one move above the start cell, so the corridor between it and the
# junction is `stem_length - 1` moves of identical observations.
CUE_ROW = 1


class TMazeStepChannel(Enum):
    """Per-step measurement channels written to ``StepData.info``."""

    CORRECT_ENDPOINT = "correct_endpoint"
    WRONG_ENDPOINT = "wrong_endpoint"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    RECORDED_STEP = "recorded_step"
    WALL_COLLISION = "wall_collision"


class TMazeMetric(Enum):
    """Episode-level metric names reported by ``compute_metrics``."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    WRONG_ENDPOINT_RATE = "wrong_endpoint_rate"
    ENDED_BY_GOAL_RATE = "ended_by_goal_rate"
    ENDED_BY_FAILURE_RATE = "ended_by_failure_rate"
    ENDED_BY_TIMEOUT_RATE = "ended_by_timeout_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_WALL_COLLISIONS = "average_wall_collisions"


def create_t_maze_state(
    position: Tuple[int, int], goal_side: float, cue_phase: float = CUE_UNSEEN
) -> np.ndarray:
    """Build a T-Maze state array.

    Args:
        position: ``(x, y)`` cell, with ``x`` signed about the stem.
        goal_side: :data:`GOAL_LEFT` or :data:`GOAL_RIGHT`.
        cue_phase: One of :data:`CUE_UNSEEN`, :data:`CUE_EMITTING`,
            :data:`CUE_CONSUMED`. Defaults to :data:`CUE_UNSEEN`.

    Returns:
        The state as a float64 array ``[x, y, goal_side, cue_phase]``.
    """
    return np.array(
        [float(position[0]), float(position[1]), float(goal_side), float(cue_phase)],
        dtype=np.float64,
    )


class TMazePOMDP(DiscreteActionsEnvironment):
    """T-Maze POMDP with a single-use noisy cue and a delayed, side-dependent reward.

    See the module docstring for the full model. The environment is a proper
    generative model: transitions and observations can be resampled from any state,
    and both carry densities, so it can back a particle filter and a tree search.

    Attributes:
        stem_length: Moves from the start cell up to the junction.
        arm_length: Moves from the junction out to an arm endpoint.
        cue_accuracy: Probability the cue names the true goal side.
        goal_reward: Paid for entering the correct endpoint.
        wrong_goal_penalty: Magnitude of the penalty for entering the wrong endpoint.
        step_penalty: Magnitude of the per-action cost on non-terminal actions.

    Example:
        >>> env = TMazePOMDP(discount_factor=0.95, stem_length=4, arm_length=1)
        >>> env.junction
        (0, 4)
        >>> env.left_endpoint, env.right_endpoint
        ((-1, 4), (1, 4))
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        stem_length: int = 4,
        arm_length: int = 1,
        cue_accuracy: float = 0.9,
        goal_reward: float = 10.0,
        wrong_goal_penalty: float = 10.0,
        step_penalty: float = 1.0,
        name: str = "TMazePOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ) -> None:
        """Initialize the T-Maze.

        Args:
            discount_factor: Discount factor for future rewards. Defaults to 0.95.
            stem_length: Moves from the start cell to the junction. Must be at least
                2, so the cue cell (one move up) is strictly below the junction and
                at least one identical-observation step separates reading the cue
                from acting on it. Defaults to 4.
            arm_length: Moves from the junction to an arm endpoint. Must be at least
                1. Defaults to 1.
            cue_accuracy: Probability the cue names the true goal side. Must lie in
                ``[0.5, 1.0]``; 1.0 is allowed and turns the task into pure memory,
                which is the setting that separates a belief-tracking planner from
                a reactive one without also testing how it handles noise. Values
                below 0.5 are rejected rather than silently relabelled — a cue that
                is anti-correlated with the goal is the same task with the labels
                swapped, and accepting it would make two different configs mean one
                environment. Defaults to 0.9.
            goal_reward: Reward for entering the correct endpoint. Defaults to 10.0.
            wrong_goal_penalty: Magnitude of the penalty for entering the wrong
                endpoint; applied as ``-wrong_goal_penalty``. Defaults to 10.0.
            step_penalty: Magnitude of the per-action cost, applied as
                ``-step_penalty`` on every non-terminal action that does not enter
                an endpoint, wall collisions included. Defaults to 1.0.
            name: Environment name. Defaults to ``"TMazePOMDP"``.
            output_dir: Optional directory for logging output.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the maze would be degenerate (``stem_length < 2`` or
                ``arm_length < 1``) or ``cue_accuracy`` is outside ``[0.5, 1.0]``.
        """
        if stem_length < 2:
            raise ValueError(
                f"stem_length must be at least 2 so the cue cell sits strictly below "
                f"the junction, got {stem_length}."
            )
        if arm_length < 1:
            raise ValueError(f"arm_length must be at least 1, got {arm_length}.")
        if not 0.5 <= cue_accuracy <= 1.0:
            raise ValueError(f"cue_accuracy must lie in [0.5, 1.0], got {cue_accuracy}.")

        self.stem_length = int(stem_length)
        self.arm_length = int(arm_length)
        self.cue_accuracy = float(cue_accuracy)
        self.goal_reward = float(goal_reward)
        self.wrong_goal_penalty = float(wrong_goal_penalty)
        self.step_penalty = float(step_penalty)

        # Every reward the environment can produce, enumerated rather than
        # estimated: -wrong_goal_penalty on entering the wrong endpoint,
        # -step_penalty on every other non-terminal action, 0.0 from an absorbing
        # terminal state, and +goal_reward on entering the correct endpoint. The
        # terminal payouts replace the step penalty rather than stacking with it,
        # so no two terms ever add. The 0.0 terms keep the bound honest when a
        # caller sets a penalty to zero or a negative reward.
        min_reward = min(0.0, -self.wrong_goal_penalty, -self.step_penalty, self.goal_reward)
        max_reward = max(0.0, self.goal_reward, -self.step_penalty, -self.wrong_goal_penalty)

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        # Derived geometry. Private so it stays out of ``config_id`` and ``to_dict``:
        # it is a pure function of the constructor arguments, and hashing it would
        # only add ways for the identity to drift.
        self._cue_cell: Tuple[int, int] = (0, CUE_ROW)
        self._junction: Tuple[int, int] = (0, self.stem_length)
        self._left_endpoint: Tuple[int, int] = (-self.arm_length, self.stem_length)
        self._right_endpoint: Tuple[int, int] = (self.arm_length, self.stem_length)
        self._valid_cells = frozenset(
            [(0, y) for y in range(self.stem_length + 1)]
            + [(x, self.stem_length) for x in range(-self.arm_length, self.arm_length + 1)]
        )

    # ── Geometry ────────────────────────────────────────────────────────
    @property
    def start_cell(self) -> Tuple[int, int]:
        """The fixed cell every episode starts from."""
        return (0, 0)

    @property
    def cue_cell(self) -> Tuple[int, int]:
        """The cell whose entry emits the one cue reading of the episode."""
        return self._cue_cell

    @property
    def junction(self) -> Tuple[int, int]:
        """The cell where the stem meets the two arms."""
        return self._junction

    @property
    def left_endpoint(self) -> Tuple[int, int]:
        """The terminal cell at the end of the left arm."""
        return self._left_endpoint

    @property
    def right_endpoint(self) -> Tuple[int, int]:
        """The terminal cell at the end of the right arm."""
        return self._right_endpoint

    @property
    def valid_cells(self) -> frozenset:
        """Every cell of the T, as ``(x, y)`` pairs."""
        return self._valid_cells

    def is_valid_cell(self, position: Tuple[int, int]) -> bool:
        """Whether ``position`` is inside the T-shaped corridor."""
        return (int(position[0]), int(position[1])) in self._valid_cells

    def goal_endpoint(self, goal_side: float) -> Tuple[int, int]:
        """The endpoint that pays the goal reward for ``goal_side``."""
        return self._left_endpoint if float(goal_side) == GOAL_LEFT else self._right_endpoint

    # ── Core dynamics ───────────────────────────────────────────────────
    def get_actions(self) -> List[str]:
        """The four movement actions, in a fixed order."""
        return list(ACTIONS)

    def _next_position(self, position: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Where ``action`` lands from ``position``; unchanged if it hits a wall."""
        dx, dy = ACTION_OFFSETS[action]
        candidate = (position[0] + dx, position[1] + dy)
        return candidate if candidate in self._valid_cells else position

    def _next_cue_phase(self, phase: float, next_position: Tuple[int, int]) -> float:
        """Advance the cue's delivery phase across one transition.

        An emitting cue is consumed by whatever action follows it, including one
        that bumps into a wall and leaves the agent standing on the cue cell — that
        is what makes the cue single-use rather than re-readable by standing still.
        """
        if phase == CUE_EMITTING:
            return CUE_CONSUMED
        if phase == CUE_UNSEEN and next_position == self._cue_cell:
            return CUE_EMITTING
        return phase

    def _successor(self, state: Any, action: str) -> np.ndarray:
        """The single deterministic successor of ``(state, action)``."""
        state_array = np.asarray(state, dtype=np.float64)
        if self.is_terminal(state_array):
            # Absorbing: an endpoint keeps its own state forever, so an over-long
            # episode cannot walk back out of a terminal state or be paid twice.
            return state_array.copy()
        position = (int(state_array[STATE_X]), int(state_array[STATE_Y]))
        next_position = self._next_position(position, action)
        return np.array(
            [
                float(next_position[0]),
                float(next_position[1]),
                float(state_array[STATE_GOAL]),
                self._next_cue_phase(float(state_array[STATE_CUE_PHASE]), next_position),
            ],
            dtype=np.float64,
        )

    def is_terminal(self, state: Any) -> bool:
        """Whether ``state`` sits on either arm endpoint."""
        state_array = np.asarray(state, dtype=np.float64)
        position = (int(state_array[STATE_X]), int(state_array[STATE_Y]))
        return position in (self._left_endpoint, self._right_endpoint)

    def sample_next_state(self, state: Any, action: str, n_samples: int = 1) -> Any:
        """Sample successors. Transitions are deterministic, so all samples agree."""
        successor = self._successor(state, action)
        if n_samples == 1:
            return successor
        return np.repeat(successor[np.newaxis, :], n_samples, axis=0)

    def sample_next_state_batch(self, states: Any, action: str) -> np.ndarray:
        """Batch transition. Returns float64, matching :meth:`sample_next_state`.

        Returning the caller's dtype here would hand a particle filter int64
        particles from the batch path and float64 ones from the single path, and a
        belief that mixes the two would silently truncate half of them.
        """
        states_array = np.asarray(states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        return np.stack([self._successor(row, action) for row in states_array], axis=0)

    def transition_log_probability(self, state: Any, action: str, next_states: Any) -> np.ndarray:
        """Log ``T(s' | s, a)``: 0 for the one successor, ``-inf`` everywhere else."""
        successor = self._successor(state, action)
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        matches = np.all(np.isclose(candidates, successor[np.newaxis, :]), axis=1)
        return np.where(matches, 0.0, -np.inf)

    # ── Observation model ───────────────────────────────────────────────
    def _observation_probs(self, next_state: Any) -> np.ndarray:
        """``P(o | s')`` over :data:`OBSERVATIONS`, in that order.

        Normalised for every state the environment can reach and for every accuracy
        in ``[0.5, 1.0]``: an emitting cue puts all its mass on the two cue readings,
        and everything else puts all of it on ``"empty"``.
        """
        state_array = np.asarray(next_state, dtype=np.float64)
        probs = np.zeros(len(OBSERVATIONS), dtype=np.float64)
        if float(state_array[STATE_CUE_PHASE]) != CUE_EMITTING:
            probs[OBSERVATIONS.index(OBSERVATION_EMPTY)] = 1.0
            return probs
        if float(state_array[STATE_GOAL]) == GOAL_LEFT:
            probs[OBSERVATIONS.index(OBSERVATION_LEFT_CUE)] = self.cue_accuracy
            probs[OBSERVATIONS.index(OBSERVATION_RIGHT_CUE)] = 1.0 - self.cue_accuracy
        else:
            probs[OBSERVATIONS.index(OBSERVATION_RIGHT_CUE)] = self.cue_accuracy
            probs[OBSERVATIONS.index(OBSERVATION_LEFT_CUE)] = 1.0 - self.cue_accuracy
        return probs

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        """Draw an observation from ``P(o | s')``. ``action`` does not enter it.

        Inlined rather than routed through a freshly built
        :class:`~POMDPPlanners.core.distributions.DiscreteDistribution`: this runs
        once per node expansion inside a tree search on a wall-clock budget, so an
        allocation and a validation pass per call buys nothing. The draw is the
        distribution's own — one ``np.random.rand`` per sample, in order, against
        the cumulative probabilities — so the RNG stream is unchanged.
        """
        del action
        cumulative = np.cumsum(self._observation_probs(next_state))
        last = len(OBSERVATIONS) - 1
        if n_samples == 1:
            return OBSERVATIONS[min(int(np.searchsorted(cumulative, np.random.rand())), last)]
        indices = np.clip(np.searchsorted(cumulative, np.random.rand(n_samples)), 0, last)
        return [OBSERVATIONS[index] for index in indices]

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` for each observation in ``observations``."""
        del action
        probs = self._observation_probs(next_state)
        lookup = {name: float(probs[index]) for index, name in enumerate(OBSERVATIONS)}
        values = [observations] if isinstance(observations, str) else list(observations)
        out = np.full(len(values), -np.inf, dtype=np.float64)
        for index, value in enumerate(values):
            probability = lookup.get(value, 0.0)
            if probability > 0.0:
                out[index] = float(np.log(probability))
        return out

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        """Log ``Z(o | s')`` of one observation against many candidate states.

        Vectorised over the particles rather than looped: this is the particle
        filter's reweighting step, so it runs once per belief update over every
        particle, and the likelihood is a two-way choice that numpy can express
        directly. The three cases below are exactly the ones
        :meth:`_observation_probs` enumerates.
        """
        del action
        states_array = np.asarray(next_states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)

        emitting = states_array[:, STATE_CUE_PHASE] == CUE_EMITTING
        if observation == OBSERVATION_EMPTY:
            # Only a non-emitting state can produce "empty", and it does so with
            # probability 1.
            return np.where(emitting, -np.inf, 0.0)
        if observation not in (OBSERVATION_LEFT_CUE, OBSERVATION_RIGHT_CUE):
            return np.full(len(states_array), -np.inf, dtype=np.float64)

        names_left = observation == OBSERVATION_LEFT_CUE
        goal_is_left = states_array[:, STATE_GOAL] == GOAL_LEFT
        matches = goal_is_left if names_left else ~goal_is_left
        with np.errstate(divide="ignore"):
            log_accuracy = float(np.log(self.cue_accuracy))
            log_error = float(np.log(1.0 - self.cue_accuracy))
        return np.where(emitting, np.where(matches, log_accuracy, log_error), -np.inf)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Observations are labels, so equality is string equality."""
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Hashable:
        """The label itself is already a hashable key."""
        return observation

    def hash_action(self, action: Any) -> Hashable:
        """The action label itself is already a hashable key."""
        return action

    # ── Reward ──────────────────────────────────────────────────────────
    def _reward_from_successor(self, state_array: np.ndarray, successor: np.ndarray) -> float:
        if self.is_terminal(state_array):
            return 0.0
        position = (int(successor[STATE_X]), int(successor[STATE_Y]))
        if position == self.goal_endpoint(float(state_array[STATE_GOAL])):
            return float(self.goal_reward)
        if position in (self._left_endpoint, self._right_endpoint):
            return float(-self.wrong_goal_penalty)
        return float(-self.step_penalty)

    def reward(self, state: Any, action: str, next_state: Any = None) -> float:
        """Immediate reward for ``(state, action)``.

        Transitions are deterministic, so ``next_state`` carries no information the
        environment cannot recompute; it is accepted and used when supplied only to
        keep this consistent with the driver that threads it through.
        """
        state_array = np.asarray(state, dtype=np.float64)
        successor = (
            self._successor(state_array, action)
            if next_state is None
            else np.asarray(next_state, dtype=np.float64)
        )
        return self._reward_from_successor(state_array, successor)

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: str,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Rewards for many states under one action."""
        states_array = np.asarray(states, dtype=np.float64)
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        if next_states is None:
            successors = self.sample_next_state_batch(states_array, action)
        else:
            successors = np.asarray(next_states, dtype=np.float64)
            if successors.ndim == 1:
                successors = successors.reshape(1, -1)
        return np.array(
            [
                self._reward_from_successor(state_row, successor_row)
                for state_row, successor_row in zip(states_array, successors)
            ],
            dtype=np.float64,
        )

    # ── Initial distributions ───────────────────────────────────────────
    def initial_state_dist(self) -> Distribution:
        """Uniform over the two goal sides, at the start cell with the cue unseen."""
        return DiscreteDistribution(
            values=[
                create_t_maze_state(self.start_cell, GOAL_LEFT, CUE_UNSEEN),
                create_t_maze_state(self.start_cell, GOAL_RIGHT, CUE_UNSEEN),
            ],
            probs=np.array([0.5, 0.5], dtype=np.float64),
        )

    def initial_observation_dist(self) -> Distribution:
        """Always ``"empty"``.

        The cue is never in the initial reading: it is emitted by *entering* the cue
        cell, and the runner only observes after an action. Putting it here would
        hand the agent the answer before it had moved.
        """
        return DiscreteDistribution(
            values=[OBSERVATION_EMPTY], probs=np.array([1.0], dtype=np.float64)
        )

    # ── Metrics ─────────────────────────────────────────────────────────
    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels for one transition.

        Every channel is a pure function of the arguments — no draw, no state kept
        on ``self`` — as :meth:`Environment.step_info` requires.

        The episode-end channels are read off the *outcome* of the step: the
        realised successor when there is one, and ``state`` itself on the terminal
        bookkeeping call, where ``action`` and ``next_state`` are both ``None``.
        Reading them off ``state`` alone would report a timeout on the very step
        that entered an endpoint, which is the last recorded step whenever a runner
        stops without appending the terminal record.

        Args:
            state: The state the step was taken from, or the final state.
            action: The action taken, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.

        Returns:
            One entry per :class:`TMazeStepChannel`.
        """
        state_array = np.asarray(state, dtype=np.float64)
        outcome = state_array if next_state is None else np.asarray(next_state, dtype=np.float64)
        outcome_position = (int(outcome[STATE_X]), int(outcome[STATE_Y]))
        goal_endpoint = self.goal_endpoint(float(state_array[STATE_GOAL]))

        at_correct = float(outcome_position == goal_endpoint)
        at_wrong = float(
            outcome_position in (self._left_endpoint, self._right_endpoint)
            and outcome_position != goal_endpoint
        )
        # A wall collision is an action that left the position unchanged from a
        # non-terminal state. The terminal bookkeeping step took no action, and a
        # state that is already terminal is absorbing rather than blocked, so both
        # report 0.0.
        collided = float(
            action is not None
            and next_state is not None
            and not self.is_terminal(state_array)
            and outcome_position == (int(state_array[STATE_X]), int(state_array[STATE_Y]))
        )
        return {
            TMazeStepChannel.CORRECT_ENDPOINT.value: at_correct,
            TMazeStepChannel.WRONG_ENDPOINT.value: at_wrong,
            TMazeStepChannel.ENDED_BY_GOAL.value: at_correct,
            TMazeStepChannel.ENDED_BY_FAILURE.value: at_wrong,
            TMazeStepChannel.ENDED_BY_TIMEOUT.value: float(at_correct == 0.0 and at_wrong == 0.0),
            TMazeStepChannel.RECORDED_STEP.value: 1.0,
            TMazeStepChannel.WALL_COLLISION.value: collided,
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the episode metrics derived from the per-step channels.

        Completion reduces with ``ANY`` because the task is to *reach* something:
        once the correct endpoint is entered the episode is over, and no later step
        can undo it.

        Reaching the wrong endpoint is reported separately from timing out because
        the two call for opposite fixes — a planner that guesses at the junction
        needs a better belief, one that times out needs a longer horizon — and a
        completion rate alone cannot tell them apart.

        There is no severity metric here. The only hazard is entering the wrong
        endpoint, which happens at most once per episode and always costs the same;
        a "worst moment" number would restate the count.

        Returns:
            One spec per metric, in the order ``compute_metrics`` reports them.
        """
        return [
            StepInfoMetric(
                name=TMazeMetric.TASK_COMPLETION_RATE.value,
                channel=TMazeStepChannel.CORRECT_ENDPOINT.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=TMazeMetric.WRONG_ENDPOINT_RATE.value,
                channel=TMazeStepChannel.WRONG_ENDPOINT.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=TMazeMetric.ENDED_BY_GOAL_RATE.value,
                channel=TMazeStepChannel.ENDED_BY_GOAL.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=TMazeMetric.ENDED_BY_FAILURE_RATE.value,
                channel=TMazeStepChannel.ENDED_BY_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=TMazeMetric.ENDED_BY_TIMEOUT_RATE.value,
                channel=TMazeStepChannel.ENDED_BY_TIMEOUT.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=TMazeMetric.AVERAGE_EPISODE_LENGTH.value,
                channel=TMazeStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=TMazeMetric.AVERAGE_WALL_COLLISIONS.value,
                channel=TMazeStepChannel.WALL_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
        ]

    # ── Visualization ───────────────────────────────────────────────────
    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Write a GIF of one episode, belief included.

        Args:
            history: The episode's step records.
            output_dir: Directory the ``.gif`` is written into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported here so the environment can be constructed and planned on without
        # matplotlib installed, matching how the other grid environments defer it.
        from POMDPPlanners.environments.t_maze_pomdp.t_maze_visualizer import (  # pylint: disable=import-outside-toplevel
            TMazeVisualizer,
        )

        TMazeVisualizer(self).create_visualization(
            history, output_dir / f"agent_path_{episode_index}.gif"
        )


__all__ = [
    "ACTIONS",
    "CUE_CONSUMED",
    "CUE_EMITTING",
    "CUE_UNSEEN",
    "GOAL_LEFT",
    "GOAL_RIGHT",
    "OBSERVATIONS",
    "OBSERVATION_EMPTY",
    "OBSERVATION_LEFT_CUE",
    "OBSERVATION_RIGHT_CUE",
    "STATE_CUE_PHASE",
    "STATE_GOAL",
    "STATE_WIDTH",
    "STATE_X",
    "STATE_Y",
    "TMazeMetric",
    "TMazePOMDP",
    "TMazeStepChannel",
    "create_t_maze_state",
]
