# SPDX-License-Identifier: MIT

"""The pure-Python Snake kernels that the C++ ``_native`` module replaced.

``SnakePOMDP`` now sends its transition, observation, reward and rollout calls
to ``snake_pomdp/_native``. This subclass keeps the Python code those calls
replaced, copied unchanged, so the native-equivalence tests can compare the two
on the same inputs. It draws from ``np.random``; the native kernels draw from
the module's own C++ RNG, so the tests compare sampled outputs by distribution
and deterministic outputs exactly.
"""

from collections.abc import Sequence as AbcSequence
from typing import Any, List

import numpy as np

from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    OBSERVATION_LIVE,
    TERMINAL_OBSERVATION,
    SnakeObservation,
    SnakePOMDP,
    SnakeQuadrant,
    SnakeState,
    SnakeTermination,
)


class SnakePOMDPPythonReference(SnakePOMDP):
    """``SnakePOMDP`` with the pre-port Python kernels."""

    def sample_next_state(self, state: SnakeState, action: int, n_samples: int = 1) -> Any:
        count = int(n_samples)
        if self.is_terminal(state):
            absorbing = np.array(state, dtype=np.float64, copy=True)
            return absorbing if count == 1 else np.tile(absorbing, (count, 1))

        new_body, eat, counter, termination = self.transition_outcome(state, int(action))
        previous_food = self.food(state)

        respawns = eat and termination is SnakeTermination.RUNNING
        if not respawns:
            successor = self._successor(new_body, eat, counter, termination, previous_food)
            return successor if count == 1 else np.tile(successor, (count, 1))

        free = self.free_cells(new_body)
        if free.size == 0:
            raise ValueError("the body fills the grid, so the food has nowhere to respawn")
        drawn = free[np.random.randint(0, free.size, size=count)]
        samples = [
            self._successor(new_body, eat, counter, termination, previous_food, int(cell))
            for cell in drawn
        ]
        return samples[0] if count == 1 else np.asarray(samples, dtype=np.float64)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.asarray(
            [self.sample_next_state(state=row, action=action) for row in rows], dtype=np.float64
        )

    def transition_log_probability(
        self, state: SnakeState, action: int, next_states: Any
    ) -> np.ndarray:
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)

        if self.is_terminal(state):
            expected = np.asarray(state, dtype=np.float64)
            return np.where(np.all(candidates == expected, axis=1), 0.0, -np.inf)

        new_body, eat, counter, termination = self.transition_outcome(state, int(action))
        previous_food = self.food(state)
        if not (eat and termination is SnakeTermination.RUNNING):
            expected = self._successor(new_body, eat, counter, termination, previous_food)
            return np.where(np.all(candidates == expected, axis=1), 0.0, -np.inf)

        free = self.free_cells(new_body)
        log_probability = -float(np.log(free.size))
        for cell in free:
            expected = self._successor(
                new_body, eat, counter, termination, previous_food, int(cell)
            )
            scores[np.all(candidates == expected, axis=1)] = log_probability
        return scores

    def sample_observation(self, next_state: SnakeState, action: int, n_samples: int = 1) -> Any:
        del action
        count = int(n_samples)
        if self.is_terminal(next_state):
            return TERMINAL_OBSERVATION if count == 1 else [TERMINAL_OBSERVATION] * count

        cells = self.body(next_state)
        head = cells[0]
        food = self._running_food(next_state)
        inside = food in self.window_cells(head)
        scent_probabilities = self.scent_probabilities(head, food)

        readings: List[SnakeObservation] = []
        for _ in range(count):
            detected = inside and bool(np.random.random() < self.detection_probability)
            quadrant = int(np.random.choice(len(SnakeQuadrant), p=scent_probabilities))
            readings.append(
                self.encode_observation_tuple(cells, food if detected else None, quadrant)
            )
        return readings[0] if count == 1 else readings

    def observation_log_probability(
        self, next_state: SnakeState, action: int, observations: Any
    ) -> np.ndarray:
        del action
        candidates = _as_observation_list(observations)
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)

        if self.is_terminal(next_state):
            for index, candidate in enumerate(candidates):
                scores[index] = 0.0 if tuple(candidate) == TERMINAL_OBSERVATION else -np.inf
            return scores

        cells = self.body(next_state)
        head = cells[0]
        food = self._running_food(next_state)
        inside = food in self.window_cells(head)
        scent_probabilities = self.scent_probabilities(head, food)

        with np.errstate(divide="ignore"):
            log_scent = np.log(scent_probabilities)
        for index, candidate in enumerate(candidates):
            flat = tuple(int(value) for value in candidate)
            if not flat or flat[0] != OBSERVATION_LIVE:
                continue
            observed_body, seen, scent = self.decode_observation(flat)
            if observed_body != cells or not 0 <= scent < len(SnakeQuadrant):
                continue
            if seen is None:
                sighting = np.log1p(-self.detection_probability) if inside else 0.0
            elif inside and seen == food:
                sighting = float(np.log(self.detection_probability))
            else:
                continue
            scores[index] = float(sighting) + float(log_scent[scent])
        return scores

    def reward(self, state: SnakeState, action: int, next_state: Any = None) -> float:
        del next_state
        if self.is_terminal(state):
            return 0.0
        _, eat, _, termination = self.transition_outcome(state, int(action))
        if eat:
            return 1.0
        if termination in (
            SnakeTermination.WALL,
            SnakeTermination.SELF,
            SnakeTermination.STARVATION,
        ):
            return -1.0
        return 0.0

    def reward_batch(self, states: Any, action: int, next_states: Any = None) -> np.ndarray:
        del next_states
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.array([self.reward(row, action) for row in rows], dtype=np.float64)

    def is_terminal(self, state: SnakeState) -> bool:
        return int(round(float(np.asarray(state, dtype=np.float64)[0]))) != int(
            SnakeTermination.RUNNING
        )

    def sample_next_step(self, state: Any, action: Any) -> Any:
        next_state = self.sample_next_state(state=state, action=action)
        observation = self.sample_observation(next_state=next_state, action=action)
        return next_state, observation, self.reward(state, action, next_state)

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """The generic recursive rollout ``python_random_rollout`` runs."""
        if depth >= max_depth or self.is_terminal(state):
            return 0.0
        action = action_sampler.sample()
        next_state = self.sample_next_state(state=state, action=action)
        reward = self.reward(state=state, action=action, next_state=next_state)
        return reward + discount_factor * self.simulate_random_rollout(
            next_state, action_sampler, max_depth, discount_factor, depth + 1
        )


def _as_observation_list(observations: Any) -> List[SnakeObservation]:
    if (
        isinstance(observations, tuple)
        and observations
        and isinstance(observations[0], (int, np.integer))
    ):
        return [observations]
    if isinstance(observations, AbcSequence) and not isinstance(observations, (str, bytes)):
        return [tuple(int(value) for value in reading) for reading in observations]
    return [tuple(int(value) for value in observations)]
