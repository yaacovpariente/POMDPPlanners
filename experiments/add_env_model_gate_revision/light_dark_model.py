# SPDX-License-Identifier: MIT

"""LightDark as the planner's model, with its dynamics and observation swapped out.

Everything else -- reward, terminal rule, hazard draw, geometry, metrics,
serialization, config identity -- is inherited from the true environment, so a
candidate differs from the world only in the two things it claims to model.
"""

from typing import Any, Optional

import numpy as np

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ObservationModelType,
)

from experiments.add_env_model_gate_revision.models import ObservationCandidate


class SubstitutedLightDarkModel(ContinuousLightDarkPOMDP):
    """The LightDark task planned with a substituted transition and observation model.

    Args:
        transition: Drives the 2-D position block. The hazard terminal slot, when
            the world carries one, is drawn by the inherited hazard rule from the
            successor position -- it is task logic, not dynamics.
        observation: Emits and scores 2-D position observations.
        model_label: Public tag naming the candidate. It enters ``config_id``,
            and so the cache key, together with the fingerprints below.
        discount_factor: Must equal the world's.
        **world_kwargs: The world's own constructor arguments.
    """

    def __init__(
        self,
        transition: TransitionModel,
        observation: ObservationCandidate,
        model_label: str,
        discount_factor: float,
        **world_kwargs: Any,
    ) -> None:
        world_kwargs = dict(world_kwargs)
        world_kwargs.setdefault("name", f"LightDarkModel[{model_label}]")
        super().__init__(discount_factor=discount_factor, **world_kwargs)
        if self.observation_model_type is not ObservationModelType.NORMAL_NOISE:
            raise NotImplementedError("only the NORMAL_NOISE observation model is substituted")
        self._transition = transition
        self._observation = observation
        self.model_label = str(model_label)
        # Public on purpose: config_id skips private attributes, and a refit must
        # move the cache key.
        self.transition_fingerprint: Optional[str] = getattr(transition, "fingerprint", None)
        self.observation_fingerprint: Optional[str] = getattr(observation, "fingerprint", None)

    @property
    def transition(self) -> TransitionModel:
        return self._transition

    @property
    def observation(self) -> ObservationCandidate:
        return self._observation

    # -- dynamics ----------------------------------------------------------

    def sample_next_state(self, state: np.ndarray, action: np.ndarray, n_samples: int = 1) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64)
        if self._hazard_terminal_enabled and state_arr.shape[0] > 2 and float(state_arr[2]) > 0.5:
            frozen = state_arr.copy()
            return frozen if n_samples == 1 else np.repeat(frozen[np.newaxis, :], n_samples, axis=0)
        pos = np.ascontiguousarray(state_arr[:2])
        raw = np.asarray(self._transition.sample_next_state(pos, action, n_samples), dtype=np.float64)
        if not self._hazard_terminal_enabled:
            return raw
        if n_samples == 1:
            nxt = raw.reshape(-1)
            return np.concatenate([nxt, [self._draw_terminal_slot(nxt)]])
        raw = raw.reshape(n_samples, 2)
        slots = np.array([self._draw_terminal_slot(raw[i]) for i in range(n_samples)])
        return np.concatenate([raw, slots.reshape(-1, 1)], axis=1)

    def sample_next_state_batch(self, states: Any, action: np.ndarray) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        pos = np.ascontiguousarray(states_array[:, :2])
        action_rows = np.broadcast_to(np.asarray(action, dtype=np.float64).reshape(1, -1), (len(pos), 2))
        raw = np.asarray(self._transition.sample_next_state(pos, action_rows), dtype=np.float64).reshape(len(pos), 2)
        if not self._hazard_terminal_enabled:
            return raw
        input_slots = states_array[:, 2] if states_array.shape[1] > 2 else np.zeros(len(states_array))
        out = np.empty((len(raw), 3), dtype=np.float64)
        for i, raw_row in enumerate(raw):
            if input_slots[i] > 0.5:
                out[i, :2] = pos[i]
                out[i, 2] = 1.0
            else:
                out[i, :2] = raw_row
                out[i, 2] = self._draw_terminal_slot(raw_row)
        return out

    def transition_log_probability(self, state: np.ndarray, action: np.ndarray, next_states: Any) -> np.ndarray:
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        return np.asarray(
            self._transition.log_probability(self._pos2d(state), action, self._pos2d(candidates)), dtype=np.float64
        )

    # -- observation -------------------------------------------------------

    def sample_observation(self, next_state: np.ndarray, action: np.ndarray, n_samples: int = 1) -> Any:
        del action
        return self._observation.sample(self._pos2d(next_state), n_samples)

    def observation_log_probability(self, next_state: np.ndarray, action: np.ndarray, observations: Any) -> np.ndarray:
        del action
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return self._observation.log_probability(self._pos2d(next_state), obs)

    def observation_log_probability_per_state(self, next_states: Any, action: np.ndarray, observation: Any) -> np.ndarray:
        del action
        return self._observation.log_probability_per_state(self._pos2d(np.asarray(next_states)), np.asarray(observation))

    def observation_log_probability_single(self, next_state: Any, action: Any, observation: Any) -> float:
        del action
        return float(self._observation.log_probability_per_state(self._pos2d(np.asarray(next_state)).reshape(1, -1), np.asarray(observation))[0])
