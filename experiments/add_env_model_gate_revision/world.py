# SPDX-License-Identifier: MIT

"""The pinned LightDark world, the preset table, and a 2-D view for the diagnostics."""

from typing import Any, Dict, Optional

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_light_dark_pinned_kwargs

from experiments.add_env_model_gate_revision.light_dark_model import SubstitutedLightDarkModel
from experiments.add_env_model_gate_revision.models import (
    GaussianMLPObservation,
    GaussianMLPTransition,
    TrueLightDarkObservation,
    bad_transition,
    truth_transition,
)

DISCOUNT = 0.95
#: Planner depth; also the horizon the drift gate is measured at.
PLANNING_DEPTH = 10

#: Eight directions at two magnitudes. The planner samples the unit disc; this
#: table is the discrete stand-in the exploration and the ranking gate use.
_ANGLES = np.arange(8) * (2.0 * np.pi / 8)
PRESETS = np.concatenate(
    [np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1) * mag for mag in (0.5, 1.0)]
)


#: Pinned kwargs, except that hazard hits are not terminal. With the terminal
#: slot on, PFT-DPW's tree scores immediate rewards through ``reward_batch``
#: without realised next states, where the hazard penalty never fires, and the
#: planner walks into the obstacles (see results/.../qa_hazard_terminal_config).
HAZARD_TERMINAL = False


def world_kwargs(**overrides: Any) -> Dict[str, Any]:
    return continuous_light_dark_pinned_kwargs(is_obstacle_hit_terminal=HAZARD_TERMINAL, **overrides)


def make_world(name: Optional[str] = None) -> ContinuousLightDarkPOMDP:
    kwargs = world_kwargs()
    if name is not None:
        kwargs["name"] = name
    return ContinuousLightDarkPOMDP(discount_factor=DISCOUNT, **kwargs)


def true_observation(world: ContinuousLightDarkPOMDP) -> TrueLightDarkObservation:
    return TrueLightDarkObservation(world.beacons, world.beacon_radius, world.observation_cov_matrix)


def make_truth_model(world: ContinuousLightDarkPOMDP) -> SubstitutedLightDarkModel:
    return SubstitutedLightDarkModel(
        transition=truth_transition(world.state_transition_cov_matrix),
        observation=true_observation(world),
        model_label="truth",
        discount_factor=DISCOUNT,
        **world_kwargs(),
    )


def make_bad_model(world: ContinuousLightDarkPOMDP) -> SubstitutedLightDarkModel:
    return SubstitutedLightDarkModel(
        transition=bad_transition(world.state_transition_cov_matrix, std_factor=2.0),
        observation=true_observation(world),
        model_label="bad-transition-std-x2",
        discount_factor=DISCOUNT,
        **world_kwargs(),
    )


def make_learned_model(
    transition: GaussianMLPTransition, observation: GaussianMLPObservation
) -> SubstitutedLightDarkModel:
    return SubstitutedLightDarkModel(
        transition=transition,
        observation=observation,
        model_label=f"learned-{transition.fingerprint[:8]}-{observation.fingerprint[:8]}",
        discount_factor=DISCOUNT,
        **world_kwargs(),
    )


class WorldPositionView:
    """The world seen through its 2-D position block, for the diagnostics.

    The diagnostics roll the model and the world side by side and need both to
    speak the fitted block. The world's hazard terminal slot is outside that
    block; here the successor position is returned and the slot is dropped, and
    the reward is scored on positions, so the hazard penalty -- a Bernoulli on
    the slot -- is outside what the ranking gate sees.
    """

    def __init__(self, world: ContinuousLightDarkPOMDP) -> None:
        self._world = world

    def _pad(self, position: np.ndarray) -> np.ndarray:
        return self._world._pad_terminal_slot(np.asarray(position, dtype=float).reshape(-1)[:2])  # pylint: disable=protected-access

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        successors = np.atleast_2d(self._world.sample_next_state(self._pad(state), np.asarray(action, dtype=float), n_samples))[:, :2]
        return successors[0] if n_samples == 1 else successors

    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        return float(
            self._world.reward(
                np.asarray(state, dtype=float).reshape(-1)[:2],
                np.asarray(action, dtype=float),
                next_state=np.asarray(next_state, dtype=float).reshape(-1)[:2],
            )
        )
