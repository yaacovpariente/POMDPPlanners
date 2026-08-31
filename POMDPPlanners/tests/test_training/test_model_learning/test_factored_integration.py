# SPDX-License-Identifier: MIT

"""Integration test: the loop against a factored world that carries a latent channel.

This is the shape every real use has. The world's state is wider than the block
being fitted, and the extra width is a latent fixed at episode start -- a hazard
type -- which the transition must carry unchanged. Fitting over it does not
raise; it regresses a persistent hidden property as if it were a random variable
of the dynamics, flattening exactly the belief dispersion a risk-sensitive
planner is there to grade.

So the property under test is not "the loop runs". It is that a model fitted
through the loop can be dropped back into the factored model as its transition
and the latent still survives a step untouched.

No Isaac Sim is involved: these generative models are standalone by design.
"""

from typing import Any, Tuple

import numpy as np

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    FactoredIsaacModelPOMDP,
    IsaacChannelSchema,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    LinearGaussianTransition,
)
from POMDPPlanners.training.model_learning import (
    DAggerModelTrainer,
    LinearGaussianLearner,
    TransitionDataset,
    block_indices,
)

SCHEMA = IsaacChannelSchema((("robot", 3), ("hazard_type", 2)))
ROBOT = ("robot",)
PRESETS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])


def _true_world() -> FactoredIsaacModelPOMDP:
    """A factored world whose transition drives the robot block only."""
    transition = LinearGaussianTransition(
        weight_state=np.array([[0.9, 0.05, 0.0], [0.0, 0.92, 0.05], [0.0, 0.0, 0.95]]),
        weight_action=np.array([[0.4, 0.0], [0.1, 0.3], [0.0, 0.2]]),
        bias=np.array([0.01, 0.0, -0.01]),
        covariance=np.diag([0.0025, 0.0025, 0.0025]),
    )
    return FactoredIsaacModelPOMDP(
        state_schema=SCHEMA,
        action_presets=list(PRESETS),
        discount_factor=0.99,
        transition=transition,
        transition_channels=ROBOT,
    )


class _WorldWithStart:
    """The factored world plus the start state and terminal rule the loop needs."""

    def __init__(self, model: FactoredIsaacModelPOMDP) -> None:
        self._model = model

    def initial_state_dist(self) -> Any:
        schema = SCHEMA

        class _Dist:
            def sample(self, num_samples: int = 1) -> np.ndarray:
                del num_samples
                # A latent drawn once per episode, exactly what must not be fitted.
                return schema.pack({"robot": np.zeros(3), "hazard_type": np.array([1.0, 0.0])})

        return _Dist()

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        return self._model.sample_next_state(state, action, n_samples)

    def is_terminal(self, state: Any) -> bool:
        del state
        return False


def _robot_view(world: _WorldWithStart) -> Any:
    """A view of the world over the robot block alone, for the diagnostics."""
    indices = block_indices(SCHEMA, ROBOT)

    class _View:
        def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
            full = SCHEMA.pack(
                {"robot": np.asarray(state, dtype=float), "hazard_type": np.array([1.0, 0.0])}
            )
            successors = np.atleast_2d(world.sample_next_state(full, action, n_samples))[
                :, indices
            ]
            return successors[0] if n_samples == 1 else successors

    return _View()


def _run_loop(rounds: int) -> Tuple[Any, list]:
    world = _WorldWithStart(_true_world())
    dataset = TransitionDataset(
        state_indices=block_indices(SCHEMA, ROBOT), holdout_fraction=0.25, seed=4
    )
    trainer = DAggerModelTrainer(
        world=world,
        learner=LinearGaussianLearner(),
        dataset=dataset,
        action_presets=PRESETS,
        # The diagnostics roll model and world side by side, so the world they
        # get must speak the fitted block, not the full state.
        diagnostics_world=_robot_view(world),
        planner_rollout_fn=None,
        num_rounds=rounds,
        episodes_per_round=12,
        steps_per_episode=25,
        horizon=10,
        seed=0,
    )
    return world, trainer.run()


def test_a_model_fitted_through_the_loop_drops_back_into_the_factored_model() -> None:
    """Purpose: Validates that a learned transition is usable as the factored model's transition

    Given: A learned transition over the 3-wide robot block of a 5-wide state
    When: It is installed in a factored model and a step is taken
    Then: The step succeeds and returns a full 5-wide successor
    """
    _, rounds = _run_loop(rounds=2)

    planner_model = FactoredIsaacModelPOMDP(
        state_schema=SCHEMA,
        action_presets=list(PRESETS),
        discount_factor=0.99,
        transition=rounds[-1].model,
        transition_channels=ROBOT,
    )
    state = SCHEMA.pack({"robot": np.zeros(3), "hazard_type": np.array([1.0, 0.0])})

    successor = planner_model.sample_next_state(state, PRESETS[0])

    assert successor.shape == (5,)


def test_the_latent_channel_survives_a_step_of_the_learned_model() -> None:
    """Purpose: Validates that the fit cannot move the carried hazard type

    Given: A learned transition installed over the robot block only, and a state
        whose latent reads [1, 0]
    When: Twenty successors are drawn
    Then: Every one still reads [1, 0], so the belief over the latent keeps its
        dispersion instead of being regressed away
    """
    _, rounds = _run_loop(rounds=2)
    planner_model = FactoredIsaacModelPOMDP(
        state_schema=SCHEMA,
        action_presets=list(PRESETS),
        discount_factor=0.99,
        transition=rounds[-1].model,
        transition_channels=ROBOT,
    )
    latent = np.array([1.0, 0.0])
    state = SCHEMA.pack({"robot": np.array([0.4, -0.2, 0.1]), "hazard_type": latent})

    successors = np.atleast_2d(planner_model.sample_next_state(state, PRESETS[1], 20))

    assert np.allclose(SCHEMA.block(successors, "hazard_type"), latent)
    # And the robot block did move, so the check above is not passing vacuously.
    assert not np.allclose(SCHEMA.block(successors, "robot"), state[:3])


def test_more_rounds_do_not_degrade_the_held_out_likelihood() -> None:
    """Purpose: Validates that aggregating rounds improves the fit rather than harming it

    Given: Three rounds against a world the model class can represent exactly
    When: The per-round held-out likelihoods are compared
    Then: The last round is at least as good as the first, which is the minimum a
        loop that refits on strictly more data must deliver
    """
    _, rounds = _run_loop(rounds=3)

    likelihoods = [result.diagnostics["held_out_log_likelihood"] for result in rounds]

    assert all(np.isfinite(value) for value in likelihoods)
    assert likelihoods[-1] >= likelihoods[0] - 0.1


def test_the_drift_ratio_is_near_one_when_the_model_class_matches_the_world() -> None:
    """Purpose: Validates that the overconfidence measure reads sane on an honest model

    Given: A linear fit of a linear world, so the model is right and its noise
        estimate is right
    When: The horizon drift ratio is read
    Then: It is small -- the model's error sits inside its own error bars, which
        is what the measure is calibrated to say
    """
    _, rounds = _run_loop(rounds=2)

    ratio = rounds[-1].diagnostics["horizon_drift_ratio"]

    assert 0.0 < ratio < 3.0
