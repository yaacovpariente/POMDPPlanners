# SPDX-License-Identifier: MIT

"""Config-coverage contract for every env-constructed vectorized model.

Each vectorized generative model is a hand-written torch duplicate of a subset
of its scalar environment's configuration space, guarding the rest with
:class:`NotImplementedError`. Nothing previously forced those guards to stay in
sync with the environment's enum config axes, so a config the scalar env fully
supports could be silently unsupported by the vectorized model — exactly the
gap that left LaserTag unable to plan over the ``PURSUE`` opponent policy.

This module closes that gap generically. For every environment whose vectorized
model is built directly from an env instance and whose configuration is
parametrized by :class:`enum.Enum` constructor arguments, it sweeps each enum
axis one member at a time and asserts the vectorized model either supports the
value or declines it with a :class:`NotImplementedError` that is on a small,
reviewed allowlist. See
:mod:`POMDPPlanners.tests.test_environments._vectorized_config_contract` for the
engine and the precise contract.

Scope: this covers the five env-constructed, enum-parametrized models
(LaserTag, continuous LightDark, Push, RockSample, PacMan). Environments whose
scope guards gate on non-enum surfaces (CartPole's integrator string,
MountainCar / Sanity / SafetyAnt action sets, PacMan's string ghost modes,
CARLA's perception models) have no enum config axis to sweep, and IsaacLab's
vectorized model is not env-constructed; those remain covered by their own
per-model tests.
"""

import pytest

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_vectorized_model import (
    LaserTagVectorizedModel,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
    ContinuousLightDarkVectorizedModel,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_vectorized_model import (
    PacManVectorizedModel,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_vectorized_model import PushVectorizedModel
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rocksample_vectorized_model import (
    RockSampleVectorizedModel,
)
from POMDPPlanners.tests.test_environments._vectorized_config_contract import (
    VectorizedContractSpec,
    assert_config_contract,
)

# Shared decline reason: only the CONSTANT_HAZARD_PENALTY reward model is
# vectorized; the shock and distance-decayed variants are out of scope.
_REWARD_DECLINES = {
    ("reward_model_type", "ZERO_MEAN_HAZARD_SHOCK"),
    ("reward_model_type", "DISTANCE_DECAYED_HAZARD_PENALTY"),
}


def _make_spec(name, env_class, model_class, base_kwargs, expected_declines):
    def build_env(**overrides):
        return env_class(**{**base_kwargs, **overrides})

    def build_model(env):
        return model_class(env)

    return VectorizedContractSpec(
        name=name,
        env_class=env_class,
        build_env=build_env,
        build_model=build_model,
        expected_declines=expected_declines,
    )


# Each spec pins the reviewed set of enum config values the vectorized model is
# expected to decline. Adding a new enum member to any of these environments
# makes the sweep fail until the member is either implemented or added here as a
# deliberate out-of-scope decision.
_SPECS = [
    _make_spec(
        "laser_tag",
        LaserTagPOMDP,
        LaserTagVectorizedModel,
        {"discount_factor": 0.95},
        # opponent_policy: all of EVADE / PURSUE / EVADE_WHEN_SPOTTED supported.
        set(_REWARD_DECLINES),
    ),
    _make_spec(
        "light_dark",
        ContinuousLightDarkPOMDP,
        ContinuousLightDarkVectorizedModel,
        # is_obstacle_hit_terminal=False isolates the enum axes: the default
        # True is a separate (non-enum) unsupported config.
        {"discount_factor": 0.95, "is_obstacle_hit_terminal": False},
        _REWARD_DECLINES
        | {
            ("observation_model_type", "NORMAL_NOISE_NO_OBS_IN_DARK"),
            ("observation_model_type", "DISTANCE_BASED"),
        },
    ),
    _make_spec(
        "push",
        PushPOMDP,
        PushVectorizedModel,
        {"discount_factor": 0.99},
        set(_REWARD_DECLINES),
    ),
    _make_spec(
        "rocksample",
        RockSamplePOMDP,
        RockSampleVectorizedModel,
        {"discount_factor": 0.95},
        set(_REWARD_DECLINES),
    ),
    _make_spec(
        "pacman",
        PacManPOMDP,
        PacManVectorizedModel,
        {"discount_factor": 0.95},
        set(_REWARD_DECLINES),
    ),
]


@pytest.mark.parametrize("spec", _SPECS, ids=[spec.name for spec in _SPECS])
def test_vectorized_model_supports_or_declines_every_enum_config(
    spec: VectorizedContractSpec,
) -> None:
    """Every enum config value is supported or explicitly, reviewably declined.

    Purpose: Validates that each env-constructed vectorized model stays in sync
        with its scalar environment's enum config axes — no config is silently
        unsupported, and the decline allowlist stays honest

    Given: An environment whose vectorized model is built from an env instance
        and whose configuration is parametrized by Enum constructor arguments
    When: Every member of every Enum-typed constructor parameter is swept and a
        vectorized model is built from it
    Then: The model either constructs (supported) or raises NotImplementedError
        for a value on the reviewed expected_declines allowlist, and no
        allowlisted value builds successfully

    Test type: integration
    """
    summary = assert_config_contract(spec)
    assert summary["supported"] >= 1
    assert summary["declined"] == len(spec.expected_declines)


def test_contract_flags_a_config_dropped_from_the_allowlist() -> None:
    """A newly declined enum value not on the allowlist fails the contract.

    Purpose: Validates the forcing function itself — that dropping a value the
        model declines from expected_declines is caught, so a future
        env-vs-model drift cannot pass silently

    Given: The LaserTag spec with its reward-model declines removed from the
        allowlist while the model still declines them
    When: The contract is evaluated
    Then: An AssertionError names the un-allowlisted declined config

    Test type: unit
    """
    spec = _make_spec(
        "laser_tag_missing_allowlist",
        LaserTagPOMDP,
        LaserTagVectorizedModel,
        {"discount_factor": 0.95},
        set(),
    )
    with pytest.raises(AssertionError, match="not in expected_declines"):
        assert_config_contract(spec)
