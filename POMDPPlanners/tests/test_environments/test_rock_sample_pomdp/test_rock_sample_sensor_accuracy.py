# SPDX-License-Identifier: MIT

"""Regression tests for the RockSample rock-check sensor accuracy law.

RockSample's check sensor used to report a rock's quality correctly with
probability ``exp(-d / sensor_efficiency)``. That expression decays to 0, so it
crosses 0.5 at ``d = sensor_efficiency * ln 2`` — about 6.9 cells at the default
``sensor_efficiency = 10.0`` — and past that point the sensor is *anti*-
informative: a check that returns "bad" is evidence the rock is good, and the
inversion tightens towards certainty as the robot moves further away. On an
11x11 map most rocks sit past that crossover, so a planner could farm reward
from a sensor that is reliably wrong.

Smith & Simmons, "Heuristic Search Value Iteration for POMDPs" (2004), define
the accuracy as ``(1 + 2 ** (-d / d0)) / 2``, which decays from 1 to 0.5 and is
never below 0.5: a far check is uninformative, never misleading.

Every test here computes the expected accuracy from the paper's expression
written out locally, not by calling the environment, so the environment is
pinned against the literature rather than against itself. Both the C++ kernels
and the torch vectorized model are checked, because the law is implemented
separately in each and they must not drift apart.
"""

from typing import List, Tuple

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RockSamplePOMDP,
    create_rock_sample_state,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (  # pylint: disable=line-too-long
    OBS_BAD,
    OBS_GOOD,
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.environments.rock_sample_pomdp.rocksample_vectorized_model import (
    RockSampleVectorizedModel,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import rock_sample_pinned_kwargs

_DEFAULT_EFFICIENCY: float = 10.0

# The old law's crossover, sensor_efficiency * ln 2. Distances past this are
# where the old formula inverted the evidence.
_OLD_CROSSOVER: float = _DEFAULT_EFFICIENCY * float(np.log(2.0))


def _paper_accuracy(distance: float, half_life: float = _DEFAULT_EFFICIENCY) -> float:
    """Smith & Simmons (2004) accuracy, transcribed from the paper."""
    return 0.5 * (1.0 + 2.0 ** (-float(distance) / half_life))


def _make_env(
    rock_positions: List[Tuple[int, int]],
    map_size: Tuple[int, int] = (20, 20),
    sensor_efficiency: float = _DEFAULT_EFFICIENCY,
) -> RockSamplePOMDP:
    return RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(
            map_size=map_size,
            rock_positions=list(rock_positions),
            init_pos=(0, 0),
            sensor_efficiency=sensor_efficiency,
        ),
    )


def _p_correct(env: RockSamplePOMDP, robot: Tuple[int, int], rock_good: bool) -> float:
    """P(the check on rock 0 reports the truth) from the env's own likelihoods."""
    state = create_rock_sample_state(robot, (rock_good,))
    log_probs = env.observation_log_probability(state, 5, ["good", "bad"])
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    return float(probs[0] if rock_good else probs[1])


# ---------------------------------------------------------------------------
# The limits of the law
# ---------------------------------------------------------------------------


def test_accuracy_is_one_at_the_rock_and_decays_to_one_half() -> None:
    """Purpose: Validates the two endpoints of the corrected accuracy law.

    Given: A rock at (0, 0) on a large map, default sensor_efficiency.
    When: The check is evaluated with the robot standing on the rock and
        again from 200 cells away.
    Then: Accuracy is exactly 1.0 at the rock, and from 200 cells it is
        above 0.5 but within 1e-6 of it — the sensor becomes uninformative,
        not wrong.

    Test type: unit
    """
    env = _make_env([(0, 0)], map_size=(220, 220))

    assert _p_correct(env, (0, 0), True) == pytest.approx(1.0, abs=1e-12)
    assert _p_correct(env, (0, 0), False) == pytest.approx(1.0, abs=1e-12)

    far = _p_correct(env, (200, 0), True)
    assert far > 0.5
    assert far - 0.5 < 1e-6


def test_accuracy_decreases_monotonically_and_never_drops_below_one_half() -> None:
    """Purpose: Validates monotone decay with a hard floor at 0.5.

    Given: A rock at (0, 0) and the robot stepping out along the row from
        distance 0 to 60 cells.
    When: P(the check reports the truth) is read at every distance.
    Then: The sequence is strictly decreasing and every value is >= 0.5.

    Test type: unit
    """
    env = _make_env([(0, 0)], map_size=(80, 80))
    values = [_p_correct(env, (d, 0), True) for d in range(0, 61)]

    assert all(v >= 0.5 for v in values), min(values)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


@pytest.mark.parametrize("distance", [8, 9, 11, 15])
def test_far_check_still_favours_the_truth(distance: int) -> None:
    """Purpose: This is the test that would have caught the shipped bug.

    Beyond the old law's crossover at ``sensor_efficiency * ln 2`` (~6.9 cells
    at the default) the old ``exp(-d / sensor_efficiency)`` accuracy fell below
    0.5, so ``P(z = "good" | rock good)`` was below a coin flip and a far check
    was evidence *against* the truth. Under the corrected law it must stay
    strictly above 0.5 at every distance.

    Given: A good rock and a robot 8 to 15 cells away, default efficiency.
    When: P(z = "good" | rock good) is evaluated.
    Then: It is strictly greater than 0.5, and it equals the paper's accuracy.

    Test type: unit
    """
    assert distance > _OLD_CROSSOVER, "distance must be past the old crossover"

    env = _make_env([(0, 0)], map_size=(distance + 5, distance + 5))
    p_good = _p_correct(env, (distance, 0), True)

    assert p_good > 0.5
    assert p_good == pytest.approx(_paper_accuracy(distance), abs=1e-12)
    # The old law's value at this distance, for contrast: below a coin flip.
    assert float(np.exp(-distance / _DEFAULT_EFFICIENCY)) < 0.5


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("half_life", [1.0, 2.0, 10.0, 50.0])
@pytest.mark.parametrize("robot", [(0, 0), (1, 1), (3, 4), (12, 5), (19, 19)])
def test_likelihood_matches_the_paper_closed_form(half_life: float, robot: Tuple[int, int]) -> None:
    """Purpose: Validates the env likelihood against the paper's expression.

    Given: A rock at (0, 0) and a robot at one of several positions, for four
        sensor_efficiency settings.
    When: P(z = "good" | rock good) and P(z = "bad" | rock good) are read.
    Then: They equal ``(1 + 2 ** (-d / d0)) / 2`` and its complement, to
        1e-12, with ``d`` the Euclidean distance.

    Test type: unit
    """
    env = _make_env([(0, 0)], sensor_efficiency=half_life)
    distance = float(np.hypot(robot[0], robot[1]))
    expected = _paper_accuracy(distance, half_life)

    state = create_rock_sample_state(robot, (True,))
    probs = np.exp(np.asarray(env.observation_log_probability(state, 5, ["good", "bad"])))

    np.testing.assert_allclose(probs[0], expected, atol=1e-12)
    np.testing.assert_allclose(probs[1], 1.0 - expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The three implementations agree
# ---------------------------------------------------------------------------


def test_scalar_batched_and_torch_paths_agree() -> None:
    """Purpose: Validates the three copies of the sensor law do not drift.

    The accuracy is written once in the C++ scalar helper, once in the C++
    batched log-likelihood loop, and once in the torch model; nothing else
    forces them to match.

    Given: A batch of states spanning distances 0 to ~27 cells and both rock
        qualities, and both check observations.
    When: The env-API scalar likelihood, the native batched likelihood used by
        the vectorized belief, and the torch model's likelihood are evaluated.
    Then: All three agree with each other and with the paper's accuracy.

    Test type: integration
    """
    env = _make_env([(0, 0)])
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)
    updater = RockSampleVectorizedUpdater.from_environment(env)

    states = np.array(
        [[float(r), float(c), float(q)] for r in (0, 3, 7, 19) for c in (0, 5, 19) for q in (1, 0)],
        dtype=np.float64,
    )

    for obs_str, obs_code in (("good", OBS_GOOD), ("bad", OBS_BAD)):
        scalar = np.array(
            [float(env.observation_log_probability(row, 5, [obs_str])[0]) for row in states]
        )
        batched = np.asarray(
            updater.batch_observation_log_likelihood(states, np.asarray(5), np.asarray(obs_code))
        )
        torch_logp = (
            model.observation_log_probs(
                torch.as_tensor(states, dtype=torch.float64),
                torch.full((states.shape[0],), 5, dtype=torch.int64),
                torch.full((states.shape[0], 1), float(obs_code), dtype=torch.float64),
            )
            .numpy()
            .astype(np.float64)
        )

        np.testing.assert_allclose(batched, scalar, atol=1e-12)
        np.testing.assert_allclose(torch_logp, scalar, atol=1e-12)

        expected = np.array(
            [
                (
                    _paper_accuracy(float(np.hypot(row[0], row[1])))
                    if (row[2] > 0.5) == (obs_code == OBS_GOOD)
                    else 1.0 - _paper_accuracy(float(np.hypot(row[0], row[1])))
                )
                for row in states
            ]
        )
        np.testing.assert_allclose(np.exp(scalar), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The sampler matches the law it reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance", [1, 12])
def test_sampling_frequency_matches_the_accuracy(distance: int) -> None:
    """Purpose: Validates the C++ Bernoulli sampler uses the corrected law.

    A likelihood fixed without fixing the sampler would leave the belief
    update and the episode disagreeing, which no likelihood test can see.

    Given: A good rock, the robot 1 cell away and then 12 cells away (past
        the old crossover), default efficiency, seeded RNG.
    When: 200_000 observations are sampled under the check action.
    Then: The fraction reported "good" matches the paper's accuracy to 5e-3,
        and at 12 cells that fraction is above 0.5.

    Test type: integration
    """
    env = _make_env([(0, 0)], map_size=(distance + 5, distance + 5))
    state = create_rock_sample_state((distance, 0), (True,))

    _native.set_seed(20260912)
    samples = env.sample_observation(next_state=state, action=5, n_samples=200_000)
    fraction_good = sum(1 for s in samples if s == "good") / len(samples)

    np.testing.assert_allclose(fraction_good, _paper_accuracy(distance), atol=5e-3)
    assert fraction_good > 0.5


def test_torch_sampling_frequency_matches_the_accuracy() -> None:
    """Purpose: Validates the torch sampler uses the corrected law too.

    Given: A good rock 12 cells away — past the old crossover — and 200_000
        parallel checks in the torch model.
    When: The fraction of "good" observations is measured.
    Then: It matches the paper's accuracy to 5e-3 and is above 0.5.

    Test type: integration
    """
    torch.manual_seed(20260912)
    env = _make_env([(0, 0)], map_size=(20, 20))
    model = RockSampleVectorizedModel(env, device=torch.device("cpu"), dtype=torch.float64)

    states = torch.tile(torch.tensor([[12.0, 0.0, 1.0]], dtype=torch.float64), (200_000, 1))
    actions = torch.full((200_000,), 5, dtype=torch.int64)
    observations = model.sample_observations(states, actions)
    fraction_good = float((observations[:, 0] == OBS_GOOD).to(torch.float64).mean())

    np.testing.assert_allclose(fraction_good, _paper_accuracy(12.0), atol=5e-3)
    assert fraction_good > 0.5


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


def test_sensor_contract_version_participates_in_the_identities() -> None:
    """Purpose: Validates the observation-law change is visible to the caches.

    The constructor arguments are identical across this fix, so without a
    contract version every cached RockSample episode from before it would be
    silently reused under the new law.

    Given: A default environment and the updater built from it.
    When: The version is bumped by hand and both identities are re-read.
    Then: Both change, and the environment reports version 2.

    Test type: unit
    """
    env = _make_env([(0, 0), (2, 2), (3, 3)], map_size=(5, 5))
    assert env.sensor_contract_version == 2

    env_id_before = env.config_id
    updater_id_before = RockSampleVectorizedUpdater.from_environment(env).config_id

    env.sensor_contract_version += 1

    assert env.config_id != env_id_before
    assert RockSampleVectorizedUpdater.from_environment(env).config_id != updater_id_before
