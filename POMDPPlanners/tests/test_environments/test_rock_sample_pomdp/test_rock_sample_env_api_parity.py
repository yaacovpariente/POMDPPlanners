"""Env-API contract / feature-driven tests for the RockSample POMDP source.

These tests target the public ``Environment`` API surface described in
:mod:`POMDPPlanners.core.environment` and probe correctness rather than
implementation parity with the native kernels. Specifically they assert
the contractual relationship between

* :meth:`RockSamplePOMDP.observation_log_probability` (scalar path), and
* :meth:`RockSamplePOMDP.observation_log_probability_per_state` (batch path),

which the base-class default implementation defines as a per-state
vectorisation of the scalar method (see ``Environment.observation_log_probability_per_state``
in ``core/environment.py``). The two paths share a single underlying
likelihood model and must therefore produce identical answers.

They also exercise feature-region coverage:

* sample/PDF consistency (Wilson 3/sqrt(N) at N=5_000) for the noisy
  check-action sensor;
* reward-formula coverage at each branch (sample-good, sample-bad,
  sensor-use, exit, dangerous-area);
* terminal-sentinel boundary handling under the env API;
* batch transition reward parity for movement / sample / exit.
"""

# pylint: disable=duplicate-code

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import (
    RockSamplePOMDP,
    create_rock_sample_state,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_env(
    *,
    map_size: Tuple[int, int] = (5, 5),
    rock_positions: List[Tuple[int, int]] | None = None,
    sensor_efficiency: float = 10.0,
    init_pos: Tuple[int, int] = (0, 0),
) -> RockSamplePOMDP:
    rocks = rock_positions if rock_positions is not None else [(1, 1), (3, 3)]
    return RockSamplePOMDP(
        map_size=map_size,
        rock_positions=rocks,
        init_pos=init_pos,
        sensor_efficiency=sensor_efficiency,
    )


# ---------------------------------------------------------------------------
# Env-API parity: scalar observation_log_probability vs batch
#                 observation_log_probability_per_state
# ---------------------------------------------------------------------------


def test_obs_log_prob_scalar_matches_batch_check_good_live_particle() -> None:
    """Scalar and batch obs-log-prob agree for a check action on a live particle.

    Purpose: Validates env-API contract that ``observation_log_probability_per_state``
        is a per-state vectorisation of ``observation_log_probability`` (per the
        base ``Environment`` default implementation).

    Given: A live particle at the same cell as a good rock, a check action, and
        the observation 'good' (which is the most likely observation here).
    When: The scalar path and the batch path are evaluated on identical inputs.
    Then: Both return the same finite log-probability within atol=1e-12.

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1)], sensor_efficiency=10.0)
    state = create_rock_sample_state((1, 1), (True,))
    states = state.reshape(1, -1)

    scalar = env.observation_log_probability(state, 5, ["good"])[0]
    batch = env.observation_log_probability_per_state(states, 5, "good")[0]

    np.testing.assert_allclose(batch, scalar, atol=1e-12, rtol=0.0)


def test_obs_log_prob_scalar_matches_batch_check_obs_none() -> None:
    """Scalar and batch agree when the observation is impossible (check + 'none').

    Purpose: Surface any silent floor / -inf disagreement between the scalar and
        batch obs-log-probability paths for the (check_action, obs='none') case,
        which is impossible under the sensor model (a valid check action always
        emits 'good' or 'bad', never 'none').

    Given: A live particle at a rock and a valid check action with obs='none'.
    When: Both paths are evaluated.
    Then: Both return the same value (whether finite floor or -inf is fine, but
        they must agree because they wrap the same kernel).

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1)], sensor_efficiency=10.0)
    state = create_rock_sample_state((1, 1), (True,))
    states = state.reshape(1, -1)

    scalar = env.observation_log_probability(state, 5, ["none"])[0]
    batch = env.observation_log_probability_per_state(states, 5, "none")[0]

    if np.isfinite(scalar) and np.isfinite(batch):
        np.testing.assert_allclose(batch, scalar, atol=1e-9, rtol=0.0)
    else:
        assert scalar == batch, (
            f"scalar and batch disagree on (check, obs=none): " f"scalar={scalar}, batch={batch}"
        )


def test_obs_log_prob_scalar_matches_batch_movement_action_obs_good() -> None:
    """Scalar and batch agree for impossible (movement_action, obs='good').

    Purpose: Movement actions deterministically emit 'none'; any non-'none' obs
        must produce the same log-probability via both API entry points.

    Given: A live particle and a movement action (north) with obs='good'.
    When: Both paths are evaluated.
    Then: Both return the same value.

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1)])
    state = create_rock_sample_state((2, 2), (True,))
    states = state.reshape(1, -1)

    scalar = env.observation_log_probability(state, 1, ["good"])[0]
    batch = env.observation_log_probability_per_state(states, 1, "good")[0]

    if np.isfinite(scalar) and np.isfinite(batch):
        np.testing.assert_allclose(batch, scalar, atol=1e-9, rtol=0.0)
    else:
        assert scalar == batch, (
            f"scalar and batch disagree on (movement, obs=good): " f"scalar={scalar}, batch={batch}"
        )


def test_obs_log_prob_scalar_matches_batch_terminal_particle_check() -> None:
    """Scalar and batch agree for a terminal particle under a check action.

    Purpose: Terminal particles (sentinel ``[-1, -1, ...]``) are absorbing; the
        same observation likelihood must be reported via both API paths.

    Given: A terminal-sentinel particle, a valid check action, and obs='good'.
    When: Both paths are evaluated.
    Then: Both return the same value.

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1)])
    terminal = create_rock_sample_state((-1, -1), (True,))
    terminals = terminal.reshape(1, -1)

    scalar = env.observation_log_probability(terminal, 5, ["good"])[0]
    batch = env.observation_log_probability_per_state(terminals, 5, "good")[0]

    if np.isfinite(scalar) and np.isfinite(batch):
        np.testing.assert_allclose(batch, scalar, atol=1e-9, rtol=0.0)
    else:
        assert scalar == batch, (
            f"scalar and batch disagree on terminal+check: " f"scalar={scalar}, batch={batch}"
        )


def test_obs_log_prob_scalar_matches_batch_invalid_check_obs_good() -> None:
    """Scalar and batch agree for an invalid-check action (rock_idx out of range).

    Purpose: Actions whose ``rock_idx >= num_rocks`` are treated as
        deterministic 'none' emitters; both API paths must report the same
        log-probability for any obs under such actions.

    Given: A 1-rock environment, action=6 (invalid 'check rock 1'), obs='good'.
    When: Both paths are evaluated.
    Then: Both return the same value.

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1)])
    state = create_rock_sample_state((1, 1), (True,))
    states = state.reshape(1, -1)

    scalar = env.observation_log_probability(state, 6, ["good"])[0]
    batch = env.observation_log_probability_per_state(states, 6, "good")[0]

    if np.isfinite(scalar) and np.isfinite(batch):
        np.testing.assert_allclose(batch, scalar, atol=1e-9, rtol=0.0)
    else:
        assert scalar == batch, (
            f"scalar and batch disagree on invalid-check + obs=good: "
            f"scalar={scalar}, batch={batch}"
        )


def test_obs_log_prob_random_batch_matches_scalar_per_state() -> None:
    """Random batch of next-states matches per-state scalar evaluation.

    Purpose: Stress-test the env-API contract on a random batch covering live
        particles at varied (row, col, rocks) configurations under a check
        action and the most-likely observation.

    Given: 64 random live next-particles, a valid check action, obs='good'.
    When: ``observation_log_probability_per_state`` is compared element-wise to
        a per-row loop over ``observation_log_probability``.
    Then: Differences are within atol=1e-10.

    Test type: integration
    """
    env = _make_env(rock_positions=[(1, 1), (3, 3), (4, 4)], sensor_efficiency=8.0)
    rng = np.random.default_rng(20260101)
    n_rocks = len(env.rock_positions)
    state_dim = 2 + n_rocks
    particles = np.zeros((64, state_dim), dtype=np.float64)
    for i in range(64):
        particles[i, 0] = int(rng.integers(0, env.map_size[0]))
        particles[i, 1] = int(rng.integers(0, env.map_size[1]))
        particles[i, 2:] = rng.integers(0, 2, size=n_rocks).astype(float)

    batch = env.observation_log_probability_per_state(particles, 5, "good")
    scalar = np.array(
        [env.observation_log_probability(row, 5, ["good"])[0] for row in particles],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batch, scalar, atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# Sample / PDF consistency for the noisy check-action sensor
# ---------------------------------------------------------------------------


def test_check_action_sample_frequency_matches_log_probability() -> None:
    """Empirical 'good'/'bad' frequencies match analytic likelihood under check.

    Purpose: Validates that the C++ observation sampler is calibrated to the
        same Bernoulli(p) it reports via ``observation_log_probability``.

    Given: A bad rock at distance sqrt(2) with sensor_efficiency=2.0; the
        analytic P(obs='bad' | bad rock) = exp(-sqrt(2)/2) ≈ 0.493.
    When: 5_000 observations are drawn under check action 5.
    Then: Empirical frequency of 'bad' is within 3/sqrt(5000) ≈ 0.0424 of
        analytic, matching Wilson-style sample tolerance.

    Test type: integration
    """
    np.random.seed(20260102)
    env = _make_env(rock_positions=[(2, 2)], sensor_efficiency=2.0)
    state = create_rock_sample_state((1, 1), (False,))
    log_probs = env.observation_log_probability(state, 5, ["good", "bad", "none"])
    p_good_analytic = float(np.exp(log_probs[0]))
    p_bad_analytic = float(np.exp(log_probs[1]))

    n_samples = 5000
    obs_list = env.sample_observation(next_state=state, action=5, n_samples=n_samples)
    p_good_emp = sum(1 for o in obs_list if o == "good") / n_samples
    p_bad_emp = sum(1 for o in obs_list if o == "bad") / n_samples

    tol = 3.0 / np.sqrt(n_samples)  # ~0.0424
    assert (
        abs(p_good_emp - p_good_analytic) < tol
    ), f"good empirical={p_good_emp} analytic={p_good_analytic} tol={tol}"
    assert (
        abs(p_bad_emp - p_bad_analytic) < tol
    ), f"bad empirical={p_bad_emp} analytic={p_bad_analytic} tol={tol}"


def test_check_action_log_probability_normalizes_to_one() -> None:
    """Check-action obs probabilities over {good, bad} sum to 1.

    Purpose: Validates the sensor model is a proper Bernoulli over the two
        valid check-action observations ('good' and 'bad'); 'none' must have
        effectively zero mass.

    Given: A live particle and a valid check action.
    When: ``observation_log_probability`` is evaluated for all three observations.
    Then: exp(log_p[good]) + exp(log_p[bad]) ≈ 1 within 1e-9; exp(log_p[none])
        is below the 1e-300 epsilon floor (≤ 1e-200).

    Test type: unit
    """
    env = _make_env(rock_positions=[(2, 2)], sensor_efficiency=5.0)
    state = create_rock_sample_state((0, 0), (True,))
    log_probs = env.observation_log_probability(state, 5, ["good", "bad", "none"])
    probs = np.exp(log_probs)
    assert abs(probs[0] + probs[1] - 1.0) < 1e-9
    assert probs[2] < 1e-200


# ---------------------------------------------------------------------------
# Reward-branch coverage
# ---------------------------------------------------------------------------


def test_reward_dangerous_area_penalty_added_when_next_state_in_zone() -> None:
    """Dangerous-area penalty is added when the *next* state lies in the zone.

    Purpose: Validates the reward branch tied to the dangerous-area predicate
        in ``_reward_from_next_state``. The penalty is keyed off the next
        state's robot position, not the current state's.

    Given: A robot at (2, 0) moving east into (2, 1) which is the centre of a
        dangerous area with radius 1.0 and penalty -5.0; step_penalty=0.
    When: ``reward(state, action=2)`` is evaluated.
    Then: Returned reward equals the dangerous-area penalty (-5.0).

    Test type: unit
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0)],
        init_pos=(0, 0),
        dangerous_areas=[(2, 1)],
        dangerous_area_radius=1.0,
        dangerous_area_penalty=-5.0,
        step_penalty=0.0,
    )
    state = create_rock_sample_state((2, 0), (True,))
    reward = env.reward(state, 2)  # east -> next at (2,1)
    assert reward == pytest.approx(-5.0)


def test_reward_check_action_includes_sensor_penalty_no_movement_penalty() -> None:
    """Check action incurs step_penalty + sensor_use_penalty only.

    Purpose: Validates the action>=5 reward branch in ``_reward_from_next_state``
        contributes ``sensor_use_penalty`` and that a check action does *not*
        trigger sample/exit branches.

    Given: A robot away from any rock with a check action.
    When: ``reward`` is computed with step_penalty=-0.25, sensor_use_penalty=-0.5.
    Then: Returned reward equals -0.25 + -0.5 = -0.75 exactly.

    Test type: unit
    """
    env = RockSamplePOMDP(
        rock_positions=[(0, 0)],
        init_pos=(0, 0),
        step_penalty=-0.25,
        sensor_use_penalty=-0.5,
    )
    state = create_rock_sample_state((2, 2), (True,))
    reward = env.reward(state, 5)  # check_rock_0
    assert reward == pytest.approx(-0.75)


def test_default_dangerous_area_penalty_is_non_positive() -> None:
    """Default dangerous_area_penalty must be ≤ 0 (additive convention).

    Purpose: Locks in the additive-reward sign convention for
        ``dangerous_area_penalty``. The kwarg is added to ``total_reward``
        in ``_reward_from_next_state`` and ``_reward_batch_vectorized``,
        so a positive default would silently *reward* the agent for
        entering danger — the opposite of the documented intent.

    Given: A RockSamplePOMDP constructed with no overrides.
    When: We inspect ``env.dangerous_area_penalty``.
    Then: The default is non-positive (≤ 0).

    Test type: unit
    """
    env = RockSamplePOMDP(rock_positions=[(0, 0)])
    assert env.dangerous_area_penalty <= 0.0, (
        "Default dangerous_area_penalty must be non-positive given the "
        "additive reward convention; a positive default rewards danger entry."
    )


def test_negative_dangerous_area_penalty_decreases_reward() -> None:
    """Passing a negative penalty subtracts from reward on danger entry.

    Purpose: Validates the additive sign convention end-to-end: a negative
        ``dangerous_area_penalty`` of ``-7.0`` must reduce the reward by
        exactly 7 when the next state lies inside a dangerous area.

    Given: A robot at (2, 0) moving east into (2, 1), which is the centre
        of a deterministic dangerous area; ``dangerous_area_penalty=-7.0``,
        ``step_penalty=0``.
    When: ``reward(state, action=east)`` is evaluated.
    Then: Returned reward equals -7.0.

    Test type: unit
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0)],
        init_pos=(0, 0),
        dangerous_areas=[(2, 1)],
        dangerous_area_radius=1.0,
        dangerous_area_penalty=-7.0,
        step_penalty=0.0,
    )
    state = create_rock_sample_state((2, 0), (True,))
    reward = env.reward(state, 2)
    assert reward == pytest.approx(-7.0)


def test_positive_dangerous_area_penalty_emits_warning() -> None:
    """Positive ``dangerous_area_penalty`` raises a UserWarning at construction.

    Purpose: Catches callers still on the legacy "magnitude" reading. With
        the additive convention, a positive value rewards danger entry, so
        the constructor must emit a UserWarning to flag the likely sign
        mistake without breaking the API.

    Given: Constructor kwargs include ``dangerous_area_penalty=5.0`` (positive).
    When: We instantiate the env inside ``pytest.warns(UserWarning)``.
    Then: A UserWarning fires whose message mentions the positive value.

    Test type: unit
    """
    with pytest.warns(UserWarning, match="positive"):
        RockSamplePOMDP(
            map_size=(5, 5),
            rock_positions=[(0, 0)],
            dangerous_areas=[(2, 2)],
            dangerous_area_penalty=5.0,
        )


def test_reward_exit_action_does_not_add_dangerous_area_penalty() -> None:
    """Exit reward short-circuits before the dangerous-area branch.

    Purpose: Validates the early-return path in ``_reward_from_next_state``
        for action=2 at the rightmost column: only step_penalty + exit_reward
        are returned, regardless of whether the next state (terminal) would
        otherwise count as 'in danger'.

    Given: A robot at the right column with east action; a dangerous area
        positioned such that the (-1, -1) terminal sentinel would *not* be
        treated as in-zone (Euclidean distance is large).
    When: ``reward`` is computed.
    Then: Returned reward equals step_penalty + exit_reward exactly, with no
        dangerous-area contribution.

    Test type: unit
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0)],
        init_pos=(0, 0),
        dangerous_areas=[(0, 0)],
        dangerous_area_radius=0.5,
        dangerous_area_penalty=-99.0,
        step_penalty=-1.0,
        exit_reward=10.0,
    )
    state = create_rock_sample_state((2, 4), (True,))
    reward = env.reward(state, 2)  # east -> exit
    assert reward == pytest.approx(-1.0 + 10.0)


# ---------------------------------------------------------------------------
# Terminal-sentinel boundary cases
# ---------------------------------------------------------------------------


def test_is_terminal_only_at_negative_one_sentinel() -> None:
    """is_terminal returns True only for robot position (-1, -1).

    Purpose: Validates the terminal predicate is exactly the (-1, -1) sentinel
        and not any other 'edge' position (e.g. col == map_cols-1, where the
        robot is *about to* exit but has not yet).

    Given: Several states near the right boundary and the explicit terminal.
    When: ``is_terminal`` is evaluated on each.
    Then: Returns True only for (-1, -1).

    Test type: unit
    """
    env = _make_env(map_size=(5, 5), rock_positions=[(0, 0)])
    edge_state = create_rock_sample_state((2, 4), (True,))
    interior_state = create_rock_sample_state((0, 0), (True,))
    terminal_state = create_rock_sample_state((-1, -1), (True,))
    assert not env.is_terminal(edge_state)
    assert not env.is_terminal(interior_state)
    assert env.is_terminal(terminal_state)


def test_sample_next_state_terminal_is_absorbing() -> None:
    """A terminal-sentinel state remains terminal under any action.

    Purpose: Validates the absorbing property of the terminal sentinel under
        ``sample_next_state``: regardless of which action is taken from
        (-1, -1, ...), the resulting state is still the terminal sentinel.

    Given: A terminal-sentinel state.
    When: ``sample_next_state`` is invoked for each of the basic actions
        (sample, north, east, south, west) plus a check action.
    Then: The returned next state is also a terminal sentinel.

    Test type: unit
    """
    env = _make_env(rock_positions=[(0, 0), (2, 2)])
    terminal = create_rock_sample_state((-1, -1), (True, False))
    for action in (0, 1, 2, 3, 4, 5):
        next_state = env.sample_next_state(state=terminal, action=action)
        assert env.is_terminal(next_state), f"action={action} did not absorb terminal"


# ---------------------------------------------------------------------------
# reward_batch parity for terminal-sentinel particles (exit branch)
# ---------------------------------------------------------------------------


def test_reward_batch_exit_branch_treats_terminal_as_exited() -> None:
    """reward_batch east-action: terminal rows receive exit_reward.

    Purpose: Validates the documented behaviour in
        ``RockSamplePOMDP._reward_batch_vectorized`` that terminal rows are
        merged with rows whose col == map_cols - 1 and both receive
        exit_reward. This ensures the batch path is consistent across the
        live-edge and terminal-sentinel populations.

    Given: A batch of 4 particles: [right-edge live, terminal, interior live,
        right-edge live with different rocks].
    When: ``reward_batch(particles, action=2)`` is invoked.
    Then: Both right-edge rows and the terminal row receive
        step_penalty + exit_reward; the interior row receives only step_penalty.

    Test type: integration
    """
    env = RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0)],
        init_pos=(0, 0),
        step_penalty=-1.0,
        exit_reward=10.0,
    )
    particles = np.array(
        [
            [2.0, 4.0, 1.0],
            [-1.0, -1.0, 1.0],
            [2.0, 2.0, 1.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    rewards = env.reward_batch(particles, 2)
    assert rewards[0] == pytest.approx(-1.0 + 10.0)
    assert rewards[1] == pytest.approx(-1.0 + 10.0)
    assert rewards[2] == pytest.approx(-1.0)
    assert rewards[3] == pytest.approx(-1.0 + 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
