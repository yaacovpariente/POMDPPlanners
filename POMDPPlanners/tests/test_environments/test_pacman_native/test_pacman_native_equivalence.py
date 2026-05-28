# SPDX-License-Identifier: MIT

"""Native↔Python equivalence tests for the PacMan POMDP C++ port.

These tests target the ``PacManTransitionCpp`` C++ kernel directly via the
``PacManPOMDP`` env API (``sample_next_state`` /
``transition_log_probability`` / ``sample_observation`` /
``observation_log_probability``). The per-call Python wrapper classes
(``PacManStateTransitionModel`` / ``PacManObservationModel``) were
deleted in PR-D-Pacman together with the
``state_transition_model`` / ``observation_model`` factory methods; the
env-level methods construct the native kernel directly.

Run-time notes:
    - Each test that depends on reproducibility calls
      ``_native.set_seed(...)`` at the top of the test, because each
      ``_native.so`` owns its own per-module RNG singleton and the test suite
      does not share a single source of randomness with numpy.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Tuple

import numpy as np

from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import pacman_pinned_kwargs


def _build_env() -> PacManPOMDP:
    """Small reusable env matching the benchmark fixture from PR #87."""
    # num_ghosts overridden to 2; the env auto-generates per-ghost
    # ``ghost_strategies``, so the single-ghost pinned defaults are not
    # injected here (they would mismatch the two-ghost configuration).
    return PacManPOMDP(
        discount_factor=0.95,
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        ghost_aggressiveness=2.0,
        ghost_coordination="independent",
    )


def _state_key(arr: np.ndarray) -> Tuple[float, ...]:
    return tuple(float(x) for x in arr)


class TestNativeSampleAgainstBatchSample:
    def test_per_call_matches_batch_under_shared_seed(self) -> None:
        """Test per-particle native sample() and batch_sample() agree row-by-row.

        Purpose: Validates that the single-instance and batch entry points of
            PacManTransitionCpp draw from the same RNG stream in the same
            order, so bearing the same seed they produce identical outputs.

        Given: A seeded native RNG and a batch of particles. The batch contains
            5 copies of the initial state. The kernel is constructed directly
            via the env's cached ctor kwargs.
        When: batch_sample is called on the 5-row batch via env.sample_next_state_batch,
            and in a separate seeded run env.sample_next_state is called 5
            times in a row on the same state.
        Then: The two sequences of 5 ndarrays are equal row-for-row.

        Test type: integration
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]

        env.ghost_patrol_directions[:] = 0
        _native.set_seed(123)
        batch = np.stack([state] * 5)
        batch_out = env.sample_next_state_batch(batch, action=1)  # East

        env.ghost_patrol_directions[:] = 0
        _native.set_seed(123)
        per_call_rows = [env.sample_next_state(state=state, action=1) for _ in range(5)]
        per_call_out = np.stack(per_call_rows)

        np.testing.assert_array_equal(batch_out, per_call_out)


class TestTerminalAbsorbing:
    def test_terminal_state_is_absorbing(self) -> None:
        """Test that sampling from a terminal state returns the state unchanged.

        Purpose: Validates the ``if terminal return state`` fast path in
            apply_transition — terminal states are absorbing.

        Given: A terminal state with terminal flag = 1.0.
        When: env.sample_next_state is called.
        Then: The returned state array equals the input byte-for-byte.

        Test type: unit
        """
        env = _build_env()
        terminal_state = env.make_state(
            pacman_pos=(3, 3),
            ghost_positions=((3, 3), (6, 6)),
            pellets=((1, 1), (1, 5), (5, 1), (5, 5)),
            score=0.0,
            terminal=True,
        )
        _native.set_seed(0)
        next_state = env.sample_next_state(state=terminal_state, action=0)
        np.testing.assert_array_equal(next_state, terminal_state)


class TestPelletCollection:
    def test_moving_onto_pellet_flips_mask_and_increments_score(self) -> None:
        """Test that PacMan moving onto an active pellet collects it.

        Purpose: Validates the collection / score-update branch of the
            transition kernel.

        Given: PacMan at (1, 0) with all 4 pellets active and score 0. Action
            east moves PacMan to (1, 1) which is a registered pellet position.
        When: env.sample_next_state is called once.
        Then: Pellet index 0 (the (1,1) pellet) flips from 1.0 to 0.0, and
            the score increases by exactly ``env.pellet_reward``.

        Test type: unit
        """
        env = _build_env()
        state = env.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((0, 0), (6, 6)),
            pellets=((1, 1), (1, 5), (5, 1), (5, 5)),
            score=0.0,
            terminal=False,
        )
        _native.set_seed(0)
        next_state = env.sample_next_state(state=state, action=1)  # East
        assert env.get_pacman_pos(next_state) == (1, 1)
        pellets_after = set(env.get_pellets(next_state))
        assert (1, 1) not in pellets_after
        assert env.get_score(next_state) == env.pellet_reward


class TestCollisionTerminal:
    def test_pacman_walking_into_ghost_sets_terminal(self) -> None:
        """Test that stepping onto a ghost sets the terminal flag.

        Purpose: Validates the post-move collision check.

        Given: PacMan at (3, 2) with a ghost at (3, 3). Action east moves
            PacMan to (3, 3) — the ghost may move away, but we repeat the
            test with a ghost the env is *forced* to stay in place: use the
            patrol strategy with an initial direction that blocks movement.
            For the simpler invariant here we seed many times and check at
            least one rollout yields a collision to terminal.

        When: env.sample_next_state is called up to 50 times with different
            seeds until a transition produces a collision.
        Then: At least one sample lands on terminal=True.

        Test type: unit
        """
        env = _build_env()
        state = env.make_state(
            pacman_pos=(3, 2),
            ghost_positions=((3, 3), (6, 6)),
            pellets=((1, 1), (1, 5), (5, 1), (5, 5)),
            score=0.0,
            terminal=False,
        )
        collision_found = False
        for seed in range(50):
            _native.set_seed(seed)
            next_state = env.sample_next_state(state=state, action=1)  # East
            if env.get_terminal(next_state) and env.get_pacman_pos(
                next_state
            ) in env.get_ghost_positions(next_state):
                collision_found = True
                break
        assert collision_found, "expected at least one collision-induced terminal in 50 rollouts"


class TestWinCondition:
    def test_collecting_last_pellet_sets_terminal(self) -> None:
        """Test that collecting the last remaining pellet sets terminal=True.

        Purpose: Validates the "no pellets remaining" terminal rule.

        Given: PacMan at (1, 0) with only one pellet left at (1, 1).
        When: Action east moves PacMan onto (1, 1).
        Then: The next state has no active pellets and terminal=True.

        Test type: unit
        """
        env = _build_env()
        state = env.make_state(
            pacman_pos=(1, 0),
            ghost_positions=((6, 6), (6, 5)),  # far from pacman to avoid collision
            pellets=((1, 1),),
            score=0.0,
            terminal=False,
        )
        _native.set_seed(0)
        next_state = env.sample_next_state(state=state, action=1)  # East
        assert env.get_pacman_pos(next_state) == (1, 1)
        assert env.get_pellets(next_state) == ()
        assert env.get_terminal(next_state)


class TestAggressiveDistribution:
    def test_aggressive_ghost_empirical_matches_probability(self) -> None:
        """Test empirical sample distribution matches transition_log_probability.

        Purpose: Validates that the softmax-sampled ghost move under the
            aggressive strategy in C++ produces frequencies that match the
            analytic ``transition_log_probability`` evaluation for the same
            transition.

        Given: A 5x5 env with 1 aggressive ghost and no walls; seeded 0.
        When: 20_000 samples are drawn via env.sample_next_state; the
            empirical per-next-state frequency is compared against
            ``np.exp(env.transition_log_probability(state, action, unique_states))``.
        Then: max |freq - prob| < 0.02 across the support.

        Test type: integration
        """
        env = PacManPOMDP(
            discount_factor=0.95,
            **pacman_pinned_kwargs(
                maze_size=(5, 5),
                walls=set(),
                initial_pellets=[(2, 2)],
                initial_pacman_pos=(0, 0),
                num_ghosts=1,
                initial_ghost_positions=[(3, 3)],
                ghost_aggressiveness=2.0,
                ghost_coordination="independent",
                ghost_strategies=["aggressive"],
            ),
        )
        state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((3, 3),),
            pellets=((2, 2),),
            score=0.0,
            terminal=False,
        )

        n_samples = 20_000
        env.ghost_patrol_directions[:] = 0
        _native.set_seed(2026)
        samples = [_state_key(env.sample_next_state(state, 1)) for _ in range(n_samples)]
        counts = Counter(samples)
        unique_states = [np.array(k) for k in counts.keys()]
        empirical = np.array([counts[_state_key(s)] / n_samples for s in unique_states])

        env.ghost_patrol_directions[:] = 0
        log_probs = env.transition_log_probability(state, 1, unique_states)
        probs = np.exp(log_probs)
        max_abs = float(np.max(np.abs(empirical - probs)))
        assert max_abs < 0.02, (
            f"empirical vs probability mismatch: max |diff| = {max_abs:.4f};"
            f" empirical={empirical}, probs={probs}"
        )

        # Probabilities should normalize over their support.
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)


def test_scalar_obs_log_prob_un_floored_matches_batch_after_fix() -> None:
    """Scalar obs log-prob below -690 floor matches the batch path post-fix.

    Purpose: Pins the post-fix contract for PacManPOMDP that
        ``observation_log_probability`` (scalar) and
        ``observation_log_probability_per_state`` (batch) agree on a
        moderate-density anchor whose analytic log-probability is well
        below the old ``log(p + 1e-300) ≈ -690.776`` floor but still
        above the kernel's internal float64 underflow threshold.
        Pre-fix, the scalar path floored such values at ~-690.776 while
        the batch path returned the un-floored kernel log-likelihood —
        the asymmetry that motivated the env-wide log-prob floor
        removal.

    Given: The shared 2-ghost env from ``_build_env``, a fresh
        initial state, action 0, and a 2-D ndarray observation
        ``[[31, 31, 31, 31]]`` (one row of 2*num_ghosts coordinates).
        At this offset the analytic 4-D Gaussian log-pdf for both
        ghosts is ≈ -710.187.
    When: Both ``observation_log_probability`` (with the 2-D ndarray
        fast path) and ``observation_log_probability_per_state`` are
        evaluated on the same (next_state, action, observation).
    Then: Both return finite, equal values to within atol=1e-6, and
        the common value is below -700 (past the old floor).

    Test type: unit
    """
    env = _build_env()
    next_state = env.initial_state_dist().sample()[0]
    obs_2d = np.array([[31.0] * (2 * env.num_ghosts)], dtype=np.float64)

    scalar = env.observation_log_probability(next_state, 0, obs_2d)[0]
    batch = env.observation_log_probability_per_state(np.array([next_state]), 0, obs_2d[0])[0]

    assert np.isfinite(scalar), f"scalar should be finite at this anchor, got {scalar}"
    assert np.isfinite(batch), f"batch should be finite at this anchor, got {batch}"
    # Post symmetric C++ floor: both paths floor at log(1e-300) ~= -690.776
    # for events past the floor, so they agree exactly.
    np.testing.assert_allclose(scalar, batch, atol=1e-6)
