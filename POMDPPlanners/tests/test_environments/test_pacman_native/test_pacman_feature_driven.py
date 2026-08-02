"""Feature-driven tests for the PacMan POMDP environment.

These tests target gaps not covered by ``test_pacman_pomdp.py`` and
``test_pacman_native_equivalence.py``. They are intentionally written from
the published env API so that any divergence between code paths that
*should* describe the same probability model surfaces as a hard test
failure.

The key invariant under test in this file is:

    For any (next_state, action, observation), the scalar API
    ``env.observation_log_probability(next_state, action, [observation])``
    and the batch API
    ``env.observation_log_probability_per_state(np.stack([next_state]),
        action, observation)`` must agree element-wise — they describe
    the *same* observation model.

Light-dark exposed an asymmetric floor on this invariant; the tests below
check the same property for PacMan and surface several other feature
regions (terminal sampling, observation clamping, transition probability
normalization, etc.).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


def _build_small_env(num_ghosts: int = 1) -> PacManPOMDP:
    """Construct a small PacMan env with no walls for predictable transitions."""
    if num_ghosts == 1:
        ghosts: List[Tuple[int, int]] = [(4, 4)]
    else:
        ghosts = [(0, 4), (4, 0)]
    return PacManPOMDP(
        maze_size=(5, 5),
        walls=set(),
        initial_pellets=[(2, 2)],
        initial_pacman_pos=(0, 0),
        num_ghosts=num_ghosts,
        initial_ghost_positions=ghosts,
    )


def _build_low_noise_env() -> PacManPOMDP:
    """Env with very small observation noise to exercise extreme log-PDFs."""
    return PacManPOMDP(
        maze_size=(20, 20),
        walls=set(),
        initial_pellets=[(0, 0)],
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(0, 1)],
        observation_noise_factor=0.01,
        max_observation_noise=0.05,
    )


class TestObservationLogProbScalarBatchParity:
    """Parity invariants between scalar and batch observation log-prob APIs."""

    def test_scalar_and_batch_agree_for_true_observation(self) -> None:
        """Scalar/batch observation log-probabilities agree for the true ghost obs.

        Purpose: Validates that ``observation_log_probability`` and
            ``observation_log_probability_per_state`` describe the same
            per-particle observation model on a finite, well-conditioned input.

        Given: A non-terminal next-state and an observation equal to the true
            ghost position, where the analytic 2-D Gaussian log-PDF is finite.
        When: The scalar API and the batch API are evaluated on the same
            (next_state, action, observation).
        Then: The two log-probabilities are equal to within tight float
            tolerance (the same C++ kernel computes both).

        Test type: unit
        """
        env = _build_small_env()
        next_state = env.make_state(pacman_pos=(2, 2), ghost_positions=((2, 3),), pellets=((2, 2),))
        scalar = env.observation_log_probability(next_state, 0, [((2, 3),)])
        batch = env.observation_log_probability_per_state(np.stack([next_state]), 0, ((2, 3),))
        assert scalar.shape == (1,)
        assert batch.shape == (1,)
        assert np.isclose(scalar[0], batch[0], rtol=1e-12, atol=0.0)

    def test_scalar_and_batch_agree_for_terminal_state_terminal_obs(self) -> None:
        """Scalar/batch agree when both state and observation are terminal sentinels.

        Purpose: Confirms the model assigns log-prob 0 (probability 1) to the
            sentinel terminal observation under a terminal state, on both APIs.

        Given: A terminal next-state (terminal flag set) and the canonical
            terminal observation ``((-1, -1),)``.
        When: Both the scalar and the batch log-prob APIs are evaluated.
        Then: Both return exactly ``log 1 = 0.0``.

        Test type: unit
        """
        env = _build_small_env()
        terminal_state = env.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=True,
        )
        scalar = env.observation_log_probability(terminal_state, 0, [((-1, -1),)])
        batch = env.observation_log_probability_per_state(
            np.stack([terminal_state]), 0, ((-1, -1),)
        )
        assert scalar[0] == 0.0
        assert batch[0] == 0.0

    def test_scalar_and_batch_agree_for_terminal_obs_under_nonterminal_state(self) -> None:
        """Scalar/batch agree on impossible terminal-obs under non-terminal state.

        Purpose: This is the asymmetry that surfaced in the light-dark POMDP:
            the scalar API runs the C++ ``probability`` (which collapses
            non-finite log-PDFs to 0) and then computes ``log(p + 1e-300)``,
            flooring at ``log(1e-300) ≈ -690.78``. The batch API
            (``batch_log_likelihood``) returns the raw ``-inf`` log-PDF.
            The two must agree because they describe the *same* model.

        Given: A non-terminal next-state and the sentinel terminal observation
            ``((-1, -1),)``, which has zero probability under any non-terminal
            state.
        When: Both the scalar and the batch log-prob APIs are evaluated.
        Then: Both return the same value (either both ``-inf`` or both equal
            up to tight floating-point tolerance).

        Test type: unit
        """
        env = _build_small_env()
        non_terminal = env.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 3),), pellets=((2, 2),)
        )
        scalar = env.observation_log_probability(non_terminal, 0, [((-1, -1),)])
        batch = env.observation_log_probability_per_state(np.stack([non_terminal]), 0, ((-1, -1),))
        # If both are -inf, treat them as equal.
        if np.isneginf(scalar[0]) and np.isneginf(batch[0]):
            return
        assert np.isclose(scalar[0], batch[0], rtol=1e-9, atol=1e-9), (
            f"scalar/batch obs log-prob mismatch on terminal-obs vs "
            f"non-terminal-state: scalar={scalar[0]}, batch={batch[0]}"
        )

    def test_scalar_and_batch_agree_for_extreme_log_pdf(self) -> None:
        """Scalar/batch agree even when the analytic log-PDF is extremely negative.

        Purpose: When noise is very small relative to the (ghost, observation)
            offset, the analytic Gaussian log-PDF can be far below
            ``log(1e-300) ≈ -690.78``. The scalar API converts log-PDF to
            probability via ``exp`` (which underflows to 0), then back via
            ``log(p + 1e-300)`` — losing information. The batch API returns
            the raw log-PDF. Both must agree.

        Given: An env with ``observation_noise_factor=0.01`` and
            ``max_observation_noise=0.05``, plus an observation at distance
            (10, 10) from the true ghost. The analytic log-PDF is on the order
            of ``-9e5``, well below the scalar floor of ~-690.
        When: Both APIs are evaluated on the same input.
        Then: Both return the same value to within tight tolerance.

        Test type: unit
        """
        env = _build_low_noise_env()
        next_state = env.make_state(pacman_pos=(0, 0), ghost_positions=((0, 1),), pellets=((0, 0),))
        scalar = env.observation_log_probability(next_state, 0, [((10, 10),)])
        batch = env.observation_log_probability_per_state(np.stack([next_state]), 0, ((10, 10),))
        if np.isneginf(scalar[0]) and np.isneginf(batch[0]):
            return
        assert np.isclose(scalar[0], batch[0], rtol=1e-9, atol=1e-9), (
            f"scalar/batch obs log-prob mismatch on extreme log-PDF: "
            f"scalar={scalar[0]}, batch={batch[0]}"
        )

    def test_scalar_and_batch_agree_for_2d_obs_array_input(self) -> None:
        """Scalar (2-D ndarray) path and batch path agree on terminal-sentinel obs.

        Purpose: ``observation_log_probability`` accepts a 2-D ndarray of
            observations as well as a sequence of tuples. Both call
            ``kernel.probability`` (with the ``log(p + 1e-300)`` floor),
            whereas ``observation_log_probability_per_state`` calls
            ``batch_log_likelihood`` (no floor). These must still agree.

        Given: A non-terminal next-state and a 2-D ndarray containing the
            terminal-sentinel observation ``[-1, -1]``.
        When: The scalar (2-D path) and the batch APIs are evaluated.
        Then: The two log-probabilities are equal (both ``-inf`` or both
            finite and close).

        Test type: unit
        """
        env = _build_small_env()
        non_terminal = env.make_state(
            pacman_pos=(2, 2), ghost_positions=((2, 3),), pellets=((2, 2),)
        )
        obs_array = np.array([[-1.0, -1.0]], dtype=np.float64)
        scalar = env.observation_log_probability(non_terminal, 0, obs_array)
        batch = env.observation_log_probability_per_state(
            np.stack([non_terminal]), 0, obs_array.ravel()
        )
        if np.isneginf(scalar[0]) and np.isneginf(batch[0]):
            return
        assert np.isclose(scalar[0], batch[0], rtol=1e-9, atol=1e-9)


class TestObservationFeatureRegions:
    """Tests for distinct feature regions of the observation model."""

    def test_terminal_state_always_samples_terminal_sentinel(self) -> None:
        """Terminal state -> sample_observation always returns the all-(-1) sentinel.

        Purpose: Validates the contract that terminal next-states emit the
            sentinel observation (per-ghost (-1, -1)), so downstream particle
            filters can treat it as a deterministic absorbing observation.

        Given: A terminal next-state.
        When: ``sample_observation`` is called many times.
        Then: Every sample is ``((-1, -1),)``.

        Test type: unit
        """
        env = _build_small_env()
        terminal_state = env.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=True,
        )
        _native.set_seed(7)
        samples = env.sample_observation(terminal_state, 0, n_samples=128)
        for obs in samples:
            assert obs == ((-1, -1),), f"unexpected terminal sample: {obs}"

    def test_observations_are_clamped_to_maze_bounds(self) -> None:
        """Sampled observations fall inside ``[0, maze_rows-1] × [0, maze_cols-1]``.

        Purpose: Validates the C++ kernel's coordinate clamping prevents
            observations from leaking outside the maze (otherwise downstream
            particle filters would assign zero likelihood unexpectedly).

        Given: A small maze with a corner ghost and large noise (factor
            ``2.0``, max ``10.0``) so unclamped Gaussian draws would routinely
            exit the maze.
        When: Many observations are sampled.
        Then: Every sampled coordinate is inside the maze bounds.

        Test type: unit
        """
        env = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(0, 4)],
            observation_noise_factor=2.0,
            max_observation_noise=10.0,
        )
        next_state = env.make_state(pacman_pos=(0, 0), ghost_positions=((0, 4),), pellets=((2, 2),))
        _native.set_seed(11)
        samples = env.sample_observation(next_state, 0, n_samples=256)
        for obs in samples:
            (row, col) = obs[0]
            assert 0 <= row <= 4, f"row {row} out of bounds"
            assert 0 <= col <= 4, f"col {col} out of bounds"

    def test_wrong_ghost_count_observation_has_zero_probability(self) -> None:
        """Observation tuple with wrong ghost count -> log-prob is ``-inf``.

        Purpose: Validates that an observation whose length disagrees with
            ``num_ghosts`` is impossible under the model. The scalar API
            documents this in code (rows with the wrong ghost count are
            skipped, so probability stays at 0). The model invariant says
            log-prob should be ``-inf`` (probability 0).

        Given: A 2-ghost env and a non-terminal next-state.
        When: The scalar API is given a 1-ghost observation tuple.
        Then: The returned log-probability is ``-inf`` (impossible event).

        Test type: unit
        """
        env = _build_small_env(num_ghosts=2)
        next_state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((0, 4), (4, 0)),
            pellets=((2, 2),),
        )
        scalar = env.observation_log_probability(next_state, 0, [((0, 4),)])
        assert scalar.shape == (1,)
        assert np.isneginf(scalar[0]), (
            f"impossible (wrong-ghost-count) observation should have log-prob "
            f"-inf, got {scalar[0]}"
        )


class TestTransitionFeatureRegions:
    """Tests for distinct feature regions of the transition model."""

    def test_transition_probabilities_normalize_to_one(self) -> None:
        """Transition probabilities sum to ~1 over the full reachable next-state set.

        Purpose: Validates that ``transition_log_probability`` returns a
            properly normalized distribution. Mis-normalization would silently
            bias particle-filter weights.

        Given: A 4×4 maze, single ghost at (2, 2), PacMan at (0, 0), action
            South. PacMan moves deterministically to (1, 0); the ghost moves
            to one of {(1, 2), (2, 1), (2, 3), (3, 2), (2, 2)}.
        When: The five reachable next-states are evaluated.
        Then: Their probabilities (un-floored, finite) sum to 1.0 to within
            tight tolerance.

        Test type: unit
        """
        env = PacManPOMDP(
            maze_size=(4, 4),
            walls=set(),
            initial_pellets=[(0, 0)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(2, 2)],
        )
        state = env.make_state(pacman_pos=(0, 0), ghost_positions=((2, 2),), pellets=((0, 0),))
        candidates = [
            env.make_state(pacman_pos=(1, 0), ghost_positions=((gr, gc),), pellets=((0, 0),))
            for (gr, gc) in [(1, 2), (2, 1), (2, 3), (3, 2), (2, 2)]
        ]
        log_probs = env.transition_log_probability(state, 2, candidates)
        probs = np.exp(log_probs)
        assert np.isclose(probs.sum(), 1.0, atol=1e-9), (
            f"transition probabilities should sum to 1 over reachable "
            f"next-states, got {probs.sum()} (probs={probs})"
        )

    def test_terminal_state_is_absorbing_under_log_probability(self) -> None:
        """From a terminal state, only the identity transition has non-zero prob.

        Purpose: Validates the absorbing-terminal contract on the analytic
            log-prob path (``transition_log_probability``), complementing the
            sample-side test in ``test_pacman_native_equivalence.py``.

        Given: A terminal state ``s`` and any action.
        When: ``transition_log_probability`` is queried at ``s`` itself and at
            a different state ``s'``.
        Then: ``log P(s | s, a) = 0`` (probability 1) and the impossible
            transition has log-prob below the scalar floor (i.e. probability
            collapsed to zero).

        Test type: unit
        """
        env = _build_small_env()
        terminal = env.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=True,
        )
        other = env.make_state(
            pacman_pos=(3, 3),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=False,
        )
        log_probs = env.transition_log_probability(terminal, 1, [terminal, other])
        assert np.isclose(log_probs[0], 0.0, atol=1e-12)
        assert log_probs[1] < -100.0


class TestTerminalAndIsTerminal:
    """Tests for the ``is_terminal`` predicate boundary cases."""

    def test_is_terminal_treats_flag_strictly_above_half(self) -> None:
        """``is_terminal`` requires ``state[idx_terminal] > 0.5``.

        Purpose: Validates the boundary semantics of the terminal predicate so
            that callers can rely on the strict ``> 0.5`` convention used by
            both Python and C++ paths.

        Given: A state whose terminal slot is set to 0.4, 0.5, 0.6, and 1.0.
        When: ``is_terminal`` is called.
        Then: Only states with terminal slot strictly greater than 0.5 are
            classified as terminal.

        Test type: unit
        """
        env = _build_small_env()
        base = env.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=False,
        )
        cases = [(0.0, False), (0.4, False), (0.5, False), (0.6, True), (1.0, True)]
        for value, expected in cases:
            arr = base.copy()
            arr[-1] = value
            assert env.is_terminal(arr) is expected, (
                f"is_terminal({value}) -> got {env.is_terminal(arr)}, " f"expected {expected}"
            )


class TestRewardFeatureRegions:
    """Tests for distinct feature regions of the reward function."""

    def test_terminal_state_reward_is_zero(self) -> None:
        """``reward`` returns 0 for a terminal state regardless of action.

        Purpose: Validates the absorbing-terminal contract on the reward
            function: terminal states should not accrue further reward.

        Given: A terminal state.
        When: ``reward`` is called for every action.
        Then: All rewards equal exactly 0.0.

        Test type: unit
        """
        env = _build_small_env()
        terminal = env.make_state(
            pacman_pos=(2, 2),
            ghost_positions=((4, 4),),
            pellets=((2, 2),),
            terminal=True,
        )
        for action in env.get_actions():
            assert env.reward(terminal, action) == 0.0

    def test_reward_batch_includes_win_bonus_when_last_pellet_eaten(self) -> None:
        """``reward_batch`` adds win bonus when the last pellet is collected.

        Purpose: Validates the deterministic win-bonus branch of
            ``_compute_reward_batch``: when an action moves PacMan onto the
            last active pellet, the batch reward equals
            ``step_penalty + pellet_reward + win_reward``.

        Given: A 5×5 wall-free env with a single pellet at (1, 1), PacMan at
            (0, 1) (one step North of the pellet) and the ghost far away.
        When: ``reward_batch`` is called for action South (which moves PacMan
            onto the pellet).
        Then: The reward equals ``-1 + 10 + 100 = 109`` (no ghost-collision
            term because batch path excludes stochastic components).

        Test type: unit
        """
        env = _build_small_env()
        state = env.make_state(pacman_pos=(0, 1), ghost_positions=((4, 4),), pellets=((2, 2),))
        # Replace the (2, 2) pellet with (1, 1), the only active pellet.
        env_with_pellet = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(1, 1)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
        )
        state = env_with_pellet.make_state(
            pacman_pos=(0, 1), ghost_positions=((4, 4),), pellets=((1, 1),)
        )
        rewards = env_with_pellet.reward_batch(np.stack([state]), 2)  # South
        expected = (
            env_with_pellet.step_penalty
            + env_with_pellet.pellet_reward
            + env_with_pellet.win_reward
        )
        assert np.isclose(
            rewards[0], expected
        ), f"expected win-condition batch reward {expected}, got {rewards[0]}"

    def test_reward_batch_step_penalty_only_when_no_pellet_eaten(self) -> None:
        """``reward_batch`` returns just the step penalty for empty-cell moves.

        Purpose: Validates the simplest reward branch: an action that moves
            PacMan to a non-pellet, non-collision cell receives only the step
            penalty.

        Given: PacMan at (0, 0), pellet at (2, 2), ghost at (4, 4); action
            East moves PacMan to (0, 1) (no pellet, no collision).
        When: ``reward_batch`` is called.
        Then: The reward equals ``step_penalty``.

        Test type: unit
        """
        env = _build_small_env()
        state = env.make_state(pacman_pos=(0, 0), ghost_positions=((4, 4),), pellets=((2, 2),))
        rewards = env.reward_batch(np.stack([state]), 1)  # East
        assert np.isclose(rewards[0], env.step_penalty)


class TestNativeBatchSampleConsistency:
    """Cross-checks between scalar ``sample_next_state`` and ``sample_next_state_batch``."""

    def test_batch_sample_marginal_matches_scalar_sample(self) -> None:
        """Empirical marginal of ``sample_next_state_batch`` matches scalar sampling.

        Purpose: Validates that the batched native sampler does not
            inadvertently couple particles or shift the per-particle
            distribution. Since the only stochastic component (with
            independent ghosts) is the ghost move, the marginal distribution
            of (g_row, g_col) should match between the two APIs at a 3-sigma
            Wilson interval at N = 5000.

        Given: A small env, a single fixed input state, action East.
        When: 5000 samples are drawn through ``sample_next_state`` and 5000
            through ``sample_next_state_batch`` (with the same fixed input
            replicated).
        Then: For every ghost cell, the empirical proportions agree to within
            ``3 / sqrt(N) ≈ 0.042`` (Wilson-style tolerance, well above the
            true 1.96 95% bound).

        Test type: unit
        """
        env = _build_small_env()
        state = env.make_state(pacman_pos=(0, 0), ghost_positions=((4, 4),), pellets=((2, 2),))
        n_samples = 5000
        action = 1  # East

        _native.set_seed(31)
        scalar_samples = [env.sample_next_state(state, action) for _ in range(n_samples)]
        scalar_ghosts = np.array([(int(s[2]), int(s[3])) for s in scalar_samples], dtype=np.int64)

        _native.set_seed(31)
        batch_input = np.stack([state] * n_samples)
        batch_samples = env.sample_next_state_batch(batch_input, action)
        batch_ghosts = np.array(
            [(int(row[2]), int(row[3])) for row in batch_samples], dtype=np.int64
        )

        # Compare per-cell empirical proportions.
        all_cells = set(map(tuple, scalar_ghosts.tolist())) | set(map(tuple, batch_ghosts.tolist()))
        tol = 3.0 / np.sqrt(n_samples)
        for cell in all_cells:
            scalar_p = float(np.mean(np.all(scalar_ghosts == np.array(cell), axis=1)))
            batch_p = float(np.mean(np.all(batch_ghosts == np.array(cell), axis=1)))
            assert abs(scalar_p - batch_p) < tol, (
                f"ghost cell {cell}: scalar={scalar_p:.4f}, batch={batch_p:.4f}, "
                f"diff exceeds 3/sqrt(N) = {tol:.4f}"
            )


class TestSampleObservationEmpiricalVsAnalytic:
    """Empirical observation samples should align with the analytic PDF."""

    def test_marginal_empirical_matches_observation_log_probability(self) -> None:
        """Empirical sample frequencies match ``observation_log_probability``.

        Purpose: Validates the sample/PDF consistency invariant of the
            observation model: for an integer-clamped 2-D Gaussian, summing
            empirical proportions over each unique sampled cell should track
            ``exp(observation_log_probability)`` of the same cell to within a
            Wilson tolerance for the cells with non-trivial mass.

        Given: A 5×5 env with moderate noise (factor 0.5, max 2.0), single
            ghost; 5000 observation samples are drawn.
        When: Empirical proportions are computed for every unique sampled
            observation and compared against the analytic continuous-Gaussian
            PDF returned by the env API.
        Then: For every sampled cell whose empirical proportion exceeds 1%,
            the analytic PDF agrees to within ``3 / sqrt(N) ≈ 0.042``. The
            larger sampled set may include high-PDF cells that did not sample
            this run; we don't enforce match in that direction.

        Test type: unit

        Note: the analytic PDF uses the *continuous* 2-D Gaussian density,
            which is only an approximation of the true (clamped, rounded)
            discrete distribution. We therefore restrict the comparison to
            cells with non-trivial empirical mass and use a generous Wilson
            tolerance.
        """
        env = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(2, 2)],
            observation_noise_factor=0.5,
            max_observation_noise=2.0,
        )
        next_state = env.make_state(pacman_pos=(0, 0), ghost_positions=((2, 2),), pellets=((2, 2),))
        _native.set_seed(77)
        n_samples = 5000
        samples = env.sample_observation(next_state, 0, n_samples=n_samples)

        from collections import Counter  # pylint: disable=import-outside-toplevel

        counts = Counter(samples)
        # Pick cells with at least 1% empirical mass — the PDF approximation
        # of the discretized obs is closest to truth in the high-density
        # central region.
        focused = [obs for obs, c in counts.items() if c / n_samples >= 0.01]
        assert focused, "expected at least one frequently-sampled observation"
        log_probs = env.observation_log_probability(next_state, 0, focused)
        analytic = np.exp(log_probs)
        empirical = np.array([counts[obs] / n_samples for obs in focused])
        # Wilson 3/sqrt(N) tolerance, plus a small absolute slack for the
        # continuous-vs-discrete mismatch.
        tol = 3.0 / np.sqrt(n_samples) + 0.06
        for obs, emp, ana in zip(focused, empirical, analytic):
            assert abs(emp - ana) < tol, (
                f"obs {obs}: empirical={emp:.4f}, analytic_pdf={ana:.4f}, "
                f"diff exceeds tolerance {tol:.4f}"
            )


@pytest.mark.parametrize("num_ghosts", [1, 2])
def test_initial_observation_dist_matches_observation_space(num_ghosts: int) -> None:
    """``initial_observation_dist`` returns an obs of the right shape per ghost count.

    Purpose: Validates that ``initial_observation_dist`` produces an
        observation tuple whose length matches ``num_ghosts``, regardless of
        whether the env has 1 or 2 ghosts. Length mismatches would silently
        misalign downstream particle-filter weight computations.

    Given: A PacMan env with ``num_ghosts ∈ {1, 2}``.
    When: ``initial_observation_dist().sample()`` is called.
    Then: The returned observation is a tuple of length ``num_ghosts``, and
        each element is itself a length-2 tuple of ints inside the maze
        bounds.

    Test type: unit
    """
    env = _build_small_env(num_ghosts=num_ghosts)
    obs = env.initial_observation_dist().sample()[0]
    assert isinstance(obs, tuple)
    assert len(obs) == num_ghosts
    for ghost_obs in obs:
        assert isinstance(ghost_obs, tuple)
        assert len(ghost_obs) == 2
        row, col = ghost_obs
        assert 0 <= row < env.maze_size[0]
        assert 0 <= col < env.maze_size[1]
