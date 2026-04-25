"""Native↔Python equivalence tests for the PacMan POMDP C++ port.

These tests target the ``PacManTransitionCpp`` C++ kernel and its Python shim
``PacManStateTransitionModel``. The observation model is covered in its own
test module once the corresponding C++ port lands.

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

from POMDPPlanners.core.environment import StateTransitionModel
from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    PacManStateTransitionModel,
)


def _build_env() -> PacManPOMDP:
    """Small reusable env matching the benchmark fixture from PR #87."""
    return PacManPOMDP(
        maze_size=(7, 7),
        num_ghosts=2,
        initial_pellets=[(1, 1), (1, 5), (5, 1), (5, 5)],
        initial_pacman_pos=(3, 3),
        initial_ghost_positions=[(0, 0), (6, 6)],
        ghost_aggressiveness=2.0,
        ghost_coordination="independent",
        discount_factor=0.95,
    )


def _state_key(arr: np.ndarray) -> Tuple[float, ...]:
    return tuple(float(x) for x in arr)


class TestAbcRegistration:
    def test_is_instance_of_state_transition_model(self) -> None:
        """Test PacManStateTransitionModel registers as StateTransitionModel.

        Purpose: Validates the virtual ABC registration (mirrors laser_tag
            pattern) so planners that check ``isinstance(model, StateTransitionModel)``
            continue to work after the C++ port.

        Given: A fresh env and a state transition model instance.
        When: isinstance against StateTransitionModel is checked.
        Then: The instance is recognised.

        Test type: unit
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]
        model = env.state_transition_model(state, 0)
        assert isinstance(model, StateTransitionModel)
        assert isinstance(model, PacManStateTransitionModel)


class TestNativeSampleAgainstBatchSample:
    def test_per_call_matches_batch_under_shared_seed(self) -> None:
        """Test per-particle native sample() and batch_sample() agree row-by-row.

        Purpose: Validates that the single-instance and batch entry points of
            PacManTransitionCpp draw from the same RNG stream in the same
            order, so bearing the same seed they produce identical outputs.

        Given: A seeded native RNG and a batch of particles. The batch contains
            5 copies of the initial state.
        When: batch_sample is called on the 5-row batch, and in a separate
            seeded run sample() is called 5 times in a row on the same state.
        Then: The two sequences of 5 ndarrays are equal row-for-row.

        Test type: integration
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]

        _native.set_seed(123)
        batch_model = PacManStateTransitionModel(state, 1, env)  # East
        batch = np.stack([state] * 5)
        batch_out = batch_model.batch_sample(batch)

        _native.set_seed(123)
        per_call_rows = []
        for _ in range(5):
            model = PacManStateTransitionModel(state, 1, env)
            per_call_rows.append(model.sample()[0])
        per_call_out = np.stack(per_call_rows)

        np.testing.assert_array_equal(batch_out, per_call_out)


class TestTerminalAbsorbing:
    def test_terminal_state_is_absorbing(self) -> None:
        """Test that sampling from a terminal state returns the state unchanged.

        Purpose: Validates the ``if terminal return state`` fast path in
            apply_transition — terminal states are absorbing.

        Given: A terminal state with terminal flag = 1.0.
        When: sample() is called.
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
        model = env.state_transition_model(terminal_state, 0)
        next_state = model.sample()[0]
        np.testing.assert_array_equal(next_state, terminal_state)


class TestPelletCollection:
    def test_moving_onto_pellet_flips_mask_and_increments_score(self) -> None:
        """Test that PacMan moving onto an active pellet collects it.

        Purpose: Validates the collection / score-update branch of the
            transition kernel.

        Given: PacMan at (1, 0) with all 4 pellets active and score 0. Action
            east moves PacMan to (1, 1) which is a registered pellet position.
        When: sample() is called once.
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
        model = env.state_transition_model(state, 1)  # East
        next_state = model.sample()[0]
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

        When: sample() is called up to 50 times with different seeds until a
            transition produces a collision.
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
            model = env.state_transition_model(state, 1)  # East
            next_state = model.sample()[0]
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
        model = env.state_transition_model(state, 1)  # East
        next_state = model.sample()[0]
        assert env.get_pacman_pos(next_state) == (1, 1)
        assert env.get_pellets(next_state) == ()
        assert env.get_terminal(next_state)


class TestAggressiveDistribution:
    def test_aggressive_ghost_empirical_matches_probability(self) -> None:
        """Test empirical sample distribution matches probability() for aggressive ghost.

        Purpose: Validates that the softmax-sampled ghost move under the
            aggressive strategy in C++ produces frequencies that match the
            analytic probability() evaluation for the same transition.

        Given: A 5x5 env with 1 aggressive ghost and no walls; seeded 0.
        When: 20_000 samples are drawn; the empirical per-next-state
            frequency is compared against probability(unique states).
        Then: max |freq - prob| < 0.02 across the support.

        Test type: integration
        """
        env = PacManPOMDP(
            maze_size=(5, 5),
            walls=set(),
            initial_pellets=[(2, 2)],
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(3, 3)],
            ghost_aggressiveness=2.0,
            ghost_coordination="independent",
            ghost_strategies=["aggressive"],
        )
        state = env.make_state(
            pacman_pos=(0, 0),
            ghost_positions=((3, 3),),
            pellets=((2, 2),),
            score=0.0,
            terminal=False,
        )

        n_samples = 20_000
        _native.set_seed(2026)
        samples = []
        for _ in range(n_samples):
            model = env.state_transition_model(state, 1)  # East
            samples.append(_state_key(model.sample()[0]))
        counts = Counter(samples)
        unique_states = [np.array(k) for k in counts.keys()]
        empirical = np.array([counts[_state_key(s)] / n_samples for s in unique_states])

        model = env.state_transition_model(state, 1)
        probs = model.probability(unique_states)
        max_abs = float(np.max(np.abs(empirical - probs)))
        assert max_abs < 0.02, (
            f"empirical vs probability mismatch: max |diff| = {max_abs:.4f};"
            f" empirical={empirical}, probs={probs}"
        )

        # Probabilities should normalize over their support.
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)


class TestSampleNextStateOverrideEquivalence:
    """Hot-path overrides skip the Python wrapper but match it bit-exactly.

    The new ``sample_next_state`` / ``sample_observation`` env-level
    overrides construct the native C++ kernel directly, bypassing the
    per-call ``PacManStateTransitionModel`` / ``PacManObservationModel``
    wrapper allocation. Under a shared C++ RNG seed and the same
    ``ghost_patrol_directions`` buffer state, both paths must produce
    identical outputs.
    """

    def test_sample_next_state_matches_state_transition_model(self) -> None:
        """Override sample_next_state matches state_transition_model.sample().

        Purpose: Validates that ``PacManPOMDP.sample_next_state`` produces
        the exact same next state as the legacy
        ``state_transition_model(state, action).sample()[0]`` path under
        a fixed C++ RNG seed and matching patrol-direction buffer state.

        Given: A small env, a non-terminal state at the initial position,
            and a sweep of all four discrete actions.
        When: Both paths are invoked with ``_native.set_seed`` reset to
            the same value and the env's ``ghost_patrol_directions``
            buffer reset to zeros before each call.
        Then: ``np.array_equal`` holds elementwise on the returned arrays.

        Test type: integration
        """
        env = _build_env()
        state = env.initial_state_dist().sample()[0]

        for action in range(4):
            env.ghost_patrol_directions[:] = 0
            _native.set_seed(2024 + action)
            via_wrapper = env.state_transition_model(state, action).sample()[0]

            env.ghost_patrol_directions[:] = 0
            _native.set_seed(2024 + action)
            via_override = env.sample_next_state(state=state, action=action)

            np.testing.assert_array_equal(via_wrapper, via_override)

    def test_sample_observation_matches_observation_model(self) -> None:
        """Override sample_observation matches observation_model.sample().

        Purpose: Validates that ``PacManPOMDP.sample_observation``
        (which constructs the native observation kernel directly) produces
        the exact same tuple-of-(row, col) observation as the legacy
        ``observation_model(...).sample()[0]`` path under a fixed C++ RNG
        seed.

        Given: A small env and a small set of representative next states
            covering different ghost configurations.
        When: Both paths are invoked with ``_native.set_seed`` reset to
            the same value before each call.
        Then: The two observations are equal.

        Test type: integration
        """
        env = _build_env()
        # Generate a few next-states by stepping the env from the initial
        # state with different actions; this gives realistic non-terminal
        # ghost positions to exercise the noise sampler over.
        initial_state = env.initial_state_dist().sample()[0]
        next_states = [initial_state]
        env.ghost_patrol_directions[:] = 0
        _native.set_seed(0)
        for action in range(4):
            next_states.append(env.state_transition_model(initial_state, action).sample()[0])

        for i, next_state in enumerate(next_states):
            for action in (0, 1, 2, 3):
                _native.set_seed(7777 + i * 11 + action)
                via_wrapper = env.observation_model(next_state, action).sample()[0]

                _native.set_seed(7777 + i * 11 + action)
                via_override = env.sample_observation(next_state=next_state, action=action)

                assert via_wrapper == via_override
