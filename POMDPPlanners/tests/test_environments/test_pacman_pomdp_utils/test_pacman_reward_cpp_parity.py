"""C++↔Python reward parity tests for the PacMan POMDP.

Stage 1 (red step) of the upcoming native reward-kernel work. These
tests assert that ``_native.reward_batch`` — the forthcoming standalone
C++ reward kernel — produces sample means that agree with the Python
reward-model implementations across all three reward variants
(``STANDARD``, ``HIGH_VARIANCE_STATES``, ``DECAYING_HIT_PROBABILITY``).

The tests are expected to fail today: ``_native.reward_batch`` does not
exist yet (reward is currently inlined inside ``_native.simulate_rollout``
and that inlined path omits the ``dangerous_area_penalty`` term
entirely). Stage 2 will add the standalone kernel; these tests pin its
shape and parity contract.
"""

from typing import List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    RewardModelType,
)


# Shared maze / hazard configuration. Mirrors the layout used by
# ``test_pacman_reward_models.py`` so the C++ kernel sees the same env
# shape the Python kernels are already validated against.
_MAZE_SIZE: Tuple[int, int] = (7, 7)
_DANGER_CENTRE: Tuple[int, int] = (3, 3)
_DANGER_RADIUS = 1.0
_DANGER_PENALTY = 5.0
_STEP_PENALTY = -1.0
_GHOST_COLLISION_PENALTY = -100.0
_PELLET_REWARD = 10.0
_WIN_REWARD = 100.0
_PENALTY_DECAY = 1.5
_N_PARTICLES = 2000

_VARIANT_CODES = {
    RewardModelType.STANDARD: 0,
    RewardModelType.HIGH_VARIANCE_STATES: 1,
    RewardModelType.DECAYING_HIT_PROBABILITY: 2,
}


def _make_env(reward_model_type: RewardModelType) -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=_MAZE_SIZE,
        walls=set(),
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(6, 6)],
        initial_pellets=[(1, 1)],
        dangerous_areas={_DANGER_CENTRE},
        dangerous_area_radius=_DANGER_RADIUS,
        dangerous_area_penalty=_DANGER_PENALTY,
        step_penalty=_STEP_PENALTY,
        ghost_collision_penalty=_GHOST_COLLISION_PENALTY,
        pellet_reward=_PELLET_REWARD,
        win_reward=_WIN_REWARD,
        reward_model_type=reward_model_type,
        penalty_decay=_PENALTY_DECAY,
    )


def _seed_states(env: PacManPOMDP, n: int) -> np.ndarray:
    # Initial-state samples give us valid non-terminal rows that share the
    # canonical layout. The pacman position is overwritten below so part of
    # the batch lands in / near the configured hazard zone — exercising the
    # variant-specific danger contribution rather than only the step penalty.
    states = np.ascontiguousarray(env.initial_state_dist().sample(n), dtype=np.float64)
    _spread_pacman_near_danger(env, states)
    return states


def _spread_pacman_near_danger(env: PacManPOMDP, states: np.ndarray) -> None:
    # About a third of the rows are placed at or inside the hazard zone so
    # the danger contribution actually fires; the rest cover assorted
    # non-danger cells across the maze. Keeps results within walls/bounds
    # for the trivial walls=set() configuration of ``_make_env``.
    n = states.shape[0]
    placements: List[Tuple[int, int]] = [
        _DANGER_CENTRE,
        (3, 2),
        (2, 3),
        (3, 4),
        (4, 3),
        (0, 0),
        (1, 1),
        (5, 5),
        (0, 6),
        (6, 0),
    ]
    for row_idx in range(n):
        pacman_pos = placements[row_idx % len(placements)]
        states[row_idx, env._idx_pac_row] = pacman_pos[0]  # pylint: disable=protected-access
        states[row_idx, env._idx_pac_col] = pacman_pos[1]  # pylint: disable=protected-access


def _per_action_row_groups(num_actions: int, n: int) -> List[np.ndarray]:
    # Stripe the rows across actions so every action appears roughly n / A
    # times and the resulting concatenated batch covers the full action
    # space — the parity contract calls for action variation across the
    # batch even though each reward kernel call uses a single action.
    row_actions = np.arange(n, dtype=np.int64) % num_actions
    return [np.flatnonzero(row_actions == action) for action in range(num_actions)]


def _compute_python_rewards_per_action(
    env: PacManPOMDP,
    states: np.ndarray,
    action: int,
    next_states: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        env.reward_model.compute_reward_batch(states, action, next_states=next_states),
        dtype=np.float64,
    )


def _compute_cpp_rewards_per_action(
    env: PacManPOMDP,
    states: np.ndarray,
    action: int,
    next_states: np.ndarray,
    variant_code: int,
    penalty_decay: float,
) -> np.ndarray:
    # pylint: disable=protected-access
    # ``reward_batch`` is the forthcoming Stage-2 C++ entry point — it does
    # not exist on ``_native`` today, so this test is expected to fail
    # with ``AttributeError`` until that kernel lands. The ``type: ignore``
    # silences the static analyser for the missing-attribute access; we
    # cannot use ``typing.cast`` per project policy.
    return np.asarray(
        _native.reward_batch(  # type: ignore[attr-defined]
            states=states,
            action=int(action),
            next_states=next_states,
            reward_variant_code=int(variant_code),
            penalty_decay=float(penalty_decay),
            dangerous_areas=env._dangerous_areas_arr,
            dangerous_area_radius=float(env.dangerous_area_radius),
            dangerous_area_penalty=float(env.dangerous_area_penalty),
            num_ghosts=int(env.num_ghosts),
            step_penalty=float(env.step_penalty),
            ghost_collision_penalty=float(env.ghost_collision_penalty),
            pellet_reward=float(env.pellet_reward),
            win_reward=float(env.win_reward),
            idx_pac_row=int(env._idx_pac_row),
            idx_pac_col=int(env._idx_pac_col),
            idx_ghosts_start=int(env._idx_ghosts_start),
            idx_pellets_start=int(env._idx_pellets_start),
            idx_pellets_end=int(env._idx_pellets_end),
            idx_score=int(env._idx_score),
            idx_terminal=int(env._idx_terminal),
        ),
        dtype=np.float64,
    )


def _gather_rewards_across_actions(
    env: PacManPOMDP,
    states: np.ndarray,
    variant_code: int,
    penalty_decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_actions = len(env.get_actions())
    row_groups = _per_action_row_groups(num_actions, states.shape[0])
    py_rewards = np.empty(states.shape[0], dtype=np.float64)
    cpp_rewards = np.empty(states.shape[0], dtype=np.float64)
    for action, action_rows in enumerate(row_groups):
        if action_rows.size == 0:
            continue
        action_states = np.ascontiguousarray(states[action_rows], dtype=np.float64)
        action_next_states = env.sample_next_state_batch(action_states, action)
        py_rewards[action_rows] = _compute_python_rewards_per_action(
            env, action_states, action, action_next_states
        )
        cpp_rewards[action_rows] = _compute_cpp_rewards_per_action(
            env,
            action_states,
            action,
            action_next_states,
            variant_code=variant_code,
            penalty_decay=penalty_decay,
        )
    return py_rewards, cpp_rewards


def _sample_mean_parity_holds(py_rewards: np.ndarray, cpp_rewards: np.ndarray) -> bool:
    n = py_rewards.shape[0]
    se = float(py_rewards.std(ddof=0)) / np.sqrt(n)
    tolerance = 3.0 * se + 1e-9
    return abs(float(py_rewards.mean()) - float(cpp_rewards.mean())) < tolerance


class TestPacManRewardCppParity:
    """C++↔Python sample-mean parity across all three reward variants.

    The batch is striped across the five PacMan actions so every action
    contributes roughly ``N/5`` rows to the concatenated reward arrays.
    Per-action sub-batches share their realised ``next_states`` between
    the Python and C++ kernels so the comparison reflects kernel-level
    parity rather than transition-sampling noise.
    """

    @pytest.mark.parametrize(
        "variant",
        [
            RewardModelType.STANDARD,
            RewardModelType.HIGH_VARIANCE_STATES,
            RewardModelType.DECAYING_HIT_PROBABILITY,
        ],
    )
    def test_reward_batch_sample_mean_matches_python(self, variant: RewardModelType):
        """``_native.reward_batch`` sample mean matches Python within 3·SE.

        Purpose: Validates the forthcoming standalone C++ reward kernel
            produces the same expected reward as the Python reward model
            across all variants and a representative mix of (state, action,
            next_state) triples — including dangerous-area transitions
            that exercise the variant-specific danger contribution.

        Given: A ``PacManPOMDP`` configured with the variant under test, a
            non-empty hazard zone, ``np.random.seed(42)``, and ``N=2000``
            (state, action, next_state) triples where states are drawn
            from ``env.initial_state_dist().sample(N)`` with the pacman
            cell spread across in-zone / near-zone / far cells, actions
            vary across the action space, and next_states are produced
            by ``env.sample_next_state_batch(states, action)`` per action.
        When: Python rewards are computed by
            ``env.reward_model.compute_reward_batch(...)`` and C++ rewards
            by ``_native.reward_batch(..., reward_variant_code=..., penalty_decay=...)``.
        Then: ``|py.mean() - cpp.mean()| < 3.0 * (py.std() / sqrt(N)) + 1e-9``.

        Test type: integration
        """
        np.random.seed(42)
        env = _make_env(variant)
        states = _seed_states(env, _N_PARTICLES)
        penalty_decay = (
            env.penalty_decay if variant == RewardModelType.DECAYING_HIT_PROBABILITY else 0.0
        )
        py_rewards, cpp_rewards = _gather_rewards_across_actions(
            env,
            states,
            variant_code=_VARIANT_CODES[variant],
            penalty_decay=penalty_decay,
        )

        assert py_rewards.shape == cpp_rewards.shape == (_N_PARTICLES,)
        assert _sample_mean_parity_holds(py_rewards, cpp_rewards)
