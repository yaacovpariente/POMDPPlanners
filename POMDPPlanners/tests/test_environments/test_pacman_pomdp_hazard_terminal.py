# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for the PacMan POMDP.

Covers the ``hazard-terminates-episode v2`` redesign for
:class:`PacManPOMDP`. PacMan already carries a terminal state slot
(``_idx_terminal``), so the redesign only changes *when/how* it is set:

* flag-off (default) behaviour is unchanged — the dangerous-area penalty
  stays position-based and the hazard never terminates;
* enabling ``is_dangerous_area_hit_terminal`` makes the transition set the
  (absorbing) terminal slot when PacMan enters a hazard zone
  (deterministically for ``CONSTANT_HAZARD_PENALTY`` since PacMan has no
  explicit hit probability, and with probability ``exp(-min_dist /
  penalty_decay)`` for ``DISTANCE_DECAYED_HAZARD_PENALTY``), and the reward
  penalty becomes deterministic given that slot (no RNG);
* ``is_dangerous_area_hit_terminal=True`` with the zero-mean shock reward
  model raises at construction;
* the native rollout kernel agrees with a Python step-by-step rollout on a
  seeded flag-on trajectory (python <-> C++ parity);
* the terminal slot is absorbing.
"""

from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)


def _danger_env(
    is_dangerous_area_hit_terminal: bool = True,
    reward_model_type: RewardModelType = RewardModelType.CONSTANT_HAZARD_PENALTY,
    **overrides,
) -> PacManPOMDP:
    """PacMan env with a single reachable dangerous zone centred at (3, 3).

    The maze is wall-free so ``(3, 2) --East--> (3, 3)`` lands PacMan on the
    zone centre; the single ghost starts far away at ``(6, 6)`` and the only
    pellet at ``(0, 6)`` so neither collision nor win interferes with the
    hazard-only transitions the tests exercise.
    """
    kwargs: Dict[str, Any] = {
        "maze_size": (7, 7),
        "walls": set(),
        "initial_pellets": [(0, 6)],
        "initial_pacman_pos": (3, 2),
        "num_ghosts": 1,
        "initial_ghost_positions": [(6, 6)],
        "dangerous_areas": {(3, 3)},
        "dangerous_area_radius": 1.0,
        "dangerous_area_penalty": 5.0,
        "reward_model_type": reward_model_type,
    }
    kwargs.update(overrides)
    return PacManPOMDP(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=is_dangerous_area_hit_terminal,
        **kwargs,
    )


def _state(env: PacManPOMDP, pos, terminal: bool = False, score: float = 0.0) -> np.ndarray:
    return env.make_state(
        pacman_pos=pos,
        ghost_positions=((6, 6),),
        pellets=((0, 6),),
        score=score,
        terminal=terminal,
    )


# ---------------------------------------------------------------------------
# (a) flag toggles reward_requires_next_state and leaves flag-off unchanged
# ---------------------------------------------------------------------------


def test_reward_requires_next_state_reflects_flag():
    """``reward_requires_next_state`` mirrors ``is_dangerous_area_hit_terminal``.

    Purpose: Validates the base-hook override that tells drivers to sample the
        transition before computing the (now next-state-dependent) reward.

    Given: One PacMan env with the flag off and one with it on.
    When: ``reward_requires_next_state`` is read on each.
    Then: It is ``False`` for the flag-off env and ``True`` for the flag-on env.

    Test type: unit
    """
    assert _danger_env(is_dangerous_area_hit_terminal=False).reward_requires_next_state is False
    assert _danger_env().reward_requires_next_state is True


def test_flag_off_hazard_does_not_terminate_or_change_reward():
    """Flag-off: entering a zone keeps the legacy position-based penalty contract.

    Purpose: Regression guard that the default (flag-off) env is byte-for-byte
        unchanged — the dangerous area never terminates and the penalty is a
        pure function of position, independent of the terminal slot.

    Given: A flag-off env with a zone at (3, 3) and PacMan at (3, 2).
    When: ``sample_next_state`` moves PacMan east into the zone and ``reward``
        is evaluated on that transition.
    Then: The next state is NOT terminal, yet the position-based penalty still
        applies (``step_penalty - dangerous_area_penalty``).

    Test type: unit
    """
    env = _danger_env(is_dangerous_area_hit_terminal=False)
    _native.set_seed(11)
    state = _state(env, (3, 2))
    next_state = env.sample_next_state(state, action=1)  # East -> (3, 3)
    assert env.get_pacman_pos(next_state) == (3, 3)
    assert env.get_terminal(next_state) is False
    reward = env.reward(state, action=1, next_state=next_state)
    assert reward == env.step_penalty - env.dangerous_area_penalty


# ---------------------------------------------------------------------------
# (b) flag-on: draw-coupled termination + deterministic penalty
# ---------------------------------------------------------------------------


def test_constant_hazard_entering_zone_sets_terminal_and_penalises():
    """Flag-on CONSTANT: zone entry deterministically terminates and penalises.

    Purpose: Validates the prob-1 constant case — PacMan has no explicit
        hit probability, so entering any zone always sets the absorbing
        terminal slot and the reward applies the deterministic penalty.

    Given: A flag-on CONSTANT env with a zone at (3, 3) and PacMan at (3, 2).
    When: ``sample_next_state`` moves PacMan east onto the centre and ``reward``
        is evaluated against the realised next state.
    Then: The next state's terminal slot is set and the reward equals
        ``step_penalty - dangerous_area_penalty``.

    Test type: unit
    """
    env = _danger_env()
    _native.set_seed(1)
    state = _state(env, (3, 2))
    next_state = env.sample_next_state(state, action=1)  # East -> (3, 3)
    assert env.get_pacman_pos(next_state) == (3, 3)
    assert env.get_terminal(next_state) is True
    reward = env.reward(state, action=1, next_state=next_state)
    assert reward == env.step_penalty - env.dangerous_area_penalty


def test_hazard_not_set_and_no_penalty_outside_zone():
    """Flag-on: a step ending outside every zone neither terminates nor penalises.

    Purpose: Validates the symmetric negative case of the constant model.

    Given: A flag-on env and PacMan at (0, 0) moving east to (0, 1), well
        outside the single zone at (3, 3).
    When: ``sample_next_state`` and ``reward`` are evaluated.
    Then: The next state is non-terminal and the reward is only ``step_penalty``.

    Test type: unit
    """
    env = _danger_env()
    _native.set_seed(2)
    state = _state(env, (0, 0))
    next_state = env.sample_next_state(state, action=1)  # East -> (0, 1)
    assert env.get_pacman_pos(next_state) == (0, 1)
    assert env.get_terminal(next_state) is False
    assert env.reward(state, action=1, next_state=next_state) == env.step_penalty


def test_reward_is_deterministic_given_terminal_slot():
    """Flag-on reward reads the terminal slot, not the raw position.

    Purpose: Validates that the flag-on penalty is coupled to the terminal
        slot (set by the transition), so an in-zone-but-non-terminal next state
        carries NO penalty while an in-zone terminal one does — with no RNG.

    Given: A flag-on env, PacMan at (3, 2), and two hand-built next states both
        placing PacMan on the zone centre (3, 3): one with terminal unset and
        one with terminal set.
    When: ``reward`` is evaluated for each (repeatedly, to prove determinism).
    Then: The non-terminal next state yields only ``step_penalty`` and the
        terminal one yields ``step_penalty - dangerous_area_penalty``, every call.

    Test type: unit
    """
    env = _danger_env()
    state = _state(env, (3, 2))
    live_next = _state(env, (3, 3), terminal=False)
    terminal_next = _state(env, (3, 3), terminal=True)
    for _ in range(5):
        assert env.reward(state, action=1, next_state=live_next) == env.step_penalty
        assert (
            env.reward(state, action=1, next_state=terminal_next)
            == env.step_penalty - env.dangerous_area_penalty
        )


def test_decayed_hazard_at_zone_centre_terminates_with_probability_one():
    """Flag-on DECAYED: landing on a zone centre (min_dist 0) always terminates.

    Purpose: Validates the decayed model's prob-1 edge (``exp(0) == 1``) so the
        draw-coupled termination and its deterministic penalty are exercised on
        a deterministic case.

    Given: A flag-on DECAYED env with a zone centre at (3, 3), PacMan at (3, 2).
    When: ``sample_next_state`` moves PacMan onto the centre under several seeds.
    Then: The terminal slot is set every time and the reward carries the penalty.

    Test type: unit
    """
    env = _danger_env(reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY)
    state = _state(env, (3, 2))
    for seed in range(4):
        _native.set_seed(seed)
        next_state = env.sample_next_state(state, action=1)  # East -> (3, 3)
        assert env.get_pacman_pos(next_state) == (3, 3)
        assert env.get_terminal(next_state) is True
        assert (
            env.reward(state, action=1, next_state=next_state)
            == env.step_penalty - env.dangerous_area_penalty
        )


# ---------------------------------------------------------------------------
# (c) shock reward model is incompatible with the flag
# ---------------------------------------------------------------------------


def test_shock_reward_with_flag_raises():
    """``is_dangerous_area_hit_terminal`` + shock model raises at construction.

    Purpose: Validates the guard — the zero-mean shock hazard has no hit
        probability to couple termination to.

    Given: Constructor kwargs enabling the flag with the shock reward model.
    When: ``PacManPOMDP`` is constructed.
    Then: A ``ValueError`` naming the incompatibility is raised.

    Test type: unit
    """
    with pytest.raises(ValueError, match=r"ZERO_MEAN_HAZARD_SHOCK"):
        _danger_env(reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK)


def test_shock_reward_allowed_when_flag_off():
    """The shock model stays usable when the hazard-terminal flag is off.

    Purpose: Validates the guard is narrow — it only rejects the flag-on combo.

    Given: The shock reward model with the flag disabled.
    When: The env is constructed.
    Then: Construction succeeds and the flag attribute is ``False``.

    Test type: unit
    """
    env = _danger_env(
        is_dangerous_area_hit_terminal=False,
        reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
    )
    assert env.is_dangerous_area_hit_terminal is False


# ---------------------------------------------------------------------------
# (d) terminal is absorbing
# ---------------------------------------------------------------------------


def test_terminal_state_is_absorbing():
    """A terminal input freezes to itself with the slot latched at 1.0.

    Purpose: Validates the absorbing contract on the flag-on transition.

    Given: A flag-on env and a terminal state.
    When: A transition is sampled from it.
    Then: The next state equals the input and stays terminal.

    Test type: unit
    """
    env = _danger_env()
    _native.set_seed(3)
    state = _state(env, (3, 3), terminal=True)
    next_state = env.sample_next_state(state, action=1)
    np.testing.assert_array_equal(next_state, state)
    assert env.is_terminal(next_state)


# ---------------------------------------------------------------------------
# (e) python <-> C++ rollout parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reward_model_type",
    [
        RewardModelType.CONSTANT_HAZARD_PENALTY,
        RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
    ],
)
def test_native_rollout_matches_python_rollout_flag_on(reward_model_type):
    """Native ``simulate_rollout`` matches a Python step rollout under one seed.

    Purpose: Validates that the all-C++ flag-on rollout draws the same module
        RNG stream (ghost-move noise then hazard uniform) and applies the same
        deterministic reward as the Python single-step ``sample_next_state`` /
        ``reward`` path.

    Given: A flag-on env (constant and decayed variants), a pre-drawn action-
        index sequence, and a fixed native seed.
    When: ``_native.simulate_rollout`` and an equivalent Python loop are each
        run under the same seed.
    Then: The two discounted returns agree to floating-point tolerance.

    Test type: integration
    """
    env = _danger_env(reward_model_type=reward_model_type)
    start = _state(env, (3, 2))
    max_depth = 10
    discount = 0.95
    action_indices = np.array([1, 3, 1, 0, 2, 1, 3, 1, 0, 2], dtype=np.int32)
    tk = env.get_transition_cpp_ctor_kwargs()

    env.ghost_patrol_directions[:] = 0
    _native.set_seed(4242)
    native_return = _native.simulate_rollout(
        state=start,
        action_indices=action_indices,
        maze_rows=tk["maze_rows"],
        maze_cols=tk["maze_cols"],
        neighbor_table=tk["neighbor_table"],
        neighbor_validity=tk["neighbor_validity"],
        pellet_positions=tk["pellet_positions"],
        ghost_aggressiveness=tk["ghost_aggressiveness"],
        ghost_coordination_code=tk["ghost_coordination_code"],
        ghost_strategy_codes=tk["ghost_strategy_codes"],
        num_ghosts=tk["num_ghosts"],
        num_pellets=tk["num_pellets"],
        pellet_reward=tk["pellet_reward"],
        idx_pac_row=tk["idx_pac_row"],
        idx_pac_col=tk["idx_pac_col"],
        idx_ghosts_start=tk["idx_ghosts_start"],
        idx_pellets_start=tk["idx_pellets_start"],
        idx_pellets_end=tk["idx_pellets_end"],
        idx_score=tk["idx_score"],
        idx_terminal=tk["idx_terminal"],
        patrol_dir_state=env.ghost_patrol_directions,
        ghost_collision_penalty=float(env.ghost_collision_penalty),
        step_penalty=float(env.step_penalty),
        win_reward=float(env.win_reward),
        discount_factor=discount,
        depth=0,
        max_depth=max_depth,
        dangerous_areas=env._dangerous_areas_arr,  # pylint: disable=protected-access
        dangerous_area_radius=float(env.dangerous_area_radius),
        dangerous_area_penalty=float(env.dangerous_area_penalty),
        reward_variant_code=int(tk["reward_variant_code"]),
        penalty_decay=float(env.penalty_decay),
        is_dangerous_area_hit_terminal=True,
    )

    env.ghost_patrol_directions[:] = 0
    _native.set_seed(4242)
    total = 0.0
    gamma = 1.0
    state = start.copy()
    for step in range(max_depth):
        if env.is_terminal(state):
            break
        action = int(action_indices[step])
        next_state = env.sample_next_state(state, action)
        total += gamma * env.reward(state, action, next_state=next_state)
        gamma *= discount
        state = next_state

    np.testing.assert_allclose(native_return, total, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# (f) vectorized belief updater threads the hazard config
# ---------------------------------------------------------------------------


def test_vectorized_updater_batch_transition_sets_terminal_flag_on():
    """The native belief updater's batch transition couples termination too.

    Purpose: Validates that ``PacManVectorizedUpdater`` threads the hazard
        config through to ``batch_sample`` so belief particles entering a zone
        get the absorbing terminal slot.

    Given: A flag-on env and its updater, with a batch of particles at (3, 2).
    When: ``batch_transition`` moves them east into the zone at (3, 3).
    Then: Every resulting particle has the terminal slot set.

    Test type: integration
    """
    env = _danger_env()
    updater = PacManVectorizedUpdater.from_environment(env)
    particles = np.stack([_state(env, (3, 2))] * 6)
    _native.set_seed(9)
    out = updater.batch_transition(particles, action=np.array(1))  # East -> (3, 3)
    idx_terminal = env._idx_terminal  # pylint: disable=protected-access
    assert np.all(out[:, idx_terminal] > 0.5)


def test_vectorized_updater_config_id_distinguishes_hazard_flag():
    """Hazard config changes the updater ``config_id``.

    Purpose: Validates that flag-on and flag-off updaters (which sample
        different transitions) never collide in the result cache.

    Given: Updaters built from a flag-off env and a flag-on env.
    When: Their ``config_id`` values are compared.
    Then: They differ.

    Test type: unit
    """
    off = PacManVectorizedUpdater.from_environment(
        _danger_env(is_dangerous_area_hit_terminal=False)
    )
    on = PacManVectorizedUpdater.from_environment(_danger_env())
    assert off.config_id != on.config_id
