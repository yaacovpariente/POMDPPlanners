# SPDX-License-Identifier: MIT

"""Tests for ``ended_in_danger_zone``, the trace field the viewers smoke on.

Each environment that can end an episode in a danger zone shares its terminal
slot with at least one other ending, so every exporter has its own rule for
telling them apart. Each rule is checked here on a hand-built episode for both
sides: the hazard ending, and the other ending it must not be confused with.
"""

from typing import Any, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments import (
    ContinuousLightDarkPOMDP,
    DiscreteLightDarkPOMDP,
    LaserTagPOMDP,
    PacManPOMDP,
    RockSamplePOMDP,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import RewardModelType
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import ContinuousPushPOMDP


def _history(
    states: Sequence[Any], actions: Sequence[Any], observation: Optional[Any] = None
) -> List[StepData]:
    """An episode through ``states``, ending on a terminal bookkeeping step."""
    history = []
    for index, state in enumerate(states):
        is_last = index == len(states) - 1
        history.append(
            StepData(
                state=state,
                action=None if is_last else actions[index],
                next_state=None if is_last else states[index + 1],
                observation=None if is_last else observation,
                reward=None if is_last else -1.0,
                belief=WeightedParticleBelief(
                    particles=[state, state], log_weights=np.log(np.full(2, 0.5))
                ),
            )
        )
    return history


def _ended(environment: Any, history: List[StepData]) -> bool:
    return environment.episode_visualizer().build_payload(history)["ended_in_danger_zone"]


# ---------------------------------------------------------------- Light-Dark


def test_light_dark_hazard_hit_is_a_danger_zone_ending():
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=True)
    states = [np.array([4.0, 5.0, 0.0]), np.array([5.0, 5.0, 1.0])]
    assert _ended(env, _history(states, [np.array([1.0, 0.0])])) is True


def test_light_dark_goal_is_not_a_danger_zone_ending():
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=True)
    goal = np.asarray(env.goal_state, dtype=float).reshape(-1)[:2]
    states = [np.array([goal[0] - 1.0, goal[1], 0.0]), np.array([goal[0], goal[1], 0.0])]
    assert _ended(env, _history(states, [np.array([1.0, 0.0])])) is False


def test_continuous_light_dark_without_the_flag_never_ends_in_a_danger_zone():
    # Without the flag the continuous world ends only at the goal, even on an
    # obstacle centre.
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    obstacle = np.asarray(env.obstacles, dtype=float).T[0]
    states = [obstacle - np.array([1.0, 0.0]), obstacle]
    assert _ended(env, _history(states, [np.array([1.0, 0.0])])) is False


def test_discrete_light_dark_obstacle_cell_ends_in_a_danger_zone_without_the_flag():
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    obstacle = np.asarray(env.obstacles, dtype=float).T[0]
    states = [obstacle - np.array([1.0, 0.0]), obstacle]
    assert _ended(env, _history(states, [env.get_actions()[0]])) is True


# -------------------------------------------------------------------- PacMan


def _pacman(**overrides: Any) -> PacManPOMDP:
    kwargs = {
        "maze_size": (7, 7),
        "num_ghosts": 1,
        "initial_ghost_positions": [(6, 6)],
        "dangerous_areas": {(0, 3)},
        "is_dangerous_area_hit_terminal": True,
    }
    kwargs.update(overrides)
    return PacManPOMDP(**kwargs)


def _pacman_history(
    env: PacManPOMDP, ghost_after: tuple, pacman_after: tuple = (0, 3)
) -> List[StepData]:
    pellets = tuple(env.initial_pellets)
    before = (pacman_after[0], pacman_after[1] - 1)
    states = [
        env.make_state(pacman_pos=before, ghost_positions=((6, 6),), pellets=pellets),
        env.make_state(
            pacman_pos=pacman_after, ghost_positions=(ghost_after,), pellets=pellets, terminal=True
        ),
    ]
    return _history(states, [1], observation=((6, 6),))


def test_pacman_entering_a_zone_is_a_danger_zone_ending():
    env = _pacman()
    assert _ended(env, _pacman_history(env, ghost_after=(6, 5))) is True


def test_pacman_decayed_hazard_outside_every_zone_is_a_danger_zone_ending():
    # The distance-decayed variant can end the episode with PacMan in no zone.
    env = _pacman(reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY)
    assert _ended(env, _pacman_history(env, (6, 5), (5, 2))) is True


def test_pacman_ghost_collision_in_a_zone_is_not_a_danger_zone_ending():
    env = _pacman()
    assert _ended(env, _pacman_history(env, ghost_after=(0, 3))) is False


def test_pacman_without_the_flag_never_ends_in_a_danger_zone():
    env = _pacman(is_dangerous_area_hit_terminal=False)
    assert _ended(env, _pacman_history(env, ghost_after=(6, 5))) is False


# ----------------------------------------------------------------- Laser Tag


def test_laser_tag_terminal_move_is_a_danger_zone_ending():
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    states = [np.array([5.0, 2.0, 0.0, 6.0, 0.0]), np.array([5.0, 3.0, 0.0, 6.0, 1.0])]
    assert _ended(env, _history(states, [1])) is True


def test_laser_tag_successful_tag_is_not_a_danger_zone_ending():
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    states = [np.array([0.0, 6.0, 0.0, 6.0, 0.0]), np.array([0.0, 6.0, 0.0, 6.0, 1.0])]
    assert _ended(env, _history(states, [4])) is False


def test_continuous_laser_tag_terminal_move_is_a_danger_zone_ending():
    env = ContinuousLaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    states = [np.array([4.5, 3.0, 9.0, 6.0, 0.0]), np.array([5.0, 3.0, 9.0, 6.0, 1.0])]
    assert _ended(env, _history(states, [np.array([1.0, 0.0, 0.0])])) is True


def test_continuous_laser_tag_successful_tag_is_not_a_danger_zone_ending():
    env = ContinuousLaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    states = [np.array([5.0, 3.0, 5.2, 3.0, 0.0]), np.array([5.0, 3.0, 5.2, 3.0, 1.0])]
    assert _ended(env, _history(states, [np.array([0.0, 0.0, 1.0])])) is False


# ---------------------------------------------------------- Continuous Push


def _push(obstacle_terminal: bool = True) -> ContinuousPushPOMDP:
    return ContinuousPushPOMDP(
        discount_factor=0.95,
        dangerous_areas=[(5.0, 5.0)],
        dangerous_area_radius=1.0,
        is_dangerous_area_hit_terminal=True,
        is_obstacle_hit_terminal=obstacle_terminal,
    )


def test_push_terminal_slot_inside_a_zone_is_a_danger_zone_ending():
    states = [
        np.array([4.0, 5.0, 2.0, 2.0, 9.0, 9.0, 0.0]),
        np.array([5.0, 5.0, 2.0, 2.0, 9.0, 9.0, 1.0]),
    ]
    assert _ended(_push(), _history(states, [np.array([1.0, 0.0])])) is True


def test_push_terminal_slot_outside_every_zone_is_not_a_danger_zone_ending():
    # The slot is shared with an obstacle hit; outside every zone it is that.
    states = [
        np.array([1.0, 8.0, 2.0, 2.0, 9.0, 9.0, 0.0]),
        np.array([1.0, 9.0, 2.0, 2.0, 9.0, 9.0, 1.0]),
    ]
    assert _ended(_push(), _history(states, [np.array([0.0, 1.0])])) is False


def test_push_terminal_slot_outside_every_zone_is_a_hazard_without_obstacle_endings():
    # With only the dangerous-area flag on, the slot is the hazard's wherever
    # the rover is: the distance-decayed variant fires outside every zone.
    states = [
        np.array([1.0, 8.0, 2.0, 2.0, 9.0, 9.0, 0.0]),
        np.array([1.0, 9.0, 2.0, 2.0, 9.0, 9.0, 1.0]),
    ]
    push = _push(obstacle_terminal=False)
    assert _ended(push, _history(states, [np.array([0.0, 1.0])])) is True


# --------------------------------------------------------------- RockSample


def _rock_sample() -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=(5, 5),
        rock_positions=[(0, 0), (2, 2), (3, 3)],
        dangerous_areas=[(1, 2)],
        is_dangerous_area_hit_terminal=True,
    )


def test_rock_sample_terminal_slot_is_a_danger_zone_ending():
    states = [
        np.array([1, 1, 1, 0, 1, 0], dtype=np.float32),
        np.array([1, 2, 1, 0, 1, 1], dtype=np.float32),
    ]
    assert _ended(_rock_sample(), _history(states, [3], observation="none")) is True


def test_rock_sample_exit_is_not_a_danger_zone_ending():
    states = [
        np.array([1, 4, 1, 0, 1, 0], dtype=np.float32),
        np.array([-1, -1, 1, 0, 1, 0], dtype=np.float32),
    ]
    assert _ended(_rock_sample(), _history(states, [1], observation="none")) is False
