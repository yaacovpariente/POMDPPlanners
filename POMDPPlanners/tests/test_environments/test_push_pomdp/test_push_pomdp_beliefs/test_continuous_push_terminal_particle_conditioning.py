# SPDX-License-Identifier: MIT

"""The Continuous Push filter must not drown in terminal particles.

The native likelihood scores the object's position and nothing else -- not the
robot, not the terminal slot. A particle whose robot was absorbed by a hazard
freezes, but its object was already at rest, so it goes on scoring exactly as
well as a live particle and never leaves the population.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.push_pomdp import ContinuousPushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.continuous_push_vectorized_updater import (  # noqa: E501
    ContinuousPushVectorizedUpdater,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_push_pinned_kwargs
from POMDPPlanners.tests.test_utils.terminal_particle_weight import (
    greedy_towards,
    replay_terminal_weights,
)

HAZARD = np.array([4.0, 4.0])

# One seed per population, each collapsing the whole belief onto terminal
# particles before the conditioning. The robot starts at (1, 1), far from the
# hazard, because starting next to it ends the real episode in two or three
# steps and the population never gets time to drift in -- the first attempt at
# measuring this environment did exactly that and read zero on a broken filter.
COLLAPSING_SEEDS = {60: 27, 400: 18, 2000: 6}


def build_env(**overrides) -> ContinuousPushPOMDP:
    """Build the pinned environment with two hazards and hazard termination."""
    return ContinuousPushPOMDP(
        discount_factor=0.95,
        **continuous_push_pinned_kwargs(
            dangerous_areas=[(4.0, 4.0), (6.0, 6.0)],
            dangerous_area_hit_probability=0.05,
            dangerous_area_radius=1.0,
            is_dangerous_area_hit_terminal=True,
            initial_state=np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0]),
            **overrides,
        ),
    )


@pytest.mark.parametrize("n_particles", [60, 400, 2000])
def test_no_weight_sits_on_terminal_states_while_the_episode_runs(n_particles):
    """A running episode is evidence, and the filter must spend it.

    Purpose: With ``is_dangerous_area_hit_terminal`` on, a hit latches a slot
        the reading never names and the state freezes. Since the likelihood
        reads only the object's position, and the object stops moving once the
        robot does, a hit particle is indistinguishable from a live one that
        happens to be holding still. Before the conditioning, walking the
        robot at the hazard at (4, 4) drove the terminal count up
        monotonically -- 0 to 97 of 400 particles over twelve steps -- and put
        1.0000 of the belief's weight on terminal particles at every
        population below.

    Given: The pinned environment with two hazards and hazard termination on.
    When: The robot is walked at one and the belief filtered along.
    Then: No weight sits on a terminal particle on any step the real episode
        survived.

    Test type: integration
    """
    env = build_env()
    weights = replay_terminal_weights(
        env,
        greedy_towards(env, HAZARD),
        n_particles=n_particles,
        seed=COLLAPSING_SEEDS[n_particles],
    )

    assert weights, "fixture is wrong: the episode ended before a single belief update"
    assert max(weights) == 0.0, (
        f"{n_particles} particles: up to {max(weights):.4f} of the belief's weight says "
        "the episode is over while it is still running"
    )


def test_the_flag_off_environment_is_left_exactly_as_it_was():
    """Only the configuration that was measured is changed.

    Purpose: With both hazard flags off the state carries no slot, the only
        terminal condition is the object reaching the goal, and that is
        exactly what the reading measures -- nothing accumulates. That
        configuration was not measured, so it is not conditioned.

    Given: A Continuous Push environment with both hazard flags off.
    When: Its updater is asked which particles a running episode rules out.
    Then: It declines to say.

    Test type: unit
    """
    env = ContinuousPushPOMDP(discount_factor=0.95, **continuous_push_pinned_kwargs())
    updater = ContinuousPushVectorizedUpdater.from_environment(env)

    assert updater.ruled_out_by_a_running_episode(np.zeros((5, 6))) is None
