# SPDX-License-Identifier: MIT

"""The light-dark filters must not drown in terminal particles.

All three hazard-terminal light-dark environments had the same defect: a hit
is a hidden Bernoulli draw the reading never names, the state freezes where it
was hit, and the belief drained into certainty that the episode was already
over while it ran on. These pin the repair at three populations, because the
first reading of the Chicheck version of this bug was that a small particle
count was to blame, and it was not.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_light_dark_discrete_actions_pinned_kwargs,
    continuous_light_dark_pinned_kwargs,
    discrete_light_dark_pinned_kwargs,
)
from POMDPPlanners.tests.test_utils.terminal_particle_weight import (
    greedy_towards,
    replay_terminal_weights,
)

# The obstacle the pinned layout puts in the robot's way. Walking at it is the
# only way to reach the hidden absorbing state at all; a random walk measures
# zero on a broken filter and pins nothing.
HAZARD = np.array([5.0, 5.0])

BUILDERS = {
    "ContinuousLightDarkPOMDP": lambda: ContinuousLightDarkPOMDP(
        discount_factor=0.95, **continuous_light_dark_pinned_kwargs()
    ),
    "ContinuousLightDarkPOMDPDiscreteActions": (
        lambda: ContinuousLightDarkPOMDPDiscreteActions(
            discount_factor=0.95,
            **continuous_light_dark_discrete_actions_pinned_kwargs(),
        )
    ),
    "DiscreteLightDarkPOMDP": lambda: DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        **discrete_light_dark_pinned_kwargs(is_obstacle_hit_terminal=True),
    ),
}


# One seed per (environment, population), each chosen because the *whole*
# belief sat on terminal particles under it before the conditioning. A seed
# that does not reproduce the failure pins nothing, and most do not: the robot
# has to reach the obstacle and survive it long enough for the population to
# drift in, which only some episodes do.
COLLAPSING_SEEDS = {
    ("ContinuousLightDarkPOMDP", 60): 1,
    ("ContinuousLightDarkPOMDP", 400): 56,
    ("ContinuousLightDarkPOMDP", 2000): 39,
    ("ContinuousLightDarkPOMDPDiscreteActions", 60): 54,
    ("ContinuousLightDarkPOMDPDiscreteActions", 400): 16,
    ("ContinuousLightDarkPOMDPDiscreteActions", 2000): 54,
    ("DiscreteLightDarkPOMDP", 60): 39,
    ("DiscreteLightDarkPOMDP", 400): 49,
    ("DiscreteLightDarkPOMDP", 2000): 22,
}


@pytest.mark.parametrize(
    ("env_name", "n_particles"), sorted(COLLAPSING_SEEDS), ids=lambda value: str(value)
)
def test_no_weight_sits_on_terminal_states_while_the_episode_runs(env_name, n_particles):
    """A running episode is evidence, and the filter must spend it.

    Purpose: With ``is_obstacle_hit_terminal`` on, a hit sets a slot no
        reading names, the state stops moving, and the vectorized updater
        drops the slot before scoring -- so a hit particle keeps whatever its
        stale position earns and keeps it forever. Under each of the seeds
        below, walking the robot at the obstacle at (5, 5) put the *entire*
        belief -- 1.0000 of the weight -- on terminal particles while the real
        episode ran on, at every one of the three populations. The Discrete
        Light-Dark cases also exercise the second half of the repair: there
        the reading was impossible under every particle as well, and the
        all--inf normalisation fallback used to hand the resample the
        ruled-out particles back at uniform weight.

    Given: A pinned hazard-terminal light-dark environment.
    When: The robot is walked at the obstacle and the belief filtered along.
    Then: No weight sits on a terminal particle on any step the real episode
        survived.

    Test type: integration
    """
    env = BUILDERS[env_name]()
    weights = replay_terminal_weights(
        env,
        greedy_towards(env, HAZARD),
        n_particles=n_particles,
        seed=COLLAPSING_SEEDS[(env_name, n_particles)],
    )

    assert weights, "fixture is wrong: the episode ended before a single belief update"
    assert max(weights) == 0.0, (
        f"{env_name} with {n_particles} particles: up to {max(weights):.4f} of the belief's "
        "weight says the episode is over while it is still running"
    )


def test_the_flag_off_environment_is_left_exactly_as_it_was():
    """Only the configuration that was measured is changed.

    Purpose: With the hazard flag off there is no hidden absorbing state to
        accumulate -- the only terminal condition is standing at the goal,
        nothing latches, and a particle there moves on like any other. That
        configuration was not measured, so it is not conditioned, and this
        pins that decision rather than leaving it to be undone by accident.

    Given: A continuous light-dark environment with the flag off.
    When: Its updater is asked which particles a running episode rules out.
    Then: It declines to say, which is the contract for "do not condition".

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        **continuous_light_dark_pinned_kwargs(is_obstacle_hit_terminal=False),
    )
    from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
        ContinuousLightDarkVectorizedUpdater,
    )

    updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)

    assert updater.ruled_out_by_a_running_episode(np.zeros((5, 2))) is None
