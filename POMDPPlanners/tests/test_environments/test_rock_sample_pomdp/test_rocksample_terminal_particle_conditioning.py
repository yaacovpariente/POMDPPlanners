# SPDX-License-Identifier: MIT

"""The RockSample filter must not drown in terminal particles.

This is the starkest of the five environments that shared the defect. On any
action that is not a rock check the sensor kernel gives every particle the
same score, so a move step reweights nothing at all; a particle whose robot
walked into a dangerous area latches its terminal slot, freezes, and collects
that identical score for the rest of the episode. The belief ended up certain
the episode was over while it ran on.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs.rocksample_vectorized_updater import (  # noqa: E501
    RockSampleVectorizedUpdater,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import rock_sample_pinned_kwargs
from POMDPPlanners.tests.test_utils.particle_belief_typing import particle_belief
from POMDPPlanners.tests.test_utils.terminal_particle_weight import (
    greedy_towards,
    replay_terminal_weights,
)

HAZARD = np.array([2.0, 3.0])

# Seed 30 collapses the belief at every population: 1.0000 of the weight on
# terminal particles at 60 particles, 0.9900 at 400 and 0.9850 at 2000, all
# while the episode was still running.
COLLAPSING_SEED = 30


def build_env(**overrides) -> RockSamplePOMDP:
    """Build the pinned map with two dangerous areas and hazard termination."""
    return RockSamplePOMDP(
        discount_factor=0.95,
        **rock_sample_pinned_kwargs(
            dangerous_areas=[(2, 3), (3, 1)],
            dangerous_area_hit_probability=0.3,
            is_dangerous_area_hit_terminal=True,
            **overrides,
        ),
    )


@pytest.mark.parametrize("n_particles", [60, 400, 2000])
def test_no_weight_sits_on_terminal_states_while_the_episode_runs(n_particles):
    """A running episode is evidence, and the filter must spend it.

    Purpose: Nothing in a RockSample reading names the terminal slot, and on a
        move step every particle scores identically, so the likelihood cannot
        separate a live world from one the robot never left. A hit state is
        absorbing, so those particles never move again either. Before the
        conditioning, walking the robot into the dangerous area at (2, 3) put
        essentially the whole belief on terminal particles -- 1.0000 of the
        weight at 60 particles, 0.9900 at 400 and 0.9850 at 2000 -- while the
        real episode ran on.

    Given: The pinned map with two dangerous areas and hazard termination on.
    When: The robot is walked into one and the belief filtered along.
    Then: No weight sits on a terminal particle on any step the real episode
        survived.

    Test type: integration
    """
    env = build_env()
    weights = replay_terminal_weights(
        env, greedy_towards(env, HAZARD), n_particles=n_particles, seed=COLLAPSING_SEED
    )

    assert weights, "fixture is wrong: the episode ended before a single belief update"
    assert max(weights) == 0.0, (
        f"{n_particles} particles: up to {max(weights):.4f} of the belief's weight says "
        "the episode is over while it is still running"
    )


def test_the_vectorized_belief_runs_at_all_with_hazard_termination_on():
    """The sensor kernel has to be handed a state it accepts.

    Purpose: This is a separate, older defect that blocked measuring the first
        one. ``RockSampleObservationCpp::batch_log_likelihood`` is written for
        the canonical ``(N, 2 + num_rocks)`` layout and rejects anything
        wider, so every update of a hazard-terminal belief raised
        ``next_particles must have shape (N, 2 + num_rocks)``. The vectorized
        belief simply could not be used with the flag on.

    Given: A hazard-terminal environment and its belief.
    When: One step is filtered.
    Then: It returns a belief, rather than raising.

    Test type: integration
    """
    from POMDPPlanners.utils.belief_factory import create_environment_belief

    np.random.seed(0)
    env = build_env()
    belief = particle_belief(create_environment_belief(env, n_particles=32))
    state = env.initial_state_dist().sample()[0]
    next_state, observation, _ = env.sample_next_step(state, env.get_actions()[0])

    updated = belief.update(
        action=env.get_actions()[0], observation=observation, pomdp=env, state=next_state
    )

    assert updated.particles.shape == belief.particles.shape


def test_the_flag_off_environment_is_left_exactly_as_it_was():
    """Only the configuration that was measured is changed.

    Purpose: With hazard termination off the state carries no slot, the only
        terminal condition is the robot having exited east, and an exited
        robot is a state the episode's own bookkeeping ends on rather than
        something the filter has to carry. That configuration was not
        measured, so it is not conditioned.

    Given: A RockSample environment with the flag off.
    When: Its updater is asked which particles a running episode rules out.
    Then: It declines to say.

    Test type: unit
    """
    env = RockSamplePOMDP(discount_factor=0.95, **rock_sample_pinned_kwargs())
    updater = RockSampleVectorizedUpdater.from_environment(env)

    assert updater.ruled_out_by_a_running_episode(np.zeros((5, 5))) is None
