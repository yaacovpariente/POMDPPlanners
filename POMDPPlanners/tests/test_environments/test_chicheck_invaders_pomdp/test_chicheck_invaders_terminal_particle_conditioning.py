# SPDX-License-Identifier: MIT

"""Neither Chicheck Invaders filter may drown in terminal particles.

Nothing in a reading names the ship-hit flag, so the sensor likelihood cannot
tell a live world from a lost one, and a hit state is absorbing -- the flag
never clears and the chicken that set it parks on row 0. Both the scalar filter
and its batched twin are pinned, because the environment ships both and the
batched one is what the belief factory hands out.

The planning half is pinned just as hard. Conditioning inside a search tree
would tell the planner the ship can never be destroyed, on an environment where
that is the whole risk.
"""

import random

import numpy as np
import pytest

from POMDPPlanners.environments.chicheck_invaders_pomdp import ChicheckInvadersPOMDP
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    SHIP_HIT_INDEX,
    ruled_out_by_a_running_episode,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import chicheck_invaders_pinned_kwargs
from POMDPPlanners.tests.test_utils.terminal_particle_weight import terminal_weight
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief

# 0 stay, 1 left, 2 right, 3 fire -- the golden GIF's sequence.
GOLDEN_ACTIONS = [1, 3, 0, 2, 3, 2, 0, 3, 1, 3, 0, 3]

# Seeds under which the belief went *entirely* certain the ship was already
# destroyed, one per (belief type, population). Before the conditioning the
# collapse happened in 8 of 20 seeds at 60 particles on the batched filter,
# 5 of 20 at 400 and 3 of 20 at 2000, and in 5, 6 and 1 of 20 on the scalar one.
COLLAPSING_SEEDS = {
    (BeliefType.VECTORIZED_PARTICLE, 60): 19,
    (BeliefType.VECTORIZED_PARTICLE, 400): 16,
    (BeliefType.VECTORIZED_PARTICLE, 2000): 19,
    (BeliefType.PARTICLE, 60): 0,
    (BeliefType.PARTICLE, 400): 17,
    (BeliefType.PARTICLE, 2000): 11,
}


def build_env() -> ChicheckInvadersPOMDP:
    """Build the pinned golden environment."""
    return ChicheckInvadersPOMDP(discount_factor=0.95, **chicheck_invaders_pinned_kwargs())


def replay(belief_type: BeliefType, n_particles: int, seed: int):
    """Filter the golden action sequence and report terminal weight per step."""
    random.seed(seed)
    np.random.seed(seed)
    env = build_env()
    belief = create_environment_belief(env, n_particles=n_particles, belief_type=belief_type)
    state = env.initial_state_dist().sample()[0]
    weights = []
    for action in GOLDEN_ACTIONS:
        if env.is_terminal(state):
            break
        next_state, observation, _ = env.sample_next_step(state, action)
        belief = belief.update(
            action=action, observation=observation, pomdp=env, state=next_state
        )
        state = next_state
        if env.is_terminal(state):
            break
        weights.append(terminal_weight(belief, env))
    return weights


@pytest.mark.parametrize(
    ("belief_type", "n_particles"), sorted(COLLAPSING_SEEDS, key=lambda key: str(key))
)
def test_no_weight_sits_on_terminal_states_while_the_episode_runs(belief_type, n_particles):
    """A running episode is evidence, and both filters must spend it.

    Purpose: The reading carries the ship's column and what the two sensors
        found, and nothing else -- it cannot separate a live world from one in
        which a chicken already reached the ship. That state is absorbing, so a
        particle that drifts in is never removed and never moves again, and the
        belief drains into certainty that the ship is gone while the real
        episode runs on. Under each seed below the whole belief ended up on
        terminal particles before the conditioning. Both belief types are
        pinned because the environment ships both, and the batched one is the
        belief factory's default.

    Given: The pinned golden environment and its golden action sequence.
    When: The belief is filtered along the episode.
    Then: No weight sits on a terminal particle on any step the real episode
        survived.

    Test type: integration
    """
    weights = replay(belief_type, n_particles, COLLAPSING_SEEDS[(belief_type, n_particles)])

    assert weights, "fixture is wrong: the episode ended before a single belief update"
    assert max(weights) == 0.0, (
        f"{belief_type.value} with {n_particles} particles: up to {max(weights):.4f} of the "
        "belief's weight says the ship is already destroyed while the episode runs on"
    )


@pytest.mark.parametrize(
    "belief_type", [BeliefType.VECTORIZED_PARTICLE, BeliefType.PARTICLE]
)
def test_a_planners_update_is_not_conditioned(belief_type):
    """Inside a search tree the ship surviving is not evidence.

    Purpose: This is the failure the first version of this fix shipped. It
        conditioned unconditionally, so the beliefs a planner builds in its
        tree were conditioned too -- and a tree that cannot reach a destroyed
        ship plans as though the ship were indestructible, on the one
        environment whose entire risk is that it is not.
        ``is_terminal_belief`` is what SparsePFT and ICVaR-PFT-DPW cut a node
        on, and PFT-DPW samples a particle for the same test. Only the episode
        driver passes a state, so only the episode driver conditions.

    Given: A belief whose particles are half destroyed ships and half live
        ones, so the mask separates them and nothing else does.
    When: The belief is updated without a state, the way every planner does,
        and with one, the way the episode driver does.
    Then: The planner's update leaves the destroyed half carrying weight, and
        the driver's removes it.

    Test type: unit
    """
    random.seed(5)
    np.random.seed(5)
    env = build_env()
    belief = create_environment_belief(env, n_particles=200, belief_type=belief_type)
    state = env.initial_state_dist().sample()[0]
    next_state, observation, _ = env.sample_next_step(state, GOLDEN_ACTIONS[0])

    particles = np.array(np.asarray(belief.particles, dtype=np.float64), copy=True)
    particles[::2, SHIP_HIT_INDEX] = 1.0
    if isinstance(belief.particles, np.ndarray):
        belief.particles = particles
    else:
        belief.particles = [np.array(row, copy=True) for row in particles]

    in_tree = belief.update(
        action=GOLDEN_ACTIONS[0], observation=observation, pomdp=env
    )
    at_execution = belief.update(
        action=GOLDEN_ACTIONS[0], observation=observation, pomdp=env, state=next_state
    )

    in_tree_mask = ruled_out_by_a_running_episode(
        np.asarray(in_tree.particles, dtype=np.float64), env.num_chickens
    )
    assert in_tree_mask.any(), (
        "fixture is wrong: a destroyed ship is absorbing, so an unconditioned update "
        "must carry those particles forward"
    )
    assert float(np.asarray(in_tree.normalized_weights)[in_tree_mask].sum()) > 0.0, (
        "a planner's belief update must leave the destroyed ship reachable: the search "
        "has to be able to value the risk it is taking"
    )
    assert terminal_weight(at_execution, env) == 0.0


def test_the_shared_step_limit_does_not_rule_any_particle_out():
    """Terminality every particle shares carries no information.

    Purpose: The step counter is deterministic and identical across particles,
        so conditioning on it would floor the entire population on the final
        step and leave a belief supported on nothing -- for nothing, since a
        factor identical across particles cancels in the posterior. Only the
        ship-hit flag and the cleared flock separate one particle from another.

    Given: Particles that have all reached the step limit, ship intact and
        flock alive.
    When: They are tested against the running-episode mask.
    Then: None of them is ruled out, even though the environment calls every
        one of them terminal.

    Test type: unit
    """
    env = ChicheckInvadersPOMDP(
        discount_factor=0.95, **chicheck_invaders_pinned_kwargs(max_steps=5)
    )
    particles = np.stack(env.initial_state_dist().sample(n_samples=8))
    particles[:, 0] = float(env.max_steps)

    ruled_out = ruled_out_by_a_running_episode(particles, env.num_chickens)

    assert all(
        env.is_terminal(particle) for particle in particles
    ), "fixture is wrong: the step limit must make every particle terminal"
    assert not ruled_out.any()
