# SPDX-License-Identifier: MIT

"""The firefighting filter must not drown in "the fire is already out".

A robot sees only the cells inside its sensing radius; everything else reports
UNKNOWN, so a reading taken while three cells burn out of sight looks exactly
like one taken over a cold grid. Burnt and wet cells never re-ignite, so a
particle whose last alight cell goes out can never come back -- and the belief
drained into certainty that the fire was out while it was still burning.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
    MultiAgentFirefightingPOMDP,
)
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_vectorized_belief import (  # noqa: E501
    FirefightingVectorizedUpdater,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    multiagent_firefighting_pinned_kwargs,
)
from POMDPPlanners.tests.test_utils.terminal_particle_weight import (
    random_actions,
    replay_terminal_weights,
)

# One seed per population, each the worst case found for it before the
# conditioning: 0.8320 of the weight on terminal particles at 60, 1.0000 at 400
# and 0.5090 at 2000, sustained for the rest of the episode.
COLLAPSING_SEEDS = {60: 8, 400: 5, 2000: 4}


def build_env(**overrides) -> MultiAgentFirefightingPOMDP:
    """Build the pinned firefighting environment."""
    return MultiAgentFirefightingPOMDP(
        discount_factor=0.95, **multiagent_firefighting_pinned_kwargs(**overrides)
    )


@pytest.mark.parametrize("n_particles", [60, 400, 2000])
def test_no_weight_sits_on_terminal_states_while_the_episode_runs(n_particles):
    """A running episode is evidence, and the filter must spend it.

    Purpose: "The fire is out" is terminal and invisible: out-of-sight cells
        report UNKNOWN, so the likelihood cannot tell a burning grid from a
        cold one, and a particle that goes cold is absorbing because burnt and
        wet cells never re-ignite. Before the conditioning, up to 1.0000 of the
        belief's weight sat on "the fire is already out" while it burned, and
        once there it stayed for the rest of the episode.

    Given: The pinned environment and a random joint policy.
    When: The belief is filtered along an episode.
    Then: No weight sits on a terminal particle on any step the real episode
        survived.

    Test type: integration
    """
    env = build_env()
    weights = replay_terminal_weights(
        env,
        random_actions(env),
        n_particles=n_particles,
        seed=COLLAPSING_SEEDS[n_particles],
    )

    assert weights, "fixture is wrong: the episode ended before a single belief update"
    assert max(weights) == 0.0, (
        f"{n_particles} particles: up to {max(weights):.4f} of the belief's weight says "
        "the fire is out while it is still burning"
    )


def test_the_shared_step_limit_does_not_rule_any_particle_out():
    """Terminality every particle shares carries no information.

    Purpose: The step counter is deterministic and identical across particles,
        so conditioning on it would floor the entire population on the final
        step and leave a belief supported on nothing -- for nothing, since a
        factor identical across particles cancels in the posterior. Only the
        fire going out and the robots going down separate one particle from
        another.

    Given: Particles that have all reached the step limit, with fire still
        alight and robots still up.
    When: They are tested against the running-episode mask.
    Then: None of them is ruled out, even though the environment calls every
        one of them terminal.

    Test type: unit
    """
    env = build_env(max_steps=4)
    particles = np.stack(env.initial_state_dist().sample(n_samples=8))
    particles[:, 0] = float(env.max_steps)
    updater = FirefightingVectorizedUpdater.from_environment(env)

    ruled_out = updater.ruled_out_by_a_running_episode(particles)

    assert all(
        env.is_terminal(particle) for particle in particles
    ), "fixture is wrong: the step limit must make every particle terminal"
    assert not ruled_out.any()


def test_a_planners_update_is_not_conditioned():
    """Inside a search tree the episode continuing is not evidence.

    Purpose: A planner expanding its tree is asking what happens *if* it acts,
        and the fire going out is one of the answers -- it is how a branch
        stops paying. ``is_terminal_belief`` is what SparsePFT and
        ICVaR-PFT-DPW cut a node on, and PFT-DPW samples a particle for the
        same test. Conditioning in there would tell the search the fire can
        never go out. Only the episode driver passes a state, so only the
        episode driver gets the conditioning.

    Given: A belief whose particles have all gone cold but one.
    When: It is updated without a state, the way a planner does.
    Then: The cold particles keep their weight.

    Test type: unit
    """
    from POMDPPlanners.utils.belief_factory import create_environment_belief

    np.random.seed(0)
    env = build_env()
    belief = create_environment_belief(env, n_particles=16)
    state = env.initial_state_dist().sample()[0]
    action = env.get_actions()[0]
    _, observation, _ = env.sample_next_step(state, action)

    in_tree = belief.update(action=action, observation=observation, pomdp=env, state=None)
    ruled_out = belief.updater.ruled_out_by_a_running_episode(np.asarray(in_tree.particles))

    assert not np.isneginf(np.asarray(in_tree.log_weights)[~ruled_out]).all()
    assert np.all(np.isfinite(np.asarray(in_tree.log_weights)[ruled_out])) or not ruled_out.any()
