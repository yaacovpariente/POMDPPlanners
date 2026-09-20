# SPDX-License-Identifier: MIT

"""Replay an episode and watch how much belief sits on terminal particles.

Five environments shared one defect: their filters never conditioned on the
episode still running, so particles that drifted into an absorbing,
unobservable terminal state were never removed and never moved again, and the
belief drained into certainty that the episode was already over. Each of those
environments pins the repaired behaviour with a test built on this helper, so
a regression fails with a number rather than silently.

The drive-at-the-hazard policy matters: a random walk mostly never reaches the
one place a hidden terminal state can be entered, and measures zero on a
broken filter. Every environment here is steered at its hazard instead.

Functions:
    terminal_weight: Share of a belief's weight on terminal particles.
    replay_terminal_weights: Per-step terminal weight over one episode.
"""

import random
from typing import Any, Callable, List, Optional

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.belief_factory import create_environment_belief
from POMDPPlanners.tests.test_utils.particle_belief_typing import particle_belief


def terminal_weight(belief: Belief, env: Environment) -> float:
    """Return the share of *belief*'s weight on states *env* calls terminal.

    Args:
        belief: A particle belief.
        env: The environment whose terminal test to apply.

    Returns:
        A number in ``[0, 1]``.
    """
    particles = particle_belief(belief)
    mask = np.array([bool(env.is_terminal(particle)) for particle in particles.particles])
    weights = np.asarray(particles.normalized_weights, dtype=float)
    return float(weights[mask].sum())


def greedy_towards(env: Environment, target: np.ndarray) -> Callable[[Any], Any]:
    """Build a policy that steps greedily toward *target*.

    Args:
        env: The environment, used for its action set and dynamics.
        target: The 2-D point to walk at, usually a hazard.

    Returns:
        A callable mapping a state to an action.
    """
    actions: Optional[List[Any]] = None
    getter = getattr(env, "get_actions", None)
    if getter is not None:
        try:
            actions = list(getter())
        except (TypeError, NotImplementedError):
            actions = None

    def policy(state: Any) -> Any:
        if actions is None:
            delta = target - np.asarray(state, dtype=float)[:2]
            return np.clip(delta, -1.0, 1.0)
        best, best_distance = actions[0], float("inf")
        for action in actions:
            candidate = env.sample_next_state(state, action)
            distance = float(np.linalg.norm(np.asarray(candidate, dtype=float)[:2] - target))
            if distance < best_distance:
                best, best_distance = action, distance
        return best

    return policy


def random_actions(env: Environment) -> Callable[[Any], Any]:
    """Build a policy that picks uniformly from the action set.

    For an environment whose hidden terminal state is not somewhere on the map
    -- "the fire is out" -- there is nothing to steer at, and a random walk
    reaches it as well as anything.

    Args:
        env: The environment, read for its action set.

    Returns:
        A callable mapping a state to an action.
    """
    actions = list(env.get_actions())  # type: ignore[attr-defined]

    def policy(state: Any) -> Any:
        del state
        return random.choice(actions)

    return policy


def replay_terminal_weights(
    env: Environment,
    policy: Callable[[Any], Any],
    n_particles: int,
    max_steps: int = 30,
    seed: int = 1000,
) -> List[float]:
    """Filter one episode and report the terminal weight after every step.

    Only steps the real episode survived are reported: the question is how
    much of the belief says the episode is over *while it is running*.

    Args:
        env: The environment to run.
        policy: Maps a state to the action to take.
        n_particles: Belief population size.
        max_steps: Cap on episode length.
        seed: Seed for both RNGs.

    Returns:
        One terminal weight per surviving step.
    """
    random.seed(seed)
    np.random.seed(seed)
    belief = create_environment_belief(env, n_particles=n_particles)
    state = env.initial_state_dist().sample()[0]
    weights: List[float] = []
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        action = policy(state)
        next_state, observation, _ = env.sample_next_step(state, action)
        belief = belief.update(action=action, observation=observation, pomdp=env, state=next_state)
        state = next_state
        if env.is_terminal(state):
            break
        weights.append(terminal_weight(belief, env))
    return weights
