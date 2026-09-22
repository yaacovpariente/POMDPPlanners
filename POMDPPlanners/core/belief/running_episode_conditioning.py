# SPDX-License-Identifier: MIT

"""Conditioning a particle filter on the fact that the episode is still running.

The agent being asked for another action is evidence, and in some environments
it is evidence nothing else carries. Where a terminal state is **absorbing and
unobservable** -- a latched hazard flag the sensors never name, a frozen world
that keeps scoring against whatever reading arrives -- a particle that drifts
into it is never contradicted by the observation model and never moves again.
Such particles only accumulate, and the belief drains into an "the episode is
already over" hypothesis while the real episode plays on.

The driver stops an episode the moment the state is terminal, so a step that
happens at all rules out every particle the environment would have called
terminal. :func:`condition_log_weights_on_a_running_episode` spends that
evidence by flooring those particles' log-weights, which is what lets the
resample drop them.

Two rules the helper deliberately keeps:

* **Only the part of terminality that varies between particles.** A shared
  step limit rules out every particle or none, and a factor identical across
  particles cancels in the posterior; flooring on it would zero a whole
  population on the last step for nothing. Callers pass a mask built from the
  per-particle part alone.
* **Never floor everything.** A belief supported on nothing cannot be
  normalised, let alone resampled. When every particle is ruled out the
  weights are returned untouched and the caller is told, so it can rebuild the
  population instead.

This is an **execution-time** filter step only. Inside a planner's search tree
there is no such evidence: the tree is asking what happens *if* the agent acts,
and termination is one of the answers. See the module docstring of
``POMDPPlanners.core.belief.vectorized_weighted_particle_belief`` for how that
split is made.

Functions:
    condition_log_weights_on_a_running_episode: Floor the ruled-out particles.
"""

from typing import Tuple

import numpy as np

# Subtracted from the worst surviving log-weight to place a ruled-out particle
# below every particle the episode is still allowed to be in.
#
# A fixed sentinel will not do. Environments that return raw log-likelihoods
# give a contradicted particle a large negative score of its own, so assigning
# ruled-out particles a constant would let them tie with -- and on a step where
# every survivor is contradicted, outrank -- particles the step allows. Anchoring
# the floor to the worst survivor keeps the ordering right whatever scale the
# environment's likelihoods live on.
RULED_OUT_LOG_MARGIN = -1e18


def condition_log_weights_on_a_running_episode(
    log_weights: np.ndarray, ruled_out: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Floor the particles that a still-running episode has ruled out.

    Args:
        log_weights: The log-weights this step's reading produced, shape (N,).
        ruled_out: Boolean mask over the same particles. ``True`` marks a
            particle the environment would call terminal for a reason that
            varies between particles.

    Returns:
        ``(log_weights, conditioned)``. ``conditioned`` is ``False`` when the
        mask ruled out every particle or none of them, and then the weights
        are the caller's array unchanged.

    Example:
        >>> import numpy as np
        >>> weights = np.array([-1.0, -2.0, -3.0])
        >>> conditioned, acted = condition_log_weights_on_a_running_episode(
        ...     weights, np.array([False, True, False])
        ... )
        >>> acted
        True
        >>> bool(conditioned[1] < conditioned.min() + 1.0)
        True
    """
    ruled_out = np.asarray(ruled_out, dtype=bool)
    if not np.any(ruled_out) or np.all(ruled_out):
        return log_weights, False

    conditioned = np.array(log_weights, dtype=np.float64, copy=True)
    worst_survivor = float(np.min(conditioned[~ruled_out]))
    if np.isneginf(worst_survivor):
        # Every particle the step allows is also impossible under the reading,
        # so anchoring to the worst survivor would put the ruled-out particles
        # at -inf as well -- and a population that is entirely -inf is
        # normalised to *uniform*, which hands the resample the very particles
        # this function just ruled out. Measured on Discrete Light-Dark, that
        # revived roughly 180 of 400 terminal particles at full weight on the
        # steps where no particle explained the reading.
        #
        # There is no information left to keep among the survivors -- they are
        # equally impossible -- so they are levelled, which is what the
        # all--inf fallback was already doing, and the ruled-out particles go
        # below them so the fallback draws only from states the step allows.
        conditioned[~ruled_out] = 0.0
        conditioned[ruled_out] = RULED_OUT_LOG_MARGIN
        return conditioned, True
    conditioned[ruled_out] = worst_survivor + RULED_OUT_LOG_MARGIN
    return conditioned, True
