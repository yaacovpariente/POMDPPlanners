# SPDX-License-Identifier: MIT

"""Narrowing a :class:`Belief` to the particle interface a test exercises.

``create_environment_belief`` is declared to return :class:`Belief`, the
abstract base, because an environment may answer with a Gaussian, a mixture or
particles. The particle classes do not share a base below ``Belief`` — each
subclasses it directly — so a test that reaches for ``particles`` or
``normalized_weights`` is reading attributes the declared type does not have,
and a type checker is right to say so.

Rather than silence that per line, a test says once which shape it asked for.
:func:`particle_belief` checks it at runtime and narrows it for the checker,
so a test that is handed the wrong belief fails with a sentence instead of an
``AttributeError`` twenty lines later.
"""

from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from POMDPPlanners.core.belief import Belief


@runtime_checkable
class ParticleBelief(Protocol):
    """The part of a particle belief these tests read and write.

    Structural on purpose: five classes in ``core.belief`` satisfy it —
    weighted, unweighted, state-update, batched and vectorized — and they
    share no base of their own.
    """

    particles: Any
    log_weights: np.ndarray
    normalized_weights: np.ndarray
    # Narrowing must not cost a test the rest of the Belief interface, so the
    # two members these tests go on to use are declared here as well.
    updater: Any

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Advance the belief; see :meth:`Belief.update`."""


def particle_belief(belief: Belief) -> ParticleBelief:
    """Return *belief* as a particle belief.

    Args:
        belief: The belief an environment or factory produced.

    Returns:
        The same object, typed as a particle belief.

    Raises:
        AssertionError: If it carries no particles, which means the test asked
            for a belief type this assertion does not hold for.
    """
    assert hasattr(belief, "particles"), (
        f"{type(belief).__name__} is not a particle belief; this test reads its "
        "particles and weights directly"
    )
    return cast(ParticleBelief, belief)
