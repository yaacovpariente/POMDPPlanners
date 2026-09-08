# SPDX-License-Identifier: MIT

"""Determinized random-number streams for scenario-based planners.

A DESPOT *scenario* is a start state paired with a fixed sequence of random
numbers, one per depth (Ye et al., JAIR 2017, Sec. 3). Two branches of the
search that share a scenario and sit at the same depth must consume the *same*
random number, even though they carry different states. That is what makes the
tree "determinized": with ``K`` scenarios fixed, the belief tree is a
deterministic function of the policy, and the paper's regret bound is stated
over exactly those ``K`` draws.

The repository's :meth:`Environment.sample_next_step` takes no random generator
-- it draws from the process-global ``numpy.random`` and ``random`` streams.
So the only way to pin a transition to ``(scenario, depth)`` without changing
the ``Environment`` API for every environment is to seed those two global
generators immediately before each call. :class:`ScenarioRandomStreams` does
that from one base seed, and restores whatever state the caller had when the
planning call finishes, so a planner does not silently reshape the simulator's
own random stream.

Limits, stated rather than hidden:

* An environment whose generative model runs in a native kernel with its own
  generator (the ``_native`` C++ paths in this repository) is **not** pinned by
  this class. Its transitions stay independent draws, which makes the search a
  sparse-sampling tree rather than a determinized one. Planners expose a flag
  to turn determinization off explicitly for those environments instead of
  claiming a property they do not have.
* Seeding costs roughly 8 microseconds per transition (2.4 for
  ``numpy.random.seed`` plus 6 for ``random.seed``, measured on this machine).
  On a cheap environment that dominates the transition itself; on any
  realistic one it does not.
"""

import random
from typing import Any, Tuple

import numpy as np


#: Upper bound for the seeds handed to ``numpy.random.seed``, which rejects
#: anything outside the unsigned 32-bit range.
_SEED_MODULUS: int = 2**32


class ScenarioRandomStreams:
    """Fixed random numbers indexed by ``(scenario id, depth)``.

    Args:
        n_scenarios: Number of scenarios ``K``. Scenario ids are
            ``0 .. n_scenarios - 1``.
        max_depth: Largest depth that will be requested. Depths are
            ``0 .. max_depth``; the table is sized ``max_depth + 1`` deep so a
            rollout that runs to the search depth inclusive stays in range.
        base_seed: Seed for the table itself. Two instances built with the same
            ``base_seed``, ``n_scenarios`` and ``max_depth`` produce identical
            seeds, so a whole planning call is reproducible.

    Raises:
        ValueError: If ``n_scenarios`` or ``max_depth + 1`` is not positive.
    """

    def __init__(self, n_scenarios: int, max_depth: int, base_seed: int) -> None:
        if n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be positive, got {n_scenarios}")
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")

        self.n_scenarios = int(n_scenarios)
        self.max_depth = int(max_depth)
        self.base_seed = int(base_seed)

        # One 32-bit seed per (scenario, depth). Drawn once from a private
        # generator so building the table never touches the global streams the
        # table is about to control.
        table_rng = np.random.default_rng(self.base_seed)
        self._seeds: np.ndarray = table_rng.integers(
            low=0,
            high=_SEED_MODULUS,
            size=(self.n_scenarios, self.max_depth + 1),
            dtype=np.uint32,
        )

    def seed_for(self, scenario_id: int, depth: int) -> int:
        """Return the fixed seed for ``(scenario_id, depth)``.

        Depths past ``max_depth`` wrap onto the last row rather than raising:
        a default-policy rollout may be asked to run one step past the search
        depth, and a wrapped-but-fixed number is still determinized, whereas an
        exception would abort a legitimate search.
        """
        row = scenario_id % self.n_scenarios
        column = depth if depth <= self.max_depth else self.max_depth
        return int(self._seeds[row, column])

    def activate(self, scenario_id: int, depth: int) -> None:
        """Point both global generators at the number for ``(scenario_id, depth)``.

        Call immediately before the single generative-model call whose
        randomness is being pinned. Seeding both generators is deliberate:
        environments in this repository draw from ``numpy.random`` (Tiger,
        Sanity) and from the standard library's ``random`` (discrete
        Light-Dark), and the planner cannot tell which one an arbitrary
        environment will reach for.
        """
        seed = self.seed_for(scenario_id=scenario_id, depth=depth)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def capture_global_state() -> Tuple[Any, Any]:
        """Snapshot both global generators so a planning call can put them back."""
        return np.random.get_state(), random.getstate()

    @staticmethod
    def restore_global_state(state: Tuple[Any, Any]) -> None:
        """Restore a snapshot from :meth:`capture_global_state`.

        Without this, a planner that seeds the global generators for its own
        scenarios would leave the episode runner drawing from a stream fixed by
        the planner's last transition -- the episode's randomness would become
        a function of the planner's internals.
        """
        numpy_state, python_state = state
        np.random.set_state(numpy_state)
        random.setstate(python_state)
