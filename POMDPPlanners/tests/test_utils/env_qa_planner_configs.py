# SPDX-License-Identifier: MIT

"""Planner configurations that were shown to solve an environment's QA gate.

These are **test inputs, not measurements.** The env-QA gate asks for evidence
that a planner can actually solve an environment and beat a random baseline; the
completion rates that evidence consists of are reported in the run and
deliberately not stored here, because a committed number goes stale silently and
starts lying. The *configuration* is what a later run needs in order to reproduce
the comparison, and that is what lives here — the same job
``env_pinned_kwargs.py`` does for environment settings.

Each function returns fresh objects on every call, so two constructions never
share a backing object.
"""

from typing import Any, Dict


def t_maze_qa_pft_dpw_kwargs(**overrides: Any) -> Dict[str, Any]:
    """PFT-DPW settings that solved the default ``TMazePOMDP``.

    Found by the env-QA search and kept as the configuration the T-Maze's
    planner-versus-random comparison is reproducible from. ``depth`` covers the
    whole task (five moves from the start to an endpoint) with room to spare
    inside the 30-step episode budget the gate ran with.

    ``time_out_in_seconds`` is a wall-clock budget, so PFT-DPW hits it exactly and
    needs no simulation-count calibration. The 0.1 s value is the budget the QA
    run was commissioned at; it sits *below* the 1-2 s band ``planner-calibration``
    recommends for MCTS planners, which is fine for a solvability gate and is not
    a setting to carry into a planner comparison without re-measuring.

    Args:
        **overrides: Values merged on top of the pinned ones (overrides win).

    Returns:
        Constructor kwargs for :class:`~POMDPPlanners.planners.mcts_planners.pft_dpw.PFT_DPW`,
        minus ``environment``, ``discount_factor``, ``name`` and ``action_sampler``,
        which the caller supplies.
    """
    pinned: Dict[str, Any] = {
        "depth": 15,
        "k_a": 4.0,
        "alpha_a": 0.0,
        "k_o": 3.0,
        "alpha_o": 0.0,
        "exploration_constant": 10.0,
        "time_out_in_seconds": 0.1,
    }
    pinned.update(overrides)
    return pinned


def t_maze_qa_belief_particles() -> int:
    """Particle count for the initial belief the T-Maze QA gate ran with."""
    return 100


def discrete_maze_qa_pft_dpw_kwargs(**overrides: Any) -> Dict[str, Any]:
    """PFT-DPW settings used by the bounded generated-Maze QA run.

    This is the best tested discrete configuration at the fixed 0.1-second
    decision budget. It is a reproducible QA input, not a claim that every maze
    size or seed is solved.
    """
    pinned: Dict[str, Any] = {
        "depth": 20,
        "k_a": 4.0,
        "alpha_a": 0.0,
        "k_o": 3.0,
        "alpha_o": 0.0,
        "exploration_constant": 10.0,
        "time_out_in_seconds": 0.1,
    }
    pinned.update(overrides)
    return pinned


def continuous_maze_qa_pft_dpw_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Best bounded continuous-Maze PFT-DPW configuration tested in QA.

    The caller must use a sampler restricted to the four unit cardinal moves.
    Deeper search with 200 particles did not improve the bounded probe, so this
    keeps the cheaper depth-20, 100-particle setting. It is not a success claim.
    """
    return discrete_maze_qa_pft_dpw_kwargs(**overrides)


def maze_qa_belief_particles() -> int:
    """Particle count used by both generated-Maze QA configurations."""
    return 100



def occupancy_grid_mapping_qa_pft_dpw_kwargs(**overrides: Any) -> Dict[str, Any]:
    """PFT-DPW settings used for the occupancy-grid mapping QA and smoke run.

    The environment has no torch vectorized model, so QA runs PFT-DPW on the
    scalar ``Environment`` API rather than VOPP.

    Both widening constants are set to *disable* widening. Progressive widening
    exists to keep a large or continuous branching factor manageable, and the
    action side has three actions, so ``k_a = 3`` with ``alpha_a = 0`` admits
    every one of them and no action is excluded by an accident of the sampler.
    The observation side is genuinely continuous -- a 24-beam range vector, so
    every simulated reading is unique -- and there ``k_o = 3`` with
    ``alpha_o = 0`` is doing real work: it caps each action node at three
    observation children so the particle sets under them get revisited instead
    of the tree spending its whole budget on one-visit leaves.

    ``depth`` is 10 against a 40-step episode. It is short on purpose: one scan
    resolves the cells around the robot, so the information a plan can still
    gain falls away quickly with depth, and spending the budget on width rather
    than length is the better trade here. It is a QA input, not a tuned value.

    Args:
        **overrides: Values merged on top of the pinned ones (overrides win).

    Returns:
        Constructor kwargs for :class:`~POMDPPlanners.planners.mcts_planners.pft_dpw.PFT_DPW`,
        minus ``environment``, ``discount_factor``, ``name`` and ``action_sampler``,
        which the caller supplies.
    """
    pinned: Dict[str, Any] = {
        "depth": 10,
        "k_a": 3.0,
        "alpha_a": 0.0,
        "k_o": 3.0,
        "alpha_o": 0.0,
        "exploration_constant": 5.0,
        "time_out_in_seconds": 1,
    }
    pinned.update(overrides)
    return pinned


def occupancy_grid_mapping_qa_belief_particles() -> int:
    """Thirty whole-map particles: a bounded QA input, with no diversity guarantee."""
    return 30


__all__ = [
    "continuous_maze_qa_pft_dpw_kwargs",
    "discrete_maze_qa_pft_dpw_kwargs",
    "maze_qa_belief_particles",
    "occupancy_grid_mapping_qa_belief_particles",
    "occupancy_grid_mapping_qa_pft_dpw_kwargs",
    "t_maze_qa_belief_particles",
    "t_maze_qa_pft_dpw_kwargs",
]
