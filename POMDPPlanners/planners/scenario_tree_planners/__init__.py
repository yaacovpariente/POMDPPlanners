# SPDX-License-Identifier: MIT

"""Scenario-based belief-tree planners.

These planners fix a set of ``K`` *scenarios* -- a start state plus a
predetermined sequence of random numbers -- before the search begins, and build
a sparse belief tree in which every node is represented by the subset of those
scenarios that reach it. Search is driven by upper and lower bounds on the
node's value rather than by visit counts and sampled returns, so there is no
UCB rule and no incremental averaging of rollout returns.

That is what separates this family from ``mcts_planners``. POMCP and its
relatives draw a fresh state at every step of every simulation, so two
simulations through the same history see different futures; here the futures
are fixed up front and the tree is a deterministic function of the policy under
those ``K`` draws, which is what the DESPOT regret bound is stated over.

Members:
    DESPOT: Determinized Sparse Partially Observable Tree search.
    ARDESPOT: Anytime Regularized DESPOT -- the same scenario tree, with the
        regularization constant charged inside the search rather than once at
        the end, and the wall clock as a first-class stopping condition.
"""

from POMDPPlanners.planners.scenario_tree_planners.ardespot import (
    ARDESPOT,
    ARDESPOTMetrics,
)
from POMDPPlanners.planners.scenario_tree_planners.adaops import AdaOPS, AdaOPSMetrics
from POMDPPlanners.planners.scenario_tree_planners.despot import DESPOT, DESPOTMetrics

__all__ = [
    "DESPOT",
    "DESPOTMetrics",
    "ARDESPOT",
    "ARDESPOTMetrics",
    "AdaOPS",
    "AdaOPSMetrics",
]
