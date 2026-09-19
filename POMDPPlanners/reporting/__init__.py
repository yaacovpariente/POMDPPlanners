# SPDX-License-Identifier: MIT

"""A local results site for POMDPPlanners simulation runs.

MLflow stays the tracking store. This package only reads it, and presents what
it finds as experiments to runs to environments to planners to episodes, with
an episode page that plays whatever the episode actually produced.
"""
