"""Tasks for POMDP simulation deployment.

This module contains task classes for distributed execution of POMDP simulations,
including episode simulation and hyperparameter optimization tasks.

Both task classes support pickling for distributed computing via __getstate__
and __setstate__ methods.
"""

from .episode_simulation_task import EpisodeSimulationTask
from .hyper_parameter_tuning_simulation_task import HyperParameterTuningSimulationTask

__all__ = ["EpisodeSimulationTask", "HyperParameterTuningSimulationTask"]
