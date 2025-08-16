import pytest
from POMDPPlanners.simulations.simulations_deployment.tasks import HyperParameterTuningSimulationTask

def test_hyper_parameter_tuning_simulation_task_creation():
    """Test that HyperParameterTuningSimulationTask can be imported."""
    assert HyperParameterTuningSimulationTask is not None
