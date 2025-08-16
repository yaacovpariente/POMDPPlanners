import pytest
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask

def test_episode_simulation_task_creation():
    """Test that EpisodeSimulationTask can be imported."""
    assert EpisodeSimulationTask is not None
