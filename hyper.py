from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType

api = LocalSimulationsAPI(debug=True)

api.run_all_hyperparameter_benchmarks(
    policy_space_info=PolicySpaceInfo(
        action_space=SpaceType.MIXED,
        observation_space=SpaceType.MIXED,
    ),
    particles=30,
    num_episodes=3,
    num_steps=3,
    n_trials=3,
    discount_factor=0.95,
    time_out_in_seconds=0.1,
    evaluation_episodes=3,
    evaluation_steps=2,
    evaluation_n_jobs=-1,
    optimization_n_jobs=-1,
    is_risk_averse=True,
    confidence_interval_level=0.95,
    alpha=0.05,
    cache_dir_path=None,
    experiment_name="All_Hyperparameter_Benchmarks",
    debug=False,
    cache_visualizations=True,
)
