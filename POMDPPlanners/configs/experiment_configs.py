from typing import List, Tuple, Optional, Sequence, Type
from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator, EnvironmentRunParams
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    ParameterToOptimizeMapper,
    HyperParamPlannerConfig,
)
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.configs.environment_configs import (
    EnvironmentConfigsAPI,
    RiskAverseEnvironmentConfigsAPI,
)
from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation.hyperparameter_tuning import HyperParamPlannerConfigGenerator
from POMDPPlanners.core.simulation.simulation_configs import (
    HyperparameterOptimizationExperimentConfigCreator,
    EvaluationExperimentConfigCreator,
)

from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP


class AverageReturnParameterToOptimizeMapper(ParameterToOptimizeMapper):
    def generate(
        self, environment: Environment, policy_cls: Optional[Type[Policy]] = None
    ) -> List[Tuple[str, HyperParameterOptimizationDirection]]:
        return [("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]


class RiskAverseParameterToOptimizeMapper(ParameterToOptimizeMapper):
    def generate(
        self, environment: Environment, policy_cls: Optional[Type[Policy]] = None
    ) -> List[Tuple[str, HyperParameterOptimizationDirection]]:
        if isinstance(environment, CartPolePOMDP):
            raise NotImplementedError("Risk-averse optimization is not supported for CartPolePOMDP")
        elif isinstance(environment, MountainCarPOMDP):
            raise NotImplementedError(
                "Risk-averse optimization is not supported for MountainCarPOMDP"
            )
        elif isinstance(environment, ContinuousLightDarkPOMDP):
            return [
                ("avg_obstacle_hit_counter", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, DiscreteLightDarkPOMDP):
            return [
                ("avg_obstacle_hit_counter", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, PushPOMDP):
            return [
                ("total_all_obstacle_collisions", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, SafeAntVelocityPOMDP):
            return [
                ("total_safety_violations", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, TigerPOMDP):
            raise NotImplementedError("Risk-averse optimization is not supported for TigerPOMDP")
        elif isinstance(environment, RockSamplePOMDP):
            return [
                ("average_dangerous_area_steps", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, LaserTagPOMDP):
            return [
                ("average_all_dangerous_encounters", HyperParameterOptimizationDirection.MINIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        elif isinstance(environment, PacManPOMDP):
            return [
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
            ]
        else:
            raise ValueError(f"Environment {environment.__class__.__name__} is not supported")


def get_hyperparameter_benchmarks(
    policy_space_info: PolicySpaceInfo, particle_count: int = 30, time_out_in_seconds: float = 3.0
) -> List[Tuple[Environment, Belief, List[HyperParamPlannerConfig]]]:
    env_configs = EnvironmentConfigsAPI(discount_factor=0.95)
    planners_hyperparam_configs = PlannersHyperparamConfigs(discount_factor=0.95)

    envs = env_configs.get_compatible_environments(
        policy_space_info=policy_space_info, n_particles=particle_count
    )
    benchmarks = []
    for env, belief in envs:
        planner_confs = planners_hyperparam_configs.get_compatible_planners(
            env=env, time_out_in_seconds=time_out_in_seconds
        )
        planner_configs = [
            HyperParamPlannerConfig(
                policy_cls=planner_conf.policy_cls,
                hyper_parameters=planner_conf.hyper_parameters,
                constant_parameters=planner_conf.constant_parameters,
            )
            for planner_conf in planner_confs
        ]
        benchmarks.append((env, belief, planner_configs))

    return benchmarks


class AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
    EvaluationExperimentConfigCreator
):
    def __init__(
        self,
        generators: Sequence[PlannerGenerator],
        n_particles: int,
        num_episodes: int,
        num_steps: int,
        is_risk_averse: bool,
    ):
        self.generators = generators
        self.n_particles = n_particles
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.is_risk_averse = is_risk_averse

    def _get_experiment_configs(self) -> List[EnvironmentRunParams]:
        if self.is_risk_averse:
            environment_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)
        else:
            environment_configs = EnvironmentConfigsAPI(discount_factor=0.95)

        configs_dict = {}

        for generator in self.generators:
            envs = environment_configs.get_compatible_environments(
                policy_space_info=generator.get_planner_space_info(),
                n_particles=self.n_particles,
            )

            for env, belief in envs:
                env_id = env.config_id
                policy = generator.generate(env)

                if env_id in configs_dict:
                    configs_dict[env_id].policies.append(policy)
                else:
                    configs_dict[env_id] = EnvironmentRunParams(
                        environment=env,
                        belief=belief,
                        policies=[policy],
                        num_episodes=self.num_episodes,
                        num_steps=self.num_steps,
                    )

        return list(configs_dict.values())


class AllHyperparameterBenchmarksExperimentConfigCreator(
    HyperparameterOptimizationExperimentConfigCreator
):
    """Experiment configuration creator for all hyperparameter benchmarks.

    This class creates hyperparameter optimization experiment configurations for all
    compatible environments and planners based on a given policy space. It automatically
    finds all environments that match the specified action and observation space types,
    and generates configurations for hyperparameter tuning experiments.

    Attributes:
        policy_space_info: Policy space information specifying action and observation
            space types for compatibility matching.
        particles: Number of particles for belief representation.
        num_episodes: Number of episodes for optimization.
        num_steps: Maximum steps per episode for optimization.
        n_trials: Number of optimization trials.
        discount_factor: Discount factor for the MDP.
        time_out_in_seconds: Timeout for planner execution.
        is_risk_averse: Whether to use risk-averse optimization metrics.
        parameter_to_optimize_mapper: Mapper for determining optimization parameters
            based on environment and risk-averse settings.

    Example:
        >>> from POMDPPlanners.core.policy import PolicySpaceInfo
        >>> from POMDPPlanners.core.environment import SpaceType
        >>> from POMDPPlanners.configs.experiment_configs import (
        ...     AllHyperparameterBenchmarksExperimentConfigCreator
        ... )
        >>>
        >>> # Create policy space info for discrete environments
        >>> space_info = PolicySpaceInfo(
        ...     action_space=SpaceType.DISCRETE,
        ...     observation_space=SpaceType.DISCRETE
        ... )
        >>>
        >>> # Create experiment config creator
        >>> creator = AllHyperparameterBenchmarksExperimentConfigCreator(
        ...     policy_space_info=space_info,
        ...     particles=10,
        ...     num_episodes=2,
        ...     num_steps=3,
        ...     n_trials=5,
        ...     discount_factor=0.95,
        ...     time_out_in_seconds=3.0,
        ...     is_risk_averse=False
        ... )
        >>>
        >>> # Get experiment configurations
        >>> configs = creator.get_experiment_configs()
        >>>
        >>> # Verify configurations
        >>> len(configs) > 0
        True
        >>> all(config.num_episodes == 2 for config in configs)
        True
        >>> all(config.num_steps == 3 for config in configs)
        True
        >>> all(config.n_trials == 5 for config in configs)
        True
    """

    def __init__(
        self,
        policy_space_info: PolicySpaceInfo,
        particles: int,
        num_episodes: int,
        num_steps: int,
        n_trials: int,
        discount_factor: float,
        time_out_in_seconds: float,
        is_risk_averse: bool,
        debug: bool = False,
    ):
        self.policy_space_info = policy_space_info
        self.particles = particles
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.n_trials = n_trials
        self.discount_factor = discount_factor
        self.time_out_in_seconds = time_out_in_seconds
        self.is_risk_averse = is_risk_averse
        self.debug = debug

        if self.is_risk_averse:
            self.parameter_to_optimize_mapper = RiskAverseParameterToOptimizeMapper()
        else:
            self.parameter_to_optimize_mapper = AverageReturnParameterToOptimizeMapper()

        if self.debug:
            self.num_episodes = 2
            self.num_steps = 2
            self.n_trials = 2

    def _get_experiment_configs(self) -> List[HyperParameterRunParams]:
        """Generate hyperparameter optimization configurations for all compatible environments.

        Returns:
            List of HyperParameterRunParams configurations, one for each compatible
            environment-planner combination.
        """
        if self.is_risk_averse:
            env_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=self.discount_factor)
        else:
            env_configs = EnvironmentConfigsAPI(discount_factor=self.discount_factor)

        envs = env_configs.get_compatible_environments(
            policy_space_info=self.policy_space_info, n_particles=self.particles
        )

        planners_hyperparam_configs = PlannersHyperparamConfigs(
            discount_factor=self.discount_factor
        )

        planner_run_params_for_each_environment: List[HyperParameterRunParams] = []
        for env, belief in envs:
            planners_configs = planners_hyperparam_configs.get_compatible_planners(
                env=env, time_out_in_seconds=self.time_out_in_seconds
            )
            for planner_config in planners_configs:
                params_to_optimize = self.parameter_to_optimize_mapper.generate(
                    env, planner_config.policy_cls
                )
                planner_run_params_for_each_environment.append(
                    HyperParameterRunParams(
                        environment=env,
                        belief=belief,
                        hyper_param_planner_config=planner_config,
                        num_episodes=self.num_episodes,
                        num_steps=self.num_steps,
                        n_trials=self.n_trials,
                        parameters_to_optimize=params_to_optimize,
                    )
                )

        return planner_run_params_for_each_environment


class PolicyHyperparameterOptimizationExperimentConfigCreator(
    HyperparameterOptimizationExperimentConfigCreator
):
    def __init__(
        self,
        generators: Sequence[HyperParamPlannerConfigGenerator],
        particles: int,
        num_episodes: int,
        num_steps: int,
        n_trials: int,
        discount_factor: float,
        time_out_in_seconds: float,
        is_risk_averse: bool,
        debug: bool = False,
    ):
        self.generators = generators
        self.particles = particles
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.n_trials = n_trials
        self.discount_factor = discount_factor
        self.time_out_in_seconds = time_out_in_seconds
        self.is_risk_averse = is_risk_averse
        self.debug = debug

        if self.debug:
            self.num_episodes = 2
            self.num_steps = 2
            self.n_trials = 2

        if self.is_risk_averse:
            self.parameter_to_optimize_mapper = RiskAverseParameterToOptimizeMapper()
        else:
            self.parameter_to_optimize_mapper = AverageReturnParameterToOptimizeMapper()

    def _get_experiment_configs(self) -> List[HyperParameterRunParams]:
        return complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=self.generators,
            parameter_to_optimize_mapper=self.parameter_to_optimize_mapper,
            particles=self.particles,
            num_episodes=self.num_episodes,
            num_steps=self.num_steps,
            n_trials=self.n_trials,
            discount_factor=self.discount_factor,
            time_out_in_seconds=self.time_out_in_seconds,
            is_risk_averse=self.is_risk_averse,
        )


def complete_environments_and_benchmarks_hyperparameter_optimization_configs(
    generators: Sequence[HyperParamPlannerConfigGenerator],
    parameter_to_optimize_mapper: ParameterToOptimizeMapper,
    particles: int = 30,
    num_episodes: int = 10,
    num_steps: int = 20,
    n_trials: int = 500,
    discount_factor: float = 0.95,
    time_out_in_seconds: float = 3.0,
    is_risk_averse: bool = False,
) -> List[HyperParameterRunParams]:
    if is_risk_averse:
        env_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=discount_factor)
    else:
        env_configs = EnvironmentConfigsAPI(discount_factor=discount_factor)

    planners_hyperparam_configs = PlannersHyperparamConfigs(discount_factor=discount_factor)

    planner_configs_for_each_environment: List[HyperParamPlannerConfig] = []
    all_envs = []
    all_beliefs = []

    for gen in generators:
        envs = env_configs.get_compatible_environments(
            policy_space_info=gen.get_planner_space_info(), n_particles=particles
        )
        envs, beliefs = zip(*envs)
        all_envs.extend(envs)
        all_beliefs.extend(beliefs)
        configs: List[HyperParamPlannerConfig] = [gen.generate(env) for env in envs]
        planner_configs_for_each_environment += configs

    planner_run_params_for_each_environment: List[HyperParameterRunParams] = []
    for env, belief, planner_config in zip(
        all_envs, all_beliefs, planner_configs_for_each_environment
    ):
        params_to_optimize = parameter_to_optimize_mapper.generate(env, planner_config.policy_cls)
        planner_run_params_for_each_environment.append(
            HyperParameterRunParams(
                environment=env,
                belief=belief,
                hyper_param_planner_config=planner_config,
                num_episodes=num_episodes,
                num_steps=num_steps,
                n_trials=n_trials,
                parameters_to_optimize=params_to_optimize,
            )
        )

    all_configs = []
    for run_param in planner_run_params_for_each_environment:
        all_configs.append(run_param)
        all_configs += get_benchmarks_hyperparameter_optimization_configs(
            conf=run_param, discount_factor=discount_factor, time_out_in_seconds=time_out_in_seconds
        )

    return all_configs


def get_benchmarks_hyperparameter_optimization_configs(
    conf: HyperParameterRunParams, discount_factor: float, time_out_in_seconds: float = 3.0
) -> List[HyperParameterRunParams]:
    planners_hyperparam_configs = PlannersHyperparamConfigs(discount_factor=discount_factor)
    planner_configs = planners_hyperparam_configs.get_compatible_planners(
        env=conf.environment, time_out_in_seconds=time_out_in_seconds
    )
    return [
        HyperParameterRunParams(
            environment=conf.environment,
            belief=conf.belief,
            hyper_param_planner_config=bench_planner_config,
            num_episodes=conf.num_episodes,
            num_steps=conf.num_steps,
            n_trials=conf.n_trials,
            parameters_to_optimize=conf.parameters_to_optimize,
        )
        for bench_planner_config in planner_configs
    ]
