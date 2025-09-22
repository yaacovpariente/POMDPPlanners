from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter,
    CategoricalHyperParameter,
)
from POMDPPlanners.utils.hyperparameter_tuning_and_eval import HyperParamPlannerConfig


class PlannersHyperparamConfigs:
    def __init__(self, discount_factor: float):
        self.discount_factor = discount_factor

    def pft_dpw_config(
        self,
        env: Environment,
        action_sampler: ActionSampler,
        name: str,
        time_out_in_seconds: float = 3.0,
    ) -> HyperParamPlannerConfig:
        max_depth_for_tuning = 10
        hyper_parameters = [
            NumericalHyperParameter(
                0.0,
                (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning,
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(
                1, 10, "k_a"
            ),  # Action progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_a"
            ),  # Action progressive widening exponent
            NumericalHyperParameter(
                1, 10, "k_o"
            ),  # Observation progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_o"
            ),  # Observation progressive widening exponent
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
            "action_sampler": action_sampler,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=PFT_DPW,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def pomcpow_config(
        self,
        env: Environment,
        action_sampler: ActionSampler,
        name: str,
        time_out_in_seconds: float = 3,
    ) -> HyperParamPlannerConfig:
        max_depth_for_tuning = 10
        hyper_parameters = [
            NumericalHyperParameter(
                0.0,
                (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning,
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(
                1, 10, "k_a"
            ),  # Action progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_a"
            ),  # Action progressive widening exponent
            NumericalHyperParameter(
                1, 10, "k_o"
            ),  # Observation progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_o"
            ),  # Observation progressive widening exponent
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
            "action_sampler": action_sampler,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=POMCPOW,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def sparse_pft_config(
        self, env: Environment, name: str, time_out_in_seconds: float = 3
    ) -> HyperParamPlannerConfig:
        max_depth_for_tuning = 10
        exploration_constant_max = (
            env.reward_range[1] - env.reward_range[0]
        ) * max_depth_for_tuning
        hyper_parameters = [
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(
                0.0, exploration_constant_max, "c_ucb"
            ),  # UCB exploration constant
            NumericalHyperParameter(
                0.0, exploration_constant_max, "beta_ucb"
            ),  # UCB beta parameter
            NumericalHyperParameter(
                3, 15, "belief_child_num"
            ),  # Number of belief children per action
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "gamma": self.discount_factor,
            "name": name,
            "environment": env,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=SparsePFT,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def sparse_sampling_config(
        self, env: Environment, name: str, time_out_in_seconds: float = 3
    ) -> HyperParamPlannerConfig:
        hyper_parameters = [
            NumericalHyperParameter(
                3, 10, "branching_factor"
            ),  # Number of samples at each node
            NumericalHyperParameter(2, 3, "depth"),  # Search depth
            CategoricalHyperParameter(
                [True, False], "resampling"
            ),  # Whether to resample particles
        ]

        constant_parameters = {
            "environment": env,
            "name": name,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def pomcp_config(
        self, env: Environment, name: str, time_out_in_seconds: float = 3
    ) -> HyperParamPlannerConfig:
        max_depth_for_tuning = 10
        hyper_parameters = [
            NumericalHyperParameter(
                0.0,
                (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning,
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
            "min_samples_per_node": 1,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=POMCP,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def pomcp_dpw_config(
        self,
        env: Environment,
        action_sampler: ActionSampler,
        name: str,
        time_out_in_seconds: float = 3,
    ) -> HyperParamPlannerConfig:
        max_depth_for_tuning = 10
        hyper_parameters = [
            NumericalHyperParameter(
                0.0,
                (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning,
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(
                1.0, 8.0, "k_a"
            ),  # Action progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_a"
            ),  # Action progressive widening exponent
            NumericalHyperParameter(
                1.0, 8.0, "k_o"
            ),  # Observation progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_o"
            ),  # Observation progressive widening exponent
            NumericalHyperParameter(
                5, 50, "min_samples_per_node"
            ),  # Minimum samples per node
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
            "action_sampler": action_sampler,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=POMCP_DPW,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def discrete_action_sequences_config(
        self, env: Environment, name: str
    ) -> HyperParamPlannerConfig:
        hyper_parameters = [
            NumericalHyperParameter(2, 3, "depth"),  # Planning horizon
            NumericalHyperParameter(
                10, 500, "n_return_samples"
            ),  # Monte Carlo samples for return estimation
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
        }

        return HyperParamPlannerConfig(
            policy_cls=DiscreteActionSequencesPlanner,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )
