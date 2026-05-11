from typing import List, Optional

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParameterFeature,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler, UnitCircleActionSampler
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType


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
        hyper_parameters: List[HyperParameterFeature] = [
            NumericalHyperParameter(
                0.0,
                self._get_exploration_constant_max(env, max_depth_for_tuning),
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(1, 10, "k_a"),  # Action progressive widening coefficient
            NumericalHyperParameter(0.01, 0.5, "alpha_a"),  # Action progressive widening exponent
            NumericalHyperParameter(1, 10, "k_o"),  # Observation progressive widening coefficient
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
                self._get_exploration_constant_max(env, max_depth_for_tuning),
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(1, 10, "k_a"),  # Action progressive widening coefficient
            NumericalHyperParameter(0.01, 0.5, "alpha_a"),  # Action progressive widening exponent
            NumericalHyperParameter(1, 10, "k_o"),  # Observation progressive widening coefficient
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
        exploration_constant_max = self._get_exploration_constant_max(env, max_depth_for_tuning)
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
            "name": name,
            "environment": env,
            "time_out_in_seconds": time_out_in_seconds,
        }

        return HyperParamPlannerConfig(
            policy_cls=SparsePFT,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def sparse_sampling_config(self, env: Environment, name: str) -> HyperParamPlannerConfig:
        hyper_parameters = [
            NumericalHyperParameter(3, 10, "branching_factor"),  # Number of samples at each node
            NumericalHyperParameter(2, 3, "depth"),  # Search depth
        ]

        constant_parameters = {
            "environment": env,
            "name": name,
        }

        return HyperParamPlannerConfig(
            policy_cls=SparseSamplingDiscreteActionsPlanner,
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
                self._get_exploration_constant_max(env, max_depth_for_tuning),
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "name": name,
            "environment": env,
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
                self._get_exploration_constant_max(env, max_depth_for_tuning),
                "exploration_constant",
            ),  # UCB1 exploration
            NumericalHyperParameter(2, max_depth_for_tuning, "depth"),  # Search depth
            NumericalHyperParameter(1.0, 8.0, "k_a"),  # Action progressive widening coefficient
            NumericalHyperParameter(0.01, 0.5, "alpha_a"),  # Action progressive widening exponent
            NumericalHyperParameter(
                1.0, 8.0, "k_o"
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

    def _get_exploration_constant_max(self, env: Environment, max_depth_for_tuning: int) -> float:
        if env.reward_range is not None:
            return (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning

        return 1.0 * max_depth_for_tuning

    def get_compatible_planners(
        self, env: Environment, time_out_in_seconds: float = 3.0
    ) -> List[HyperParamPlannerConfig]:
        """Get all planners that are compatible with the given environment.

        This function analyzes the environment's space information and returns
        a list of configured planners that can solve this environment. The compatibility
        is determined by checking if the planner's space requirements match the
        environment's space types, following the logic from Policy._verify_environment_compatibility.

        Args:
            env: The POMDP environment to find compatible planners for
            time_out_in_seconds: Time limit for each planner. Defaults to 3.0.

        Returns:
            List of HyperParamPlannerConfig objects for compatible planners

        Note:
            - For discrete action spaces, uses DiscreteActionSampler
            - For continuous action spaces, uses UnitCircleActionSampler
            - Only includes planners that can handle the environment's space types
        """
        compatible_planners = []
        env_space_info = env.space_info

        # Always compatible planners (handle MIXED spaces)
        # POMCPOW - supports MIXED action and observation spaces
        action_sampler = self._get_action_sampler_for_environment(env)
        if action_sampler is not None:
            compatible_planners.append(
                self.pomcpow_config(env, action_sampler, f"POMCPOW_{env.name}", time_out_in_seconds)
            )

            # POMCP_DPW - supports MIXED action and observation spaces
            compatible_planners.append(
                self.pomcp_dpw_config(
                    env, action_sampler, f"POMCP_DPW_{env.name}", time_out_in_seconds
                )
            )

            # PFT_DPW - requires CONTINUOUS action space but supports MIXED observation
            if env_space_info.action_space in [SpaceType.CONTINUOUS, SpaceType.MIXED]:
                compatible_planners.append(
                    self.pft_dpw_config(
                        env, action_sampler, f"PFT_DPW_{env.name}", time_out_in_seconds
                    )
                )

        # Discrete action space planners
        if env_space_info.action_space in [SpaceType.DISCRETE, SpaceType.MIXED]:
            # POMCP - requires DISCRETE actions and DISCRETE observations
            if env_space_info.observation_space in [SpaceType.DISCRETE, SpaceType.MIXED]:
                compatible_planners.append(
                    self.pomcp_config(env, f"POMCP_{env.name}", time_out_in_seconds)
                )

            # Sparse PFT - requires DISCRETE actions, supports MIXED observations
            compatible_planners.append(
                self.sparse_pft_config(env, f"SparsePFT_{env.name}", time_out_in_seconds)
            )

            # Sparse Sampling - requires DISCRETE actions, supports MIXED observations
            compatible_planners.append(
                self.sparse_sampling_config(env, f"SparseSampling_{env.name}")
            )

            # Discrete Action Sequences - requires DISCRETE actions, supports MIXED observations
            compatible_planners.append(
                self.discrete_action_sequences_config(env, f"DiscreteActionSequences_{env.name}")
            )

        return compatible_planners

    def _get_action_sampler_for_environment(self, env: Environment) -> Optional[ActionSampler]:
        """Get appropriate action sampler for the environment's action space.

        Args:
            env: The environment to get action sampler for

        Returns:
            ActionSampler instance or None if environment doesn't support action sampling
        """
        if env.space_info.action_space == SpaceType.DISCRETE:
            # For discrete action spaces, use DiscreteActionSampler
            if isinstance(env, DiscreteActionsEnvironment):
                return DiscreteActionSampler(env.get_actions())
            # If environment doesn't have get_actions method, can't create sampler
            return None
        if env.space_info.action_space == SpaceType.CONTINUOUS:
            # For continuous action spaces, use UnitCircleActionSampler
            return UnitCircleActionSampler()
        if env.space_info.action_space == SpaceType.MIXED:
            # For mixed action spaces, prefer continuous sampler if available
            # This is a reasonable default for mixed spaces
            return UnitCircleActionSampler()

        return None
