# POMDPPlanners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

POMDPPlanners is a set of reliable implementations of **POMDP (Partially Observable Markov Decision Process)** planning algorithms and environments in Python. It provides standardized simulation studies for research and production-quality planners for industrial applications — from classic benchmarks like Tiger and RockSample to photorealistic autonomous driving and robotic manipulation.

<p align="center">
  <img src="docs/images/carla_chase_camera.png" alt="CARLA autonomous driving environment rendered by CarlaPOMDP's chase camera" width="49%">
  <img src="docs/images/isaac_lab_franka_reach.png" alt="Isaac Sim Franka reach task rendered by IsaacLabPOMDP's viewport camera" width="49%">
</p>
<p align="center">
  <em>Rendered by the package itself: the <a href="POMDPPlanners/environments/carla_pomdp">CARLA</a> driving environment (left) and the <a href="POMDPPlanners/environments/isaac_lab_pomdp">Isaac Sim / IsaacLab</a> Franka reach environment (right). Realistic environments are integrated from the open-source simulators <a href="https://github.com/carla-simulator/carla">CARLA</a> and <a href="https://github.com/isaac-sim/IsaacLab">NVIDIA Isaac Lab</a> — credit to their authors.</em>
</p>

## Main Features

| **Features**                                          | **POMDPPlanners** |
| ----------------------------------------------------- | ----------------- |
| State-of-the-art online POMDP planners                | :heavy_check_mark: |
| Classic benchmarks & realistic simulator environments | :heavy_check_mark: |
| Easy to define custom environments                    | :heavy_check_mark: |
| Rich belief representations                           | :heavy_check_mark: |
| GPU-vectorized planning & belief updates              | :heavy_check_mark: |
| Risk-sensitive (CVaR) & constrained planning          | :heavy_check_mark: |
| Parallel experiment framework with persistent caching | :heavy_check_mark: |
| Hyperparameter tuning (Optuna)                        | :heavy_check_mark: |
| Progress tracking & Slack notifications               | :heavy_check_mark: |
| Tutorial notebooks                                    | :heavy_check_mark: |
| Documentation                                         | :heavy_check_mark: |
| Comprehensive test suite & type hints                 | :heavy_check_mark: |

## Documentation

Documentation is available online: https://yaacovpariente.github.io/POMDPPlanners/

## Installation

**Note:** POMDPPlanners requires Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/yaacovpariente/POMDPPlanners.git
cd POMDPPlanners

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Example

Plan with POMCP on the classic Tiger problem in just a few lines:

```python
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.utils.belief_factory import create_environment_belief

env = TigerPOMDP(discount_factor=0.95)
planner = POMCP(environment=env, discount_factor=0.95, depth=20,
                exploration_constant=10.0, n_simulations=1000,
                name="POMCP")
belief = create_environment_belief(env, n_particles=200)

actions, _ = planner.action(belief)
print(f"Recommended action: {actions[0]}")
```

## Running Experiments

The recommended entry point for end-to-end experiments is `LocalSimulationsAPI`,
which runs parallel episodes, applies persistent caching, and returns aggregated
statistics (mean return, CVaR, VaR, confidence intervals).

```python
from POMDPPlanners.environments import ContinuousLightDarkPOMDPDiscreteActions
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.utils.belief_factory import create_environment_belief
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.core.simulation import EnvironmentRunParams

env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
sampler = DiscreteActionSampler(env.get_actions())

pomcpow = POMCPOW(environment=env, discount_factor=0.95, depth=10,
                  exploration_constant=10.0, k_o=2.0, k_a=2.0,
                  alpha_o=0.5, alpha_a=0.5, n_simulations=500,
                  action_sampler=sampler, name="POMCPOW")
pft_dpw = PFT_DPW(environment=env, discount_factor=0.95, depth=10,
                  exploration_constant=10.0, n_simulations=500,
                  action_sampler=sampler, name="PFT_DPW")
belief = create_environment_belief(env, n_particles=200)

api = LocalSimulationsAPI()
_, stats = api.run_multiple_environments_and_policies(
    environment_run_params=[EnvironmentRunParams(
        environment=env, belief=belief,
        policies=[pomcpow, pft_dpw], num_episodes=100, num_steps=30)],
    alpha=0.1, confidence_interval_level=0.95,
    experiment_name="LightDark_Evaluation",
)
```

For hyperparameter search, `LocalSimulationsAPI.run_optimize_and_evaluate(...)`
accepts `HyperParameterRunParams` with Optuna search ranges and forwards the
best configuration to evaluation automatically.

Long-running experiments can report progress to Slack and a local progress
database, including detection of crashed or stalled runs — set
`SLACK_WEBHOOK_URL` in your environment and notifications are picked up
automatically. See
[`NotificationConfig`](POMDPPlanners/simulations/simulations_deployment/run_progress/config.py)
for details.

## Tutorial Notebooks

Self-contained Jupyter notebooks with executable end-to-end examples live in
[`docs/examples/`](docs/examples/):

| Notebook | What it covers |
|---|---|
| [`basic_usage.ipynb`](docs/examples/basic_usage.ipynb) | Environment setup, belief initialization, single-planner evaluation |
| [`planners_comparison.ipynb`](docs/examples/planners_comparison.ipynb) | Side-by-side comparison of POMCP / POMCPOW / PFT-DPW on a shared environment |
| [`belief_representations.ipynb`](docs/examples/belief_representations.ipynb) | Particle, Gaussian, and Gaussian-mixture beliefs |
| [`hyperparameter_tuning.ipynb`](docs/examples/hyperparameter_tuning.ipynb) | End-to-end Optuna search via `run_optimize_and_evaluate` |
| [`advanced_optimization.ipynb`](docs/examples/advanced_optimization.ipynb) | Multi-config tuning, custom search spaces |
| [`custom_environment.ipynb`](docs/examples/custom_environment.ipynb) | Implementing a new `Environment` subclass |
| [`tree_analysis_debugging.ipynb`](docs/examples/tree_analysis_debugging.ipynb) | Inspecting and debugging search trees |

## Implemented Algorithms

| **Algorithm** | **Description** |
| ------------- | --------------- |
| POMCP | Monte Carlo tree search with unweighted particle beliefs (Silver & Veness, 2010) |
| POMCP-DPW | POMCP with double progressive widening for large action/observation spaces |
| POMCPOW | Weighted-particle MCTS for continuous observation spaces (Sunberg & Kochenderfer, 2018) |
| PFT-DPW | Particle filter tree with double progressive widening (Sunberg & Kochenderfer, 2018) |
| Sparse PFT | Particle filter tree with sparse observation branching |
| Sparse Sampling | Depth-limited sparse sampling of the belief MDP (Kearns et al., 2002) |
| BetaZero | Neural-network-guided belief-state MCTS with learned policy and value |
| ConstrainedZero | Safety-constrained variant of BetaZero |
| Constrained POMCPOW / Constrained PFT-DPW | Cost-constrained online planning |
| iCVaR POMCPOW / iCVaR PFT-DPW / iCVaR Sparse Sampling | Risk-averse planning with iterated CVaR objectives |
| VOPP | Fully GPU-vectorized online POMDP planning (Hoerger et al., 2025) |
| Discrete Action Sequences | Open-loop baseline planner |

## Implemented Environments

| **Environment** | **Description** |
| --------------- | --------------- |
| Tiger | Classic information-gathering benchmark |
| Light-Dark | Navigation under state-dependent observation noise (continuous & discrete variants) |
| RockSample | Rover science mission with sensing trade-offs |
| LaserTag | Pursuit with laser range-finder observations |
| PacMan | Arcade-style pursuit-evasion with rendering |
| CartPole / MountainCar | Partially observable versions of the Gym classics |
| Push | Object manipulation under contact uncertainty |
| Safety-Ant-Velocity | Safety-constrained quadruped locomotion |
| CARLA | Photorealistic autonomous driving in the [CARLA](https://github.com/carla-simulator/carla) simulator |
| Isaac Lab | Franka reach manipulation in [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) / Isaac Sim |
| nuPlan | Autonomous driving planning on real-world driving logs |
| Sanity | Minimal environment for quick sanity checks |

Custom environments are first-class: subclass `Environment`, implement the
transition, observation, and reward models, and every planner and the whole
experiment framework work with it out of the box. See
[`custom_environment.ipynb`](docs/examples/custom_environment.ipynb).

## Belief Representations

Beliefs are pluggable and planner-independent: unweighted and weighted particle
filters, batched particle beliefs, Gaussian and Gaussian-mixture beliefs, and
GPU-vectorized particle belief updaters for large-scale simulation.

## Citing the Project

If you use POMDPPlanners in your research, please cite:

```bibtex
@misc{pariente2026pomdpplannersopensourcepackagepomdp,
      title={POMDPPlanners: Open-Source Package for POMDP Planning}, 
      author={Yaacov Pariente and Vadim Indelman},
      year={2026},
      eprint={2602.20810},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.20810}, 
}
```

## Contributing & Support

Questions, bug reports, and feature requests are welcome on the
[issue tracker](https://github.com/yaacovpariente/POMDPPlanners/issues).

## License

This project is licensed under the MIT License — see the [LICENSE.md](LICENSE.md) file for details.
