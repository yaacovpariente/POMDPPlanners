# POMDPPlanners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for **POMDP (Partially Observable Markov Decision Process)** planning algorithms and environments. POMDPPlanners provides standardized simulation studies for research and reliable implementations of planning algorithms for industrial applications.

## 🎯 Key Features

- **Comprehensive Algorithm Library**: Implementations of state-of-the-art POMDP planning algorithms including POMCP, POMCPOW, POMCP-DPW, PFT-DPW, Sparse PFT, BetaZero, ConstrainedZero, and more
- **Rich Environment Collection**: Classic and modern POMDP environments (Tiger, Light-Dark, RockSample, LaserTag, PacMan, CartPole, Push, Safety-Ant-Velocity, etc.)
- **Flexible Belief Representations**: Particle filters, weighted beliefs, Gaussian beliefs, Gaussian mixture beliefs, and vectorized belief updaters
- **Simulation Framework**: Complete experiment management with hyperparameter tuning, high-level evaluation workflows, and distributed computing support
- **Visualization Tools**: Built-in plotting and visualization capabilities for analysis and debugging
- **Production Ready**: Designed for both research experiments and industrial applications

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yaacovpariente/POMDPPlanners.git
cd POMDPPlanners

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package (standard)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import WeightedParticleBelief

# Create environment and planner
env = TigerPOMDP()
planner = POMCP(env, num_simulations=1000)

# Initialize belief and run planning
initial_belief = WeightedParticleBelief.create_uniform_belief(
    env.get_states(), num_particles=1000
)

# Get action from planner
action = planner.get_action(initial_belief)
print(f"Recommended action: {action}")
```

## 🏗️ Architecture Overview

### Core Components

- **`POMDPPlanners.core`**: Fundamental abstractions and interfaces
  - `environment.py`: Base `Environment`, `DiscreteActionsEnvironment`, `SpaceType`, `ObservationModel`, `StateTransitionModel`
  - `policy.py`: Base `Policy` and `TrainablePolicy` classes with config management and logging
  - `belief/`: Belief state representations including:
    - `WeightedParticleBelief`, `UnweightedParticleBelief` — particle filter beliefs
    - `GaussianBelief`, `GaussianMixtureBelief` — parametric beliefs
    - `VectorizedWeightedParticleBelief` — vectorized particle belief for performance
    - Gaussian belief updaters and vectorized particle belief updaters
  - `cost.py`: Cost function abstractions for constrained planning
  - `distributions.py`: Probability distribution implementations
  - `tree.py`: Tree-based data structures for planning algorithms
  - `config_types.py`: Centralized configuration type definitions

- **`POMDPPlanners.environments`**: POMDP environment implementations
- **`POMDPPlanners.planners`**: Planning algorithm implementations
- **`POMDPPlanners.simulations`**: Experiment management and execution framework
- **`POMDPPlanners.utils`**: Helper functions and visualization tools

### Supported Algorithms

#### MCTS-Based Planners
- **POMCP**: Partially Observable Monte Carlo Planning
- **POMCPOW**: POMCP with Online Widening (progressive widening for observations)
- **POMCP-DPW**: POMCP with Double Progressive Widening
- **PFT-DPW**: Progressive Widening with Particle Filter Trees
- **Sparse PFT**: Sparse sampling with Particle Filter Trees

#### Neural MCTS Planners
- **BetaZero**: Neural-guided MCTS for POMDPs — adapts AlphaZero to belief-space planning with a dual-head neural network (value + policy)
- **ConstrainedZero**: Safety-constrained extension of BetaZero for chance-constrained POMDPs — adds a failure-probability head and adaptive threshold calibration via conformal inference

#### Other Planners
- **Sparse Sampling**: Classical sparse sampling algorithm
- **Open Loop Planners**: Non-feedback planning approaches (discrete action sequences)

### Available Environments

- **Tiger POMDP**: Classic two-door problem
- **Light-Dark POMDP**: Navigation with position-dependent observation noise (discrete and continuous variants)
- **Rock Sample POMDP**: Grid-world rover navigation with multiple rock samples; rover must sense and collect good rocks
- **Laser Tag POMDP**: Tag-based pursuit environment (discrete and continuous geometry variants)
- **PacMan POMDP**: Partially observable Pac-Man navigation with ghost uncertainty
- **CartPole POMDP**: Partially observable cart-pole balancing
- **Mountain Car POMDP**: Partially observable mountain car
- **Push POMDP**: Object manipulation environment
- **Safety Ant Velocity**: Safety-constrained locomotion task

### Simulations Framework

- **`POMDPPlanners.simulations.simulator`**: Main simulation runner
- **`POMDPPlanners.simulations.episodes`**: Episode execution logic
- **`POMDPPlanners.simulations.workflows`**: High-level evaluation and optimization workflows
  - `evaluation.py`, `optimization.py`, `integrated.py`
  - `planner_evaluation_workflow.py`, `hyperparameter_tuning_evaluation_workflows.py`
- **`POMDPPlanners.simulations.simulation_apis`**: Pluggable backends for experiment execution (see Distributed Computing section)
- **`POMDPPlanners.simulations.simulations_deployment`**: Task managers and caching for distributed runs

### Utilities

- **`POMDPPlanners.utils.visualization`**: Visualization submodule with
  - `metrics_plots.py`, `returns_plots.py`, `tree_plots.py`, `policy_simulation_plots.py`
- **`POMDPPlanners.utils.config_loader`**: Configuration file management
- **`POMDPPlanners.utils.logger`**: Centralized logging setup
- **`POMDPPlanners.utils.statistics_utils`**: Statistical analysis functions
- **`POMDPPlanners.utils.tree_statistics`**: Tree statistics computation

## 📊 Running Experiments

### Simple Experiment

```python
from POMDPPlanners.simulations.simulator import Simulator
from POMDPPlanners.utils.config_loader import load_config

# Load experiment configuration
config = load_config("experiments/configs/tiger_pomcp_experiment.yaml")

# Run simulation
simulator = Simulator(config)
results = simulator.run()

print(f"Average reward: {results['average_reward']}")
```

### Hyperparameter Tuning

```python
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterTuningSimulations

# Define hyperparameter space
hyperparams = {
    "num_simulations": [500, 1000, 2000],
    "exploration_constant": [1.0, 2.0, 5.0],
    "discount_factor": [0.9, 0.95, 0.99]
}

# Run tuning experiment
tuner = HyperParameterTuningSimulations(
    base_config="experiments/configs/base_config.yaml",
    hyperparameters=hyperparams
)
best_params = tuner.optimize()
```

### Example Notebooks

Interactive Jupyter notebooks with detailed usage examples are available in `docs/examples/`:

- `docs/examples/basic_usage.ipynb` — Environment setup, belief initialization, and basic planning
- `docs/examples/hyperparameter_tuning.ipynb` — End-to-end hyperparameter search
- `docs/examples/planners_comparison.ipynb` — Side-by-side comparison of planning algorithms

## 🌐 Distributed Computing

POMDPPlanners supports multiple execution backends for scaling experiments:

| Backend | Description | Use Case |
|---------|-------------|----------|
| **Local** | Sequential single-process | Development and debugging |
| **Dask** | Distributed multi-machine cluster | Large-scale parallelism |
| **PBS** | HPC cluster via `dask-jobqueue` | Supercomputer / PBS job scheduler |

Select the backend via the simulation API in `POMDPPlanners.simulations.simulation_apis`:

```python
# Local sequential execution
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI

# Dask distributed execution
from POMDPPlanners.simulations.simulation_apis.dask_simulations_api import DaskSimulationsAPI

# PBS (HPC) cluster execution
from POMDPPlanners.simulations.simulation_apis.pbs_simulations_api import PBSSimulationsAPI
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run specific test categories
pytest POMDPPlanners/tests/test_core/
pytest POMDPPlanners/tests/test_environments/
pytest POMDPPlanners/tests/test_planners/

# Run with verbose output
pytest -v

# Run specific test file
pytest POMDPPlanners/tests/test_core/test_belief.py
```

## 🔧 Development

### Code Quality

```bash
# Format code
black .

# Type checking
python -m pyright POMDPPlanners/

# Run linting
pylint POMDPPlanners/
flake8 .

# Install pre-commit hooks
pre-commit install
```

### Virtual Environment

**Important**: Always activate the virtual environment before development:

```bash
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

All commands should be run within this environment for consistent dependency management.

## 📚 Documentation

Comprehensive documentation is generated from docstrings using Sphinx:

```bash
# Build documentation
cd docs/
sphinx-build -b html . _build/html

# Serve locally
python -m http.server 8000 -d _build/html
```

Visit the documentation at: [Project Documentation](https://yaacovpariente.github.io/POMDPPlanners/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🎓 Citation

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

## 🛠️ Requirements

- Python 3.10 or higher
- Core dependencies managed via `pyproject.toml` (`pip install -e .`)
- Development dependencies: `pip install -e ".[dev]"`

---
