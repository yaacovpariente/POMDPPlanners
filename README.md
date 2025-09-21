# POMDPPlanners

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for **POMDP (Partially Observable Markov Decision Process)** planning algorithms and environments. POMDPPlanners provides standardized simulation studies for research and reliable implementations of planning algorithms for industrial applications.

## 🎯 Key Features

- **Comprehensive Algorithm Library**: Implementations of state-of-the-art POMDP planning algorithms including POMCP, PFT-DPW, Sparse PFT, and more
- **Rich Environment Collection**: Classic and modern POMDP environments (Tiger, Light-Dark, CartPole, Push, Safety-Ant-Velocity, etc.)
- **Flexible Belief Representations**: Particle filters, weighted beliefs, and custom belief state implementations
- **Simulation Framework**: Complete experiment management with hyperparameter tuning and distributed computing support
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

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode
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

- **`POMDPPlanners.core`**: Fundamental abstractions (Environment, Policy, Belief, Distributions)
- **`POMDPPlanners.environments`**: POMDP environment implementations
- **`POMDPPlanners.planners`**: Planning algorithm implementations
- **`POMDPPlanners.simulations`**: Experiment management and execution framework
- **`POMDPPlanners.utils`**: Helper functions and visualization tools

### Supported Algorithms

#### MCTS-Based Planners
- **POMCP**: Partially Observable Monte Carlo Planning
- **PFT-DPW**: Progressive Widening with Particle Filter Trees
- **Sparse PFT**: Sparse sampling with Particle Filter Trees

#### Other Planners
- **Sparse Sampling**: Classical sparse sampling algorithm
- **Open Loop Planners**: Non-feedback planning approaches

### Available Environments

- **Tiger POMDP**: Classic two-door problem
- **Light-Dark POMDP**: Navigation with position-dependent observation noise
- **CartPole POMDP**: Partially observable cart-pole balancing
- **Mountain Car POMDP**: Partially observable mountain car
- **Push POMDP**: Object manipulation environment
- **Safety Ant Velocity**: Safety-constrained locomotion task

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

### Visualization Examples

The repository includes several visualization examples:

```bash
# Run visualization demos
python light_dark_pomdp_visualization_example.py
python pomcpow_rock_sample_visualization_demo.py
python push_pomdp_visualization_example.py
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

Visit the documentation at: [Project Documentation](https://yaacovpariente.github.io/POMDPPlanners/) *(available when repository becomes public)*

## 🤝 Contributing

We welcome contributions of all kinds! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Coding guidelines and standards
- Submitting pull requests
- Reporting bugs and requesting features

Please also read our [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🎓 Citation

If you use POMDPPlanners in your research, please cite:

```bibtex
@software{pomdpplanners2025,
  title={POMDPPlanners: A Python Package for POMDP Planning Algorithms},
  author={Pariente, Yaacov},
  year={2025},
  url={https://github.com/yaacovpariente/POMDPPlanners}
}
```

## 🛠️ Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`
- Optional: Development dependencies in `requirements-dev.txt`

## 🔗 Links

- **Repository**: https://github.com/yaacovpariente/POMDPPlanners
- **Documentation**: https://yaacovpariente.github.io/POMDPPlanners/ *(available when repository becomes public)*
- **Issues**: https://github.com/yaacovpariente/POMDPPlanners/issues
- **Discussions**: https://github.com/yaacovpariente/POMDPPlanners/discussions

---

**Made with ❤️ for the POMDP research community**