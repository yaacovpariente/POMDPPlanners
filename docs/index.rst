POMDPPlanners Documentation
============================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

POMDPPlanners is a comprehensive Python package for **POMDP (Partially Observable Markov Decision Process)** planning algorithms and environments. It provides standardized simulation studies for research and reliable implementations of planning algorithms for industrial applications.

🎯 Key Features
---------------

- **Comprehensive Algorithm Library**: State-of-the-art POMDP planning algorithms including POMCP, PFT-DPW, Sparse PFT, and more
- **Rich Environment Collection**: Classic and modern POMDP environments (Tiger, Light-Dark, CartPole, Push, Safety-Ant-Velocity, etc.)
- **Flexible Belief Representations**: Particle filters (weighted and unweighted), vectorized particle beliefs with batched NumPy updates, Gaussian beliefs, and custom belief state implementations
- **Simulation Framework**: Complete experiment management with hyperparameter tuning and distributed computing support
- **Visualization Tools**: Built-in plotting and visualization capabilities for analysis and debugging
- **Production Ready**: Designed for both research experiments and industrial applications

🚀 Quick Start
--------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yaacovpariente/POMDPPlanners.git
   cd POMDPPlanners

   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -e .  # Install in development mode

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

📚 Documentation Sections
--------------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart


.. toctree::
   :maxdepth: 3
   :caption: Core Components

   core/beliefs
   core/environments
   core/planners
   core/simulations

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage
   examples/planners_comparison
   examples/hyperparameter_tuning

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Release Notes

   changelog

🏗️ Architecture Overview
-------------------------

**Core Components**

- **POMDPPlanners.core**: Fundamental abstractions (Environment, Policy, Belief, Distributions)
- **POMDPPlanners.environments**: POMDP environment implementations
- **POMDPPlanners.planners**: Planning algorithm implementations
- **POMDPPlanners.simulations**: Experiment management and execution framework
- **POMDPPlanners.utils**: Helper functions and visualization tools

**Supported Algorithms**

*MCTS-Based Planners*

- **POMCP**: Partially Observable Monte Carlo Planning
- **PFT-DPW**: Progressive Widening with Particle Filter Trees
- **Sparse PFT**: Sparse sampling with Particle Filter Trees

*Other Planners*

- **Sparse Sampling**: Classical sparse sampling algorithm
- **Open Loop Planners**: Non-feedback planning approaches

**Available Environments**

- **Tiger POMDP**: Classic two-door problem
- **Light-Dark POMDP**: Navigation with position-dependent observation noise (continuous and discrete variants)
- **CartPole POMDP**: Partially observable cart-pole balancing
- **Mountain Car POMDP**: Partially observable mountain car
- **Push POMDP**: Object manipulation environment (discrete and continuous variants)
- **RockSample POMDP**: Classic rock-sampling benchmark with long-horizon information gathering
- **PacMan POMDP**: Partially observable PacMan domain
- **Continuous LaserTag POMDP**: Laser-based pursuit-evasion with continuous state
- **Safety Ant Velocity**: Safety-constrained locomotion task

🤝 Contributing
---------------

We welcome contributions of all kinds! Please see our :doc:`contributing` guide for details on:

- Setting up the development environment
- Coding guidelines and standards
- Submitting pull requests
- Reporting bugs and requesting features

📄 License
----------

This project is licensed under the MIT License - see the `LICENSE.md <https://github.com/yaacovpariente/POMDPPlanners/blob/master/LICENSE.md>`_ file for details.

🎓 Citation
-----------

If you use POMDPPlanners in your research, please cite:

.. code-block:: bibtex

   @misc{pariente2026pomdpplannersopensourcepackagepomdp,
         title={POMDPPlanners: Open-Source Package for POMDP Planning},
         author={Yaacov Pariente and Vadim Indelman},
         year={2026},
         eprint={2602.20810},
         archivePrefix={arXiv},
         primaryClass={cs.AI},
         url={https://arxiv.org/abs/2602.20810},
   }

🔗 Links
--------

- **Repository**: https://github.com/yaacovpariente/POMDPPlanners
- **Issues**: https://github.com/yaacovpariente/POMDPPlanners/issues
- **Discussions**: https://github.com/yaacovpariente/POMDPPlanners/discussions

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`