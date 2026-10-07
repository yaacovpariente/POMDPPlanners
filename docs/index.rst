POMDPPlanners
=============

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

A Python package for POMDP planning: a library of planning algorithms, a suite
of environments to run them on, and a simulation framework that runs batches
of episodes, caches each episode under its configuration, and reports
confidence intervals.

Start here
----------

- :doc:`quickstart` — **run your first example**, from install to a planned
  action.
- :doc:`environments/base` — **choose an environment**: the catalog, with each
  one's state, actions, observations and dependencies.
- :doc:`planners/base` — **compare planners**: what each one supports.

Install
-------

.. code-block:: bash

   git clone https://github.com/yaacovpariente/POMDPPlanners.git
   cd POMDPPlanners
   python -m venv .venv && source .venv/bin/activate
   pip install -e .

Full instructions, including the optional dependencies for the realistic
simulators, are in :doc:`installation`.

Plan one action
---------------

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=500)

   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="tiger_planner",
       time_out_in_seconds=2.0,
   )

   # ``action`` returns a list; it has length 1 for closed-loop planning.
   actions, run_data = planner.action(belief)
   print(f"Recommended action: {actions[0]}")

What is in the package
----------------------

- ``POMDPPlanners.core`` — the abstractions: ``Environment``, ``Policy``,
  ``Belief``, distributions and search trees.
- ``POMDPPlanners.environments`` — 20 benchmark environments plus four wrappers
  around external simulators.
- ``POMDPPlanners.planners`` — MCTS planners (POMCP, PFT-DPW, Sparse PFT),
  sparse sampling, and open-loop planners.
- ``POMDPPlanners.simulations`` — running batches of episodes, caching them,
  and hyperparameter tuning.
- ``POMDPPlanners.utils`` — statistics, visualization and analysis helpers.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quickstart
   examples/index
   core/beliefs
   core/simulations
   examples/planners_comparison
   examples/hyperparameter_tuning
   environments/custom
   guide/custom_planners
   guide/saving_loading

.. toctree::
   :maxdepth: 1
   :caption: Planners
   :hidden:

   planners/base
   planners/pomcp
   planners/pomcp_dpw
   planners/pomcpow
   planners/pft_dpw
   planners/sparse_pft
   planners/sparse_sampling
   planners/vopp
   planners/beta_zero
   planners/constrained_zero
   planners/constrained_and_cvar
   planners/open_loop

.. toctree::
   :maxdepth: 1
   :caption: Environments
   :hidden:

   environments/base
   environments/tiger
   environments/rock_sample
   environments/battleship
   environments/capture_the_flag
   environments/occupancy_grid_mapping
   environments/chicheck_invaders
   environments/firefighting
   environments/snake
   environments/pacman
   environments/maze
   environments/light_dark
   environments/cartpole
   environments/push
   environments/laser_tag
   environments/safety_ant_velocity
   environments/sanity
   environments/mountain_car
   environments/realistic

.. toctree::
   :maxdepth: 1
   :caption: Common
   :hidden:

   common/beliefs
   common/distributions_and_spaces
   common/simulation_api
   common/metrics
   common/statistics
   common/visualization

.. toctree::
   :maxdepth: 1
   :caption: Misc
   :hidden:

   misc/changelog
   misc/citation
   misc/papers

Citation
--------

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

Links
-----

- `Repository <https://github.com/yaacovpariente/POMDPPlanners>`_
- `Issues <https://github.com/yaacovpariente/POMDPPlanners/issues>`_
- `Discussions <https://github.com/yaacovpariente/POMDPPlanners/discussions>`_
- Licensed under the MIT License.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
