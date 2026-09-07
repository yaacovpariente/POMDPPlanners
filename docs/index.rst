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
of environments to run them on, and a simulation framework that makes the
comparison between them reproducible.

Start here
----------

- :doc:`quickstart` — **run your first example**, from install to a planned
  action.
- :doc:`environments/index` — **choose an environment**: the catalog, with each
  one's state, actions, observations and dependencies.
- :doc:`core/planners` — **choose a planner**: POMCP, PFT-DPW, Sparse PFT,
  sparse sampling and the open-loop planners.

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
       n_simulations=1000,
   )

   # ``action`` returns a list; it has length 1 for closed-loop planning.
   actions, run_data = planner.action(belief)
   print(f"Recommended action: {actions[0]}")

What is in the package
----------------------

- ``POMDPPlanners.core`` — the abstractions: ``Environment``, ``Policy``,
  ``Belief``, distributions and search trees.
- ``POMDPPlanners.environments`` — 16 benchmark environments plus four wrappers
  around external simulators.
- ``POMDPPlanners.planners`` — MCTS planners (POMCP, PFT-DPW, Sparse PFT),
  sparse sampling, and open-loop planners.
- ``POMDPPlanners.simulations`` — running batches of episodes, caching them,
  and hyperparameter tuning.
- ``POMDPPlanners.utils`` — statistics, visualization and analysis helpers.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installation
   quickstart
   examples/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Environments
   :hidden:

   environments/index

.. toctree::
   :maxdepth: 2
   :caption: Planners
   :hidden:

   core/planners
   core/beliefs

.. toctree::
   :maxdepth: 2
   :caption: Running experiments
   :hidden:

   core/simulations
   examples/planners_comparison
   examples/hyperparameter_tuning

.. toctree::
   :maxdepth: 2
   :caption: Extending the package
   :hidden:

   environments/custom

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   api/modules
   misc/changelog

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
