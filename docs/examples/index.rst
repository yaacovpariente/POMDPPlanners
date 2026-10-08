Examples
========

Worked examples, from a first planned action to a tuning study. Three are pages
in these docs. The rest are Jupyter notebooks in ``docs/examples/`` of the
repository. The docs build does not run notebooks, so they are linked on
GitHub, which shows any outputs saved in them.

To run a notebook yourself, install the package from a checkout (see
:doc:`../installation`) and open it with Jupyter:

.. code-block:: bash

   pip install jupyter
   jupyter notebook docs/examples/basic_usage.ipynb

Pages
-----

:doc:`basic_usage`
   Plan with POMCP on Tiger, run a full episode and then a batch with
   statistics, try other environments, compare planners, and draw an episode.

:doc:`planners_comparison`
   Compare POMCPOW and PFT-DPW on the Push and Light-Dark problems through the
   simulation API, with confidence intervals for each pair.

:doc:`hyperparameter_tuning`
   Search a planner's parameters with Optuna: on one environment and on
   several, with the predefined search ranges, with risk-averse objectives and
   categorical parameters, and how to read the results.

Notebooks
---------

`basic_usage.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/basic_usage.ipynb>`_
   The plan, act, observe and update loop on Tiger, ``run_episode``,
   continuous actions on Light-Dark, a visualization, and a minimal custom
   planner.

`belief_representations.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/belief_representations.ipynb>`_
   The three belief types side by side: weighted particles, a Gaussian with
   linear, extended and unscented Kalman updates, and a Gaussian mixture. Shows
   what resampling does to the effective sample size.

`custom_environment.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/custom_environment.ipynb>`_
   Builds a small Weather POMDP method by method, runs POMCP and
   PFT-DPW on it, and adds environment metrics. A companion to
   :doc:`../environments/custom`.

`tree_analysis_debugging.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/tree_analysis_debugging.ipynb>`_
   Reads the search metrics every MCTS planner returns, prints and plots the
   search tree, and shows how ``n_simulations`` and ``exploration_constant``
   change it. Ends with a table of common problems and their fixes.

`advanced_optimization.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/advanced_optimization.ipynb>`_
   More tuning: finding the planners and metrics that fit an
   environment, optimizing several metrics at once, risk-averse objectives,
   and reading the results table.

`planners_comparison.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/planners_comparison.ipynb>`_
   The notebook version of :doc:`planners_comparison`, with saved outputs.

`isaac_state_and_observation.ipynb <https://github.com/yaacovpariente/POMDPPlanners/blob/develop/docs/examples/isaac_state_and_observation.ipynb>`_
   How state and observation are defined for the Isaac Lab tasks, and why the
   simulator (the world) and the planner's model are two separate
   environments. Runs without Isaac Sim. A companion to
   :doc:`../environments/realistic`.

.. toctree::
   :hidden:

   basic_usage
