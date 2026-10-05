Installation
============

POMDPPlanners needs Python 3.10 or newer and a C++17 compiler. The compiler is
needed because several environments have C++ parts that are built when the
package is installed, from PyPI as well as from a clone.

From PyPI
---------

.. code-block:: bash

   pip install POMDPPlanners

The release on PyPI is a source distribution, so pip compiles the C++ modules
during the install. Install a compiler first: ``xcode-select --install`` on
macOS, ``g++`` (for example ``sudo apt-get install g++``) on Debian or Ubuntu.

From a clone
------------

Install from a clone to get the latest code on ``develop`` or to change the
package:

.. code-block:: bash

   git clone https://github.com/yaacovpariente/POMDPPlanners.git
   cd POMDPPlanners
   python -m venv .venv && source .venv/bin/activate
   pip install -e .

``-e`` makes the install editable: changes to the Python files take effect
without reinstalling. Changes to the C++ files do not. After editing one,
rebuild the extensions in place:

.. code-block:: bash

   python setup.py build_ext --inplace

Optional extras
---------------

The package defines two extras:

``dev``
   The test, lint and type-check tools: pytest, black, pylint, pyright and
   pre-commit. It also installs ``highway-env``, the simulator behind the
   Racetrack environment.

``docs``
   Sphinx and its plugins, to build this documentation.

.. code-block:: bash

   pip install -e ".[dev,docs]"

PyTorch is not an extra. It is a core dependency, used by the learned and
vectorized planners, so a plain install already pulls it in.

The simulators behind the realistic environments (CARLA, Isaac Lab, nuPlan)
are not installed by any extra. Each needs its own install, described in
:doc:`environments/realistic`.

Check the install
-----------------

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   env = TigerPOMDP(discount_factor=0.95)
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=5,
       exploration_constant=50.0,
       name="check",
       time_out_in_seconds=2.0,
   )
   actions, _ = planner.action(get_initial_belief(env, n_particles=100))
   print(actions)

If this prints a list with one Tiger action, the package and its dependencies
are working. :doc:`quickstart` goes on from here.
