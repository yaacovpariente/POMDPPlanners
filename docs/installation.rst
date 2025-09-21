Installation Guide
==================

This guide covers different ways to install POMDPPlanners and its dependencies.

Requirements
------------

**System Requirements**

- Python 3.8 or higher
- Operating System: Linux, macOS, or Windows
- Minimum 4GB RAM recommended
- 1GB free disk space

**Python Dependencies**

Core dependencies are automatically installed with the package:

- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- PyYAML >= 5.4.0
- Gymnasium >= 0.26.0 (for Gym environments)

Installation Methods
-------------------

Development Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or if you want the latest features:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yaacovpariente/POMDPPlanners.git
   cd POMDPPlanners

   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   source .venv/bin/activate          # Linux/macOS
   # .venv\Scripts\activate           # Windows

   # Install dependencies
   pip install -r requirements.txt

   # Install in development mode
   pip install -e .

PyPI Installation (Future)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once published to PyPI, you can install with:

.. code-block:: bash

   pip install pomdpplanners

Virtual Environment Setup
-------------------------

**Why Use Virtual Environments?**

Virtual environments isolate your project dependencies and prevent conflicts with other Python projects.

**Creating a Virtual Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv .venv

   # Activate on Linux/macOS
   source .venv/bin/activate

   # Activate on Windows
   .venv\Scripts\activate

   # Verify activation (should show virtual environment path)
   which python

**Deactivating Virtual Environment**

.. code-block:: bash

   deactivate

Development Dependencies
-----------------------

For contributors and developers, install additional development tools:

.. code-block:: bash

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install

Development dependencies include:

- **Testing**: pytest, pytest-cov
- **Code Quality**: black, pylint, flake8
- **Documentation**: sphinx, sphinx-rtd-theme
- **Pre-commit**: pre-commit hooks for code formatting

Optional Dependencies
--------------------

**Distributed Computing**

For running experiments on clusters:

.. code-block:: bash

   pip install ray[default]  # For Ray distributed computing
   pip install dask[complete]  # For Dask distributed computing

**Advanced Visualization**

For enhanced plotting capabilities:

.. code-block:: bash

   pip install seaborn  # Statistical plotting
   pip install plotly   # Interactive plots

**Deep Learning**

For neural network-based components:

.. code-block:: bash

   pip install torch torchvision  # PyTorch
   # or
   pip install tensorflow         # TensorFlow

Verification
-----------

Verify your installation by running the test suite:

.. code-block:: bash

   # Ensure virtual environment is activated
   source .venv/bin/activate

   # Run tests
   pytest

   # Run a quick example
   python -c "
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   env = TigerPOMDP()
   print('Installation successful!')
   print(f'Tiger POMDP has {len(env.get_states())} states')
   "

Troubleshooting
--------------

**Common Issues**

*Import Errors*

.. code-block:: bash

   # Ensure virtual environment is activated
   source .venv/bin/activate

   # Reinstall in development mode
   pip install -e .

*Missing Dependencies*

.. code-block:: bash

   # Update pip
   pip install --upgrade pip

   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall

*Virtual Environment Issues*

.. code-block:: bash

   # Remove and recreate virtual environment
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .

**Platform-Specific Notes**

*Windows*

- Use ``python`` instead of ``python3``
- Use ``pip`` instead of ``pip3``
- Use backslashes (``\``) in paths or forward slashes with raw strings

*macOS*

- You may need to install Xcode command line tools: ``xcode-select --install``
- Consider using Homebrew for Python installation

*Linux*

- Install Python development headers: ``sudo apt-get install python3-dev`` (Ubuntu/Debian)
- For CentOS/RHEL: ``sudo yum install python3-devel``

Getting Help
-----------

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/yaacovpariente/POMDPPlanners/issues>`_
2. Create a new issue with:
   - Your operating system and Python version
   - Complete error message
   - Steps you've tried
3. Join the `Discussions <https://github.com/yaacovpariente/POMDPPlanners/discussions>`_

Next Steps
----------

Once installed, proceed to the :doc:`quickstart` guide to begin using POMDPPlanners!