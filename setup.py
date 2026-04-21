"""Setuptools entry point for native C++ extensions.

Project metadata lives in pyproject.toml; this file exists only to register
the pybind11 C++ extension modules that ship with the package. The
extension (``MountainCarPOMDP`` sampling hot path) is built automatically
by ``pip install`` / ``pip install -e .`` as long as a C++17 compiler is
available on the system.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        name="POMDPPlanners.environments.mountain_car_pomdp._native",
        sources=["POMDPPlanners/environments/mountain_car_pomdp/_cpp/mountain_car.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-fno-math-errno"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
