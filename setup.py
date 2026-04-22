"""Setuptools entry point for native C++ extensions.

Project metadata lives in pyproject.toml; this file exists only to register
the pybind11 C++ extension modules that ship with the package. Extensions
are built automatically by ``pip install`` / ``pip install -e .`` as long
as a C++17 compiler is available on the system.

The shared ``POMDPPlanners.core._native`` extension provides the
``pomdp_native`` header-only library (Cholesky-factored Gaussian, RNG state,
Python <-> C++ marshalling, transition/observation base classes) that
per-environment extensions include via ``#include <pomdp_native/...>``.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

POMDP_NATIVE_INCLUDE = "POMDPPlanners/core/_cpp/include"


def _make_ext(name: str, sources: list) -> Pybind11Extension:
    return Pybind11Extension(
        name=name,
        sources=sources,
        include_dirs=[POMDP_NATIVE_INCLUDE],
        cxx_std=17,
        extra_compile_args=["-O3", "-fno-math-errno"],
    )


ext_modules = [
    _make_ext(
        name="POMDPPlanners.core._native",
        sources=["POMDPPlanners/core/_cpp/module.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.mountain_car_pomdp._native",
        sources=["POMDPPlanners/environments/mountain_car_pomdp/_cpp/mountain_car.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.cartpole_pomdp._native",
        sources=["POMDPPlanners/environments/cartpole_pomdp/_cpp/cartpole.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.light_dark_pomdp._native",
        sources=["POMDPPlanners/environments/light_dark_pomdp/_cpp/continuous_light_dark.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.laser_tag_pomdp._native",
        sources=["POMDPPlanners/environments/laser_tag_pomdp/_cpp/continuous_laser_tag.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.push_pomdp._native",
        sources=["POMDPPlanners/environments/push_pomdp/_cpp/continuous_push.cpp"],
    ),
    _make_ext(
        name="POMDPPlanners.environments.rock_sample_pomdp._native",
        sources=["POMDPPlanners/environments/rock_sample_pomdp/_cpp/rock_sample.cpp"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
