"""Type stubs for the shared native core module.

Declares the Python-visible API of ``POMDPPlanners.core._native`` so pyright
can type-check modules that import from it. The runtime implementation lives
in ``_cpp/module.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

def set_default_seed(seed: int) -> None:
    """Seed the default RNG owned by this module.

    Each compiled native extension has its own default RNG (one per
    ``.so`` due to ODR on the inline function-static); seeding this module
    does not affect environment extensions. Environment modules expose
    their own seeding entry points.
    """
    ...
