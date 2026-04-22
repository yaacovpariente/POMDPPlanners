"""Type stubs for the PacMan POMDP native C++ extension.

Stage 1 surface: the module-local RNG seeding function only. Classes will
be declared here as they are added in subsequent commits.
"""

def set_seed(seed: int) -> None:
    """Seed the module-local RNG used by ``sample()`` / batch entry points."""
