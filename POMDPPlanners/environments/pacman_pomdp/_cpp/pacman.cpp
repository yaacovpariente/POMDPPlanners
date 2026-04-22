// PacMan POMDP native extension.
//
// Stage 1 (this commit): scaffold only — registers the module and seeds the
// per-module RNG singleton from pomdp_native/rng.hpp. Subsequent commits add
// PacManTransitionCpp and PacManObservationCpp.

#include <cstdint>

#include <pybind11/pybind11.h>

#include "pomdp_native/rng.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
    m.doc() = "PacMan POMDP native C++ kernels (pomdp_native).";

    m.def(
        "set_seed",
        [](std::uint64_t seed) { pomdp_native::set_default_seed(seed); },
        py::arg("seed"),
        "Seed the module-local RNG used by sample()/batch entry points.");
}
