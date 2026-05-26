// Copyright 2025 Yaacov Pariente
// SPDX-License-Identifier: MIT

// Python-facing surface of the shared native core.
//
// Compiles to the POMDPPlanners.core._native extension module. The C++
// primitives in pomdp_native/*.hpp (RNGState, GaussianND, TransitionModelCpp,
// ObservationModelCpp, marshalling helpers) are intentionally NOT bound to
// Python -- they exist only for inclusion by per-environment .cpp files. The
// only Python-visible symbol is set_default_seed, which seeds this module's
// process-local default RNG.
//
// Note: each compiled extension that #includes <pomdp_native/rng.hpp> gets
// its own default_rng instance (ODR on the inline function-static). Seeding
// POMDPPlanners.core._native therefore does NOT affect an environment
// module's RNG -- each env extension owns its sampler state and must be
// seeded through its own module-level entry point.

#include <pybind11/pybind11.h>

#include "pomdp_native/rng.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
    m.doc() = "POMDPPlanners shared native core (pomdp_native headers + seeding).";
    m.def("set_default_seed", &pomdp_native::set_default_seed, py::arg("seed"),
          "Seed the default RNG used by this module. Each native extension has "
          "its own default RNG; seeding here does not affect environment extensions.");
}
