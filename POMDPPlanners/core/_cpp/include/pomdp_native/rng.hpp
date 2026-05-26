// SPDX-License-Identifier: MIT

// POMDPPlanners shared native RNG state.
//
// Header-only. Each pybind11 extension that #includes this header gets its
// own process-local RNGState (one per .so via ODR on the inline static),
// which matches the pre-existing per-module semantics: seeding
// POMDPPlanners.core._native does not silently affect an env's sampler,
// and each env keeps an independent RNG the caller can reset explicitly.

#ifndef POMDP_NATIVE_RNG_HPP_
#define POMDP_NATIVE_RNG_HPP_

#include <cstdint>
#include <random>

namespace pomdp_native {

class RNGState {
  public:
    RNGState() : engine_(std::random_device{}()) {}
    explicit RNGState(std::uint64_t seed) : engine_(seed) {}

    void seed(std::uint64_t seed) { engine_.seed(seed); }
    std::mt19937_64 &engine() noexcept { return engine_; }

  private:
    std::mt19937_64 engine_;
};

inline RNGState &default_rng() {
    static RNGState rng;
    return rng;
}

inline void set_default_seed(std::uint64_t seed) { default_rng().seed(seed); }

}  // namespace pomdp_native

#endif  // POMDP_NATIVE_RNG_HPP_
