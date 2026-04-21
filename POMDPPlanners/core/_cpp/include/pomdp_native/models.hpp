// Header-only C++ base classes for native transition / observation models,
// templated on the compile-time state dimension.
//
// These classes are NOT exposed to Python. Per-env pybind11 extensions
// instantiate the base at their dimension (e.g. MountainCar uses
// ``TransitionModelCpp<2>``), override the env-specific ``compute_mean``
// (and optional ``post_sample_transform``) hook, and bind the concrete
// subclass. The sample() / probability() loops, RNG selection, and
// stack-allocated scratch all live here.

#ifndef POMDP_NATIVE_MODELS_HPP_
#define POMDP_NATIVE_MODELS_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "pomdp_native/gaussian.hpp"
#include "pomdp_native/marshalling.hpp"
#include "pomdp_native/rng.hpp"

namespace pomdp_native {

template <std::size_t Dim>
class TransitionModelCpp {
  public:
    TransitionModelCpp(const std::array<double, Dim> &state, pybind11::object action,
                       const GaussianND<Dim> &noise)
        : state_(state), action_(std::move(action)), noise_(noise) {}

    virtual ~TransitionModelCpp() = default;

    pybind11::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        double mean[Dim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(mean);

        double buf[Dim];  // NOLINT(modernize-avoid-c-arrays)
        pybind11::list out;
        RNGState &rng = default_rng();
        for (int i = 0; i < n_samples; ++i) {
            noise_.sample_into(buf, mean, rng);
            post_sample_transform(buf);
            out.append(array_from_vector(buf, Dim));
        }
        return out;
    }

    pybind11::array_t<double> probability(const pybind11::object &values) const {
        double mean[Dim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(mean);

        auto batch = extract_rows_nd(values, Dim);
        auto out = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * Dim;
            buf(static_cast<pybind11::ssize_t>(i)) = std::exp(noise_.log_pdf(x, mean));
        }
        return out;
    }

    const std::array<double, Dim> &state_arr() const noexcept { return state_; }
    const pybind11::object &action_obj() const noexcept { return action_; }

  protected:
    // Env-specific deterministic next-state computation. Writes Dim doubles
    // to out. Called once per sample() / probability() call.
    virtual void compute_mean(double *out) const = 0;

    // Optional hook, called after each additive Gaussian draw so envs can
    // clip, wrap, or otherwise project the sample onto their feasible set.
    virtual void post_sample_transform(double * /*sample*/) const {}

    std::array<double, Dim> state_;
    pybind11::object action_;
    GaussianND<Dim> noise_;
};

template <std::size_t Dim>
class ObservationModelCpp {
  public:
    ObservationModelCpp(const std::array<double, Dim> &next_state, pybind11::object action,
                        const GaussianND<Dim> &noise)
        : next_state_(next_state), action_(std::move(action)), noise_(noise) {}

    virtual ~ObservationModelCpp() = default;

    pybind11::list sample(int n_samples) const {
        if (n_samples < 0) {
            throw std::invalid_argument("n_samples must be non-negative");
        }
        double mean[Dim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(mean);

        double buf[Dim];  // NOLINT(modernize-avoid-c-arrays)
        pybind11::list out;
        RNGState &rng = default_rng();
        for (int i = 0; i < n_samples; ++i) {
            noise_.sample_into(buf, mean, rng);
            out.append(array_from_vector(buf, Dim));
        }
        return out;
    }

    pybind11::array_t<double> probability(const pybind11::object &values) const {
        double mean[Dim];  // NOLINT(modernize-avoid-c-arrays)
        compute_mean(mean);

        auto batch = extract_rows_nd(values, Dim);
        auto out = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * Dim;
            buf(static_cast<pybind11::ssize_t>(i)) = std::exp(noise_.log_pdf(x, mean));
        }
        return out;
    }

    const std::array<double, Dim> &next_state_arr() const noexcept { return next_state_; }
    const pybind11::object &action_obj() const noexcept { return action_; }

  protected:
    // Default behavior: observation mean equals the input next_state. Envs
    // with state-dependent observation means (e.g. light-dark) can override.
    virtual void compute_mean(double *out) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            out[i] = next_state_[i];
        }
    }

    std::array<double, Dim> next_state_;
    pybind11::object action_;
    GaussianND<Dim> noise_;
};

}  // namespace pomdp_native

#endif  // POMDP_NATIVE_MODELS_HPP_
