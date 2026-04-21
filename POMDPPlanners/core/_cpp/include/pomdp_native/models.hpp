// Header-only C++ base classes for native transition / observation models,
// templated on the compile-time state dimension.
//
// These classes are NOT exposed to Python. Per-env pybind11 extensions
// instantiate the base at their dimension (e.g. MountainCar uses
// ``TransitionModelCpp<2>``), override the env-specific
// ``compute_mean_from_state`` (transition) /
// ``compute_mean_from_next_state`` (observation) hook (plus optional
// ``post_sample_transform``), and bind the concrete subclass. The
// sample() / probability() loops, RNG selection, and stack-allocated
// scratch all live here.
//
// The ``compute_mean_from_*`` hooks take the state / next_state as an
// explicit argument so batched entry points (see Layer 2) can vary it
// across rows without mutating the model's ``state_`` / ``next_state_``
// members. Single-instance ``sample()`` / ``probability()`` simply pass
// ``state_.data()`` / ``next_state_.data()`` through.

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
        compute_mean_from_state(state_.data(), mean);

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
        compute_mean_from_state(state_.data(), mean);

        auto batch = extract_rows_nd(values, Dim);
        auto out = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * Dim;
            buf(static_cast<pybind11::ssize_t>(i)) = std::exp(noise_.log_pdf(x, mean));
        }
        return out;
    }

    // Batch sample: given N particles (N, Dim), return N next particles
    // (N, Dim) using the model's stored action, noise, and compute_mean_from_state
    // + post_sample_transform hooks. Uses the module default RNG.
    // The model's own state_ is NOT read on this path -- state is varied per row
    // from the input array.
    pybind11::array_t<double> batch_sample(
        pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> particles)
        const {
        if (particles.ndim() != 2 ||
            static_cast<std::size_t>(particles.shape(1)) != Dim) {
            throw std::invalid_argument("particles must have shape (N, Dim)");
        }
        const auto n_rows = static_cast<std::size_t>(particles.shape(0));
        auto particles_view = particles.template unchecked<2>();

        auto out = pybind11::array_t<double>(
            {static_cast<pybind11::ssize_t>(n_rows), static_cast<pybind11::ssize_t>(Dim)});
        auto out_view = out.template mutable_unchecked<2>();

        double state_row[Dim];  // NOLINT(modernize-avoid-c-arrays)
        double mean[Dim];       // NOLINT(modernize-avoid-c-arrays)
        double buf[Dim];        // NOLINT(modernize-avoid-c-arrays)
        RNGState &rng = default_rng();
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t d = 0; d < Dim; ++d) {
                state_row[d] = particles_view(static_cast<pybind11::ssize_t>(i),
                                              static_cast<pybind11::ssize_t>(d));
            }
            compute_mean_from_state(state_row, mean);
            noise_.sample_into(buf, mean, rng);
            post_sample_transform(buf);
            for (std::size_t d = 0; d < Dim; ++d) {
                out_view(static_cast<pybind11::ssize_t>(i),
                         static_cast<pybind11::ssize_t>(d)) = buf[d];
            }
        }
        return out;
    }

    const std::array<double, Dim> &state_arr() const noexcept { return state_; }
    const pybind11::object &action_obj() const noexcept { return action_; }

  protected:
    // Env-specific deterministic next-state computation. Given a state vector
    // (length Dim) and the model's stored action, writes the deterministic
    // next-state mean into out (also length Dim). Called once per sample() /
    // probability() invocation; called N times per batch_sample() call with
    // varying state. env-specific params (power, gravity, ...) stay as
    // members on the subclass.
    virtual void compute_mean_from_state(const double *state, double *out) const = 0;

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
        compute_mean_from_next_state(next_state_.data(), mean);

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
        compute_mean_from_next_state(next_state_.data(), mean);

        auto batch = extract_rows_nd(values, Dim);
        auto out = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(batch.n));
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < batch.n; ++i) {
            const double *x = batch.flat.data() + i * Dim;
            buf(static_cast<pybind11::ssize_t>(i)) = std::exp(noise_.log_pdf(x, mean));
        }
        return out;
    }

    // Batch log-likelihood: given N next-state particles (N, Dim) and a single
    // observation (Dim,), return the per-particle log P(observation | next_state,
    // action). Uses the model's stored action, noise, and
    // compute_mean_from_next_state hook. The model's own next_state_ is NOT read
    // on this path.
    pybind11::array_t<double> batch_log_likelihood(
        pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
            next_particles,
        pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
            observation) const {
        if (next_particles.ndim() != 2 ||
            static_cast<std::size_t>(next_particles.shape(1)) != Dim) {
            throw std::invalid_argument("next_particles must have shape (N, Dim)");
        }
        if (observation.ndim() != 1 ||
            static_cast<std::size_t>(observation.shape(0)) != Dim) {
            throw std::invalid_argument("observation must have shape (Dim,)");
        }
        const auto n_rows = static_cast<std::size_t>(next_particles.shape(0));
        auto particles_view = next_particles.template unchecked<2>();
        auto obs_view = observation.template unchecked<1>();

        double obs_buf[Dim];  // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t d = 0; d < Dim; ++d) {
            obs_buf[d] = obs_view(static_cast<pybind11::ssize_t>(d));
        }

        auto out = pybind11::array_t<double>(static_cast<pybind11::ssize_t>(n_rows));
        auto out_view = out.mutable_unchecked<1>();

        double next_state_row[Dim];  // NOLINT(modernize-avoid-c-arrays)
        double mean[Dim];            // NOLINT(modernize-avoid-c-arrays)
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t d = 0; d < Dim; ++d) {
                next_state_row[d] = particles_view(static_cast<pybind11::ssize_t>(i),
                                                   static_cast<pybind11::ssize_t>(d));
            }
            compute_mean_from_next_state(next_state_row, mean);
            out_view(static_cast<pybind11::ssize_t>(i)) = noise_.log_pdf(obs_buf, mean);
        }
        return out;
    }

    const std::array<double, Dim> &next_state_arr() const noexcept { return next_state_; }
    const pybind11::object &action_obj() const noexcept { return action_; }

  protected:
    // Default behavior: observation mean equals the input next_state. Envs
    // with state-dependent observation means (e.g. light-dark) can override.
    // Called once per sample() / probability() invocation; called N times
    // per batch_log_likelihood() call with varying next_state.
    virtual void compute_mean_from_next_state(const double *next_state, double *out) const {
        for (std::size_t i = 0; i < Dim; ++i) {
            out[i] = next_state[i];
        }
    }

    std::array<double, Dim> next_state_;
    pybind11::object action_;
    GaussianND<Dim> noise_;
};

}  // namespace pomdp_native

#endif  // POMDP_NATIVE_MODELS_HPP_
