// SPDX-License-Identifier: MIT

// N-dimensional Gaussian with Cholesky-factored covariance, templated on
// the compile-time dimension.
//
// Generalizes the hand-coded 2D Cholesky that previously lived in
// POMDPPlanners/environments/mountain_car_pomdp/_cpp/mountain_car.cpp while
// keeping the compiler's ability to fully unroll the inner loops (the
// reason the pre-port version was fast). Each env that uses the shared
// core picks its dimension at compile time: MountainCar instantiates
// ``GaussianND<2>``; a 4-D env would use ``GaussianND<4>``.
//
// Storage layout is packed lower-triangular (row-major): L(i, j) for j <= i
// lives at lower_tri_[i*(i+1)/2 + j]. Covariance is factored once at
// construction; sample / log_pdf are O(Dim^2).

#ifndef POMDP_NATIVE_GAUSSIAN_HPP_
#define POMDP_NATIVE_GAUSSIAN_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "pomdp_native/rng.hpp"

namespace pomdp_native {

namespace detail {
constexpr double kLog2Pi = 1.8378770664093454835606594728112352797227949;  // log(2*pi)

constexpr std::size_t packed_index(std::size_t row, std::size_t col) noexcept {
    return row * (row + 1) / 2 + col;
}
}  // namespace detail

template <std::size_t Dim>
class GaussianND {
    static_assert(Dim > 0, "GaussianND dim must be > 0");

  public:
    static constexpr std::size_t kPackedSize = Dim * (Dim + 1) / 2;

    static GaussianND from_covariance(const pybind11::array_t<double> &cov) {
        auto unchecked = cov.unchecked<2>();
        if (static_cast<std::size_t>(unchecked.shape(0)) != Dim ||
            static_cast<std::size_t>(unchecked.shape(1)) != Dim) {
            throw std::invalid_argument("covariance must be " + std::to_string(Dim) + "x" +
                                        std::to_string(Dim));
        }
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = i + 1; j < Dim; ++j) {
                if (std::abs(unchecked(static_cast<pybind11::ssize_t>(i),
                                       static_cast<pybind11::ssize_t>(j)) -
                             unchecked(static_cast<pybind11::ssize_t>(j),
                                       static_cast<pybind11::ssize_t>(i))) > 1e-12) {
                    throw std::invalid_argument("covariance must be symmetric");
                }
            }
        }

        std::array<double, kPackedSize> lower{};
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double sum = unchecked(static_cast<pybind11::ssize_t>(i),
                                       static_cast<pybind11::ssize_t>(j));
                for (std::size_t k = 0; k < j; ++k) {
                    sum -= lower[detail::packed_index(i, k)] *
                           lower[detail::packed_index(j, k)];
                }
                if (i == j) {
                    if (sum <= 0.0) {
                        throw std::invalid_argument("covariance is not positive definite");
                    }
                    lower[detail::packed_index(i, j)] = std::sqrt(sum);
                } else {
                    lower[detail::packed_index(i, j)] =
                        sum / lower[detail::packed_index(j, j)];
                }
            }
        }

        double log_det = 0.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            log_det += std::log(lower[detail::packed_index(i, i)]);
        }
        log_det *= 2.0;
        const double log_norm =
            -0.5 * (static_cast<double>(Dim) * detail::kLog2Pi + log_det);

        return GaussianND{lower, log_norm};
    }

    static constexpr std::size_t dim() noexcept { return Dim; }

    // Draw x = mean + L * z where z ~ N(0, I_Dim). Writes Dim doubles to out.
    void sample_into(double *out, const double *mean, RNGState &rng) const {
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        // Store z into out first, then walk i from high to low so we read
        // z[j] (for j <= i) before overwriting out[i] with x[i].
        for (std::size_t i = 0; i < Dim; ++i) {
            out[i] = standard_normal(rng.engine());
        }
        for (std::size_t idx = Dim; idx > 0; --idx) {
            const std::size_t i = idx - 1;
            double xi = mean[i];
            for (std::size_t j = 0; j <= i; ++j) {
                xi += lower_tri_[detail::packed_index(i, j)] * out[j];
            }
            out[i] = xi;
        }
    }

    // log pdf at x via forward substitution: solve L y = (x - mean);
    // return log_normalization - 0.5 * y^T y.
    double log_pdf(const double *x, const double *mean) const {
        double y[Dim];  // NOLINT(modernize-avoid-c-arrays) -- Dim known at compile time
        double mahalanobis_sq = 0.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            double sum = x[i] - mean[i];
            for (std::size_t j = 0; j < i; ++j) {
                sum -= lower_tri_[detail::packed_index(i, j)] * y[j];
            }
            y[i] = sum / lower_tri_[detail::packed_index(i, i)];
            mahalanobis_sq += y[i] * y[i];
        }
        return log_normalization_ - 0.5 * mahalanobis_sq;
    }

  private:
    GaussianND(const std::array<double, kPackedSize> &lower_tri, double log_normalization)
        : lower_tri_(lower_tri), log_normalization_(log_normalization) {}

    std::array<double, kPackedSize> lower_tri_;
    double log_normalization_;
};

}  // namespace pomdp_native

#endif  // POMDP_NATIVE_GAUSSIAN_HPP_
