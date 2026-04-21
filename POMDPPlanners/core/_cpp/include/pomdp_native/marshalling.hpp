// Python <-> C++ numeric marshalling helpers shared across native extensions.
//
// Generalizes the duck-typed (tuple / list / ndarray) conversion helpers
// that previously lived inline in mountain_car.cpp. Each function enforces
// a caller-declared expected dimension so per-env code stays free of shape
// checks.

#ifndef POMDP_NATIVE_MARSHALLING_HPP_
#define POMDP_NATIVE_MARSHALLING_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace pomdp_native {

template <std::size_t Dim>
inline std::array<double, Dim> to_array(const pybind11::object &obj, const char *label) {
    namespace py = pybind11;
    std::array<double, Dim> out{};
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != Dim) {
            throw std::invalid_argument(std::string(label) + " ndarray must be 1-D with length " +
                                        std::to_string(Dim));
        }
        auto unchecked = arr.unchecked<1>();
        for (std::size_t i = 0; i < Dim; ++i) {
            out[i] = unchecked(static_cast<py::ssize_t>(i));
        }
        return out;
    }
    auto seq = obj.cast<py::sequence>();
    if (static_cast<std::size_t>(py::len(seq)) != Dim) {
        throw std::invalid_argument(std::string(label) + " must have length " +
                                    std::to_string(Dim));
    }
    for (std::size_t i = 0; i < Dim; ++i) {
        out[i] = seq[i].cast<double>();
    }
    return out;
}

inline std::vector<double> to_vector(const pybind11::object &obj, std::size_t expected_dim,
                                     const char *label) {
    namespace py = pybind11;
    std::vector<double> out;
    if (py::isinstance<py::array>(obj)) {
        auto arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1 ||
            static_cast<std::size_t>(arr.shape(0)) != expected_dim) {
            throw std::invalid_argument(std::string(label) + " ndarray must be 1-D with length " +
                                        std::to_string(expected_dim));
        }
        auto unchecked = arr.unchecked<1>();
        out.reserve(expected_dim);
        for (std::size_t i = 0; i < expected_dim; ++i) {
            out.push_back(unchecked(static_cast<py::ssize_t>(i)));
        }
        return out;
    }
    auto seq = obj.cast<py::sequence>();
    if (static_cast<std::size_t>(py::len(seq)) != expected_dim) {
        throw std::invalid_argument(std::string(label) + " must have length " +
                                    std::to_string(expected_dim));
    }
    out.reserve(expected_dim);
    for (std::size_t i = 0; i < expected_dim; ++i) {
        out.push_back(seq[i].cast<double>());
    }
    return out;
}

inline pybind11::array_t<double> array_from_vector(const double *data, std::size_t n) {
    namespace py = pybind11;
    auto arr = py::array_t<double>(static_cast<py::ssize_t>(n));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        buf(static_cast<py::ssize_t>(i)) = data[i];
    }
    return arr;
}

struct RowBatch {
    std::vector<double> flat;  // length n * dim, row-major
    std::size_t n;
    std::size_t dim;
};

inline RowBatch extract_rows_nd(const pybind11::object &values, std::size_t expected_dim) {
    namespace py = pybind11;
    RowBatch batch;
    batch.dim = expected_dim;
    if (py::isinstance<py::array>(values)) {
        auto arr = values.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() == 1) {
            if (static_cast<std::size_t>(arr.shape(0)) != expected_dim) {
                throw std::invalid_argument(
                    "1-D values ndarray must have length " + std::to_string(expected_dim));
            }
            auto u = arr.unchecked<1>();
            batch.n = 1;
            batch.flat.reserve(expected_dim);
            for (std::size_t j = 0; j < expected_dim; ++j) {
                batch.flat.push_back(u(static_cast<py::ssize_t>(j)));
            }
            return batch;
        }
        if (arr.ndim() == 2) {
            if (static_cast<std::size_t>(arr.shape(1)) != expected_dim) {
                throw std::invalid_argument("2-D values ndarray must have shape (n, " +
                                            std::to_string(expected_dim) + ")");
            }
            auto u = arr.unchecked<2>();
            batch.n = static_cast<std::size_t>(u.shape(0));
            batch.flat.reserve(batch.n * expected_dim);
            for (py::ssize_t i = 0; i < u.shape(0); ++i) {
                for (std::size_t j = 0; j < expected_dim; ++j) {
                    batch.flat.push_back(u(i, static_cast<py::ssize_t>(j)));
                }
            }
            return batch;
        }
        throw std::invalid_argument("values ndarray must be 1-D or 2-D");
    }
    auto seq = values.cast<py::sequence>();
    const std::size_t rows = static_cast<std::size_t>(py::len(seq));
    batch.n = rows;
    batch.flat.reserve(rows * expected_dim);
    for (std::size_t i = 0; i < rows; ++i) {
        auto row = to_vector(seq[static_cast<py::ssize_t>(i)].cast<py::object>(), expected_dim,
                             "values element");
        batch.flat.insert(batch.flat.end(), row.begin(), row.end());
    }
    return batch;
}

}  // namespace pomdp_native

#endif  // POMDP_NATIVE_MARSHALLING_HPP_
