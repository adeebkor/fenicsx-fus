// Copyright (C) 2024 Adeeb Arif Kor
// SPDX-License-Identifier:    MIT

#pragma once

#include "precompute.hpp"

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cmath>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

// -------------- //
// Mass operators //
// -------------- //

namespace mass {
template <typename T, int Nd, int Nq>
inline void transform(T* __restrict__ detJ, T* __restrict__ phi, T __restrict__ coeff,
                      T* __restrict__ in, T* __restrict__ out) {

  for (int iq = 0; iq < Nq; ++iq) {
    T w0 = 0.0;
    for (int id = 0; id < Nd; ++id)
      w0 += coeff * in[id] * phi[iq * Nd + id];

    const T fw0 = w0 * detJ[iq];

    for (int i = 0; i < Nd; ++i) {
      out[i] += fw0 * phi[iq * Nd + i];
    }
  }
}
} // namespace mass

/// Precompute Mass operator
template <typename T, int P, int Q>
class Mass2D {
public:
  Mass2D(std::shared_ptr<fem::FunctionSpace>& V) : dofmap_(0) {
    // Create map between number of quadrature points to basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[3] = 4;
    Qdegree[4] = 5;
    Qdegree[5] = 6;
    Qdegree[6] = 8;
    Qdegree[7] = 10;
    Qdegree[8] = 12;
    Qdegree[9] = 14;
    Qdegree[10] = 16;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap
    dofmap_ = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto [points, weights] = basix::quadrature::make_quadrature(
        basix::quadrature::type::gll, basix::cell::type::quadrilateral, Qdegree[Q]);

    // Tabulate basis functions at quadrature points
    auto element
        = basix::create_element(basix::element::family::P, basix::cell::type::quadrilateral, P,
                                basix::element::lagrange_variant::gll_warped);

    auto [basis, shape] = element.tabulate(0, points, {weights.size(), 2});
    phi.resize(Nd * Nq);
    std::copy(basis.begin(), basis.end(), phi.begin());

    // Compute the scaled of the Jacobian determinant
    detJ_ = compute_scaled_jacobian_determinant(mesh, points, weights);
  }

  /// Operator y = Mx
  /// @param[in] x Input vector
  /// @param[in] coeffs Coefficients
  /// @param[out] y Output vector
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs, la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    T* phi_ = phi.data();

    for (std::int32_t c = 0; c < Nc; ++c) {
      // Get cell degrees of freedom
      auto cell_dofs = dofmap_.links(c);

      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[cell_dofs[i]];

      std::fill(y_.begin(), y_.end(), 0.0);
      T* detJ = detJ_.data() + c * Nq;
      mass::transform<T, Nd, Nq>(detJ, phi_, coeffs[c], x_.data(), y_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[cell_dofs[i]] += y_[i];
    }
  }

private:
  // Number of dofs per element
  static constexpr int Nd = (P + 1) * (P + 1);

  // Number of quadrature points per element
  static constexpr int Nq = Q * Q;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled Jacobian determinant
  std::vector<T> detJ_;

  // Dofmap
  graph::AdjacencyList<std::int32_t> dofmap_;

  // Basis function
  std::vector<T> phi;

  // Local input array
  std::array<T, Nd> x_;

  // Local output array
  std::array<T, Nd> y_;
};

template <typename T, int P, int Q>
class Mass3D {
public:
  Mass3D(std::shared_ptr<fem::FunctionSpace>& V) : dofmap_(0) {
    // Create map between number of quadrature points to basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[3] = 4;
    Qdegree[4] = 5;
    Qdegree[5] = 6;
    Qdegree[6] = 8;
    Qdegree[7] = 10;
    Qdegree[8] = 12;
    Qdegree[9] = 14;
    Qdegree[10] = 16;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap
    dofmap_ = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto [points, weights] = basix::quadrature::make_quadrature(
        basix::quadrature::type::gll, basix::cell::type::hexahedron, Qdegree[Q]);

    // Tabulate basis functions at quadrature points
    auto element = basix::create_element(basix::element::family::P, basix::cell::type::hexahedron,
                                         P, basix::element::lagrange_variant::gll_warped);

    auto [basis, shape] = element.tabulate(0, points, {weights.size(), 3});
    phi.resize(Nd * Nq);
    std::copy(basis.begin(), basis.end(), phi.begin());

    // Compute the scaled of the Jacobian determinant
    detJ_ = compute_scaled_jacobian_determinant(mesh, points, weights);
  }

  /// Operator y = Mx
  /// @param[in] x Input vector
  /// @param[in] coeffs Coefficients
  /// @param[out] y Output vector
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs, la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    T* phi_ = phi.data();

    for (std::int32_t c = 0; c < Nc; ++c) {
      // Get cell degrees of freedom
      auto cell_dofs = dofmap_.links(c);

      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[cell_dofs[i]];

      std::fill(y_.begin(), y_.end(), 0.0);
      T* detJ = detJ_.data() + c * Nq;
      mass::transform<T, Nd, Nq>(detJ, phi_, coeffs[c], x_.data(), y_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[cell_dofs[i]] += y_[i];
    }
  }

private:
  // Number of dofs per element
  static constexpr int Nd = (P + 1) * (P + 1) * (P + 1);

  // Number of quadrature points per element
  static constexpr int Nq = Q * Q * Q;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled Jacobian determinant
  std::vector<T> detJ_;

  // Dofmap
  graph::AdjacencyList<std::int32_t> dofmap_;

  // Basis function
  std::vector<T> phi;

  // Local input array
  std::array<T, Nd> x_;

  // Local output array
  std::array<T, Nd> y_;
};

// ------------------- //
// Stiffness operators //
// ------------------- //

namespace stiffness {
template <typename T, int Nd, int Nq>
inline void transform2(T* __restrict__ G, T* __restrict__ dphi, T __restrict__ coeff,
                       T* __restrict__ in, T* __restrict__ out) {
  for (int iq = 0; iq < Nq; ++iq) {

    T w0 = 0.0;
    T w1 = 0.0;

    for (int id = 0; id < Nd; ++id) {
      w0 += in[id] * dphi[iq * Nd + id];           // dx
      w1 += in[id] * dphi[Nd * Nq + iq * Nd + id]; // dy
    }

    const T* _G = G + iq * 3;
    const T fw0 = coeff * (_G[0] * w0 + _G[1] * w1);
    const T fw1 = coeff * (_G[1] * w0 + _G[2] * w1);

    for (int i = 0; i < Nd; ++i) {
      out[i] += fw0 * dphi[iq * Nd + i] + fw1 * dphi[Nd * Nq + iq * Nd + i];
    }
  }
}

template <typename T, int Nd, int Nq>
inline void transform3(T* __restrict__ G, T* __restrict__ dphi, T __restrict__ coeff,
                       T* __restrict__ in, T* __restrict__ out) {
  for (int iq = 0; iq < Nq; ++iq) {

    T w0 = 0.0;
    T w1 = 0.0;
    T w2 = 0.0;

    for (int id = 0; id < Nd; ++id) {
      w0 += in[id] * dphi[iq * Nd + id];               // dx
      w1 += in[id] * dphi[Nd * Nq + iq * Nd + id];     // dy
      w2 += in[id] * dphi[2 * Nd * Nq + iq * Nd + id]; // dz
    }

    const T* _G = G + iq * 6;
    const T fw0 = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
    const T fw1 = coeff * (_G[1] * w0 + _G[3] * w1 + _G[4] * w2);
    const T fw2 = coeff * (_G[2] * w0 + _G[4] * w1 + _G[5] * w2);

    for (int i = 0; i < Nd; ++i) {
      out[i] += fw0 * dphi[iq * Nd + i] + fw1 * dphi[Nd * Nq + iq * Nd + i]
                + fw2 * dphi[2 * Nd * Nq + iq * Nd + i];
    }
  }
}
} // namespace stiffness

/// Precompute Stiffness operator
template <typename T, int P, int Q>
class Stiffness2D {
public:
  Stiffness2D(std::shared_ptr<fem::FunctionSpace>& V) : dofmap_(0) {
    // Create map between number of quadrature points to basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[3] = 4;
    Qdegree[4] = 5;
    Qdegree[5] = 6;
    Qdegree[6] = 8;
    Qdegree[7] = 10;
    Qdegree[8] = 12;
    Qdegree[9] = 14;
    Qdegree[10] = 16;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap
    dofmap_ = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto [points, weights] = basix::quadrature::make_quadrature(
        basix::quadrature::type::gll, basix::cell::type::quadrilateral, Qdegree[Q]);

    // Tabulate basis functions at quadrature points
    auto element
        = basix::create_element(basix::element::family::P, basix::cell::type::quadrilateral, P,
                                basix::element::lagrange_variant::gll_warped);

    auto [basis, shape] = element.tabulate(1, points, {weights.size(), 2});
    dphi.resize(2 * Nd * Nq);
    std::copy(basis.begin() + Nd * Nq, basis.end(), dphi.begin());

    // Compute the scaled of the geometrical factor
    G_ = compute_scaled_geometrical_factor(mesh, points, weights);
  }

  /// Operator y = Kx
  /// @param[in] x Input vector
  /// @param[in] coeffs Coefficients
  /// @param[out] y Output vector
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs, la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    T* dphi_ = dphi.data();
    for (std::int32_t c = 0; c < Nc; ++c) {
      // Get cell degrees of freedom
      auto cell_dofs = dofmap_.links(c);

      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[cell_dofs[i]];

      std::fill(y_.begin(), y_.end(), 0.0);
      T* G = G_.data() + c * Nq * 3;
      stiffness::transform2<T, Nd, Nq>(G, dphi_, coeffs[c], x_.data(), y_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[cell_dofs[i]] += y_[i];
    }
  }

private:
  // Number of dofs per element
  static constexpr int Nd = (P + 1) * (P + 1);

  // Number of quadrature points per element
  static constexpr int Nq = Q * Q;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled geometrical factor
  std::vector<T> G_;

  // Dofmap
  graph::AdjacencyList<std::int32_t> dofmap_;

  // Derivatives of basis function
  std::vector<T> dphi;

  // Local input array
  std::array<T, Nd> x_;

  // Local output array
  std::array<T, Nd> y_;
};

template <typename T, int P, int Q>
class Stiffness3D {
public:
  Stiffness3D(std::shared_ptr<fem::FunctionSpace>& V) : dofmap_(0) {
    // Create map between number of quadrature points to basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[3] = 4;
    Qdegree[4] = 5;
    Qdegree[5] = 6;
    Qdegree[6] = 8;
    Qdegree[7] = 10;
    Qdegree[8] = 12;
    Qdegree[9] = 14;
    Qdegree[10] = 16;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap
    dofmap_ = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto [points, weights] = basix::quadrature::make_quadrature(
        basix::quadrature::type::gll, basix::cell::type::hexahedron, Qdegree[Q]);

    // Tabulate basis functions at quadrature points
    auto element = basix::create_element(basix::element::family::P, basix::cell::type::hexahedron,
                                         P, basix::element::lagrange_variant::gll_warped);

    auto [basis, shape] = element.tabulate(1, points, {weights.size(), 3});
    dphi.resize(3 * Nd * Nq);
    std::copy(basis.begin() + Nd * Nq, basis.end(), dphi.begin());

    // Compute the scaled of the geometrical factor
    G_ = compute_scaled_geometrical_factor(mesh, points, weights);
  }

  /// Operator y = Kx
  /// @param[in] x Input vector
  /// @param[in] coeffs Coefficients
  /// @param[out] y Output vector
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs, la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    T* dphi_ = dphi.data();
    for (std::int32_t c = 0; c < Nc; ++c) {
      // Get cell degrees of freedom
      auto cell_dofs = dofmap_.links(c);

      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[cell_dofs[i]];

      std::fill(y_.begin(), y_.end(), 0.0);
      T* G = G_.data() + c * Nq * 6;
      stiffness::transform3<T, Nd, Nq>(G, dphi_, coeffs[c], x_.data(), y_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[cell_dofs[i]] += y_[i];
    }
  }

private:
  // Number of dofs per element
  static constexpr int Nd = (P + 1) * (P + 1) * (P + 1);

  // Number of quadrature points per element
  static constexpr int Nq = Q * Q * Q;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled geometrical factor
  std::vector<T> G_;

  // Dofmap
  graph::AdjacencyList<std::int32_t> dofmap_;

  // Derivatives of basis function
  std::vector<T> dphi;

  // Local input array
  std::array<T, Nd> x_;

  // Local output array
  std::array<T, Nd> y_;
};