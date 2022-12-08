#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "sum_factorisation.hpp"

#include <cmath>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;


// -------------- //
// Mass operators //
// -------------- //

namespace mass {
  template <typename T, int P, int Nq>
  inline void transform(T* __restrict__ detJ, T __restrict__ coeff, T* __restrict__ fw) {
    
    for (int iq = 0; iq < Nq; ++iq)
      fw[iq] = coeff * fw[iq] * detJ[iq];
  }
}

template <typename T, int P>
class MassSpectral2D {
public:
  MassSpectral2D(std::shared_ptr<fem::FunctionSpace>& V) {

    // Create a map between polynomial degree and basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[2] = 3;
    Qdegree[3] = 4;
    Qdegree[4] = 6;
    Qdegree[5] = 8;
    Qdegree[6] = 10;
    Qdegree[7] = 12;
    Qdegree[8] = 14;
    Qdegree[9] = 16;
    Qdegree[10] = 18;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap and reorder
    dofmap_ = V->dofmap()->list().array();
    tensor_dofmap_.resize(dofmap_.size());
    reorder_dofmap(tensor_dofmap_, dofmap_, basix::cell::type::quadrilateral, P);
    
    // Tabulate quadrature points and weights
    auto [points, weights]
      = basix::quadrature::make_quadrature(
          basix::quadrature::type::gll,
          basix::cell::type::quadrilateral,
          Qdegree[P]);

    // Compute the scaled of the Jacobian determinant
    detJ_ = compute_scaled_jacobian_determinant(mesh, points, weights);
  }

  /// Operator y = Mx
  /// @param[in] x Input vector
  /// @param[in] coeffs Coefficients
  /// @param[out] y Output vector
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs,
                  la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    for (std::int32_t c = 0; c < Nc; ++c)
    {
      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[tensor_dofmap_[c * Nd + i]];

      double* sdetJ = detJ_.data() + c * Nd;
      mass::transform<T, P, Nd>(sdetJ, coeffs[c], x_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[tensor_dofmap_[c * Nd + i]] += x_[i];
    }
  }

private:
  // Number of dofs and quadrature points in 1D
  static constexpr int N = (P + 1);

  // Number of dofs and quadrature points per element
  static constexpr int Nd = N * N;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled Jacobian determinant
  std::vector<T> detJ_;

  // Dofmap
  std::vector<std::int32_t> dofmap_;
  std::vector<std::int32_t> tensor_dofmap_;

  // Local input array
  std::array<T, Nd> x_;
};

template <typename T, int P>
class MassSpectral3D {
public:
  
};


// ------------------- //
// Stiffness operators //
// ------------------- //

namespace stiffness {
  template <typename T, int P>
  inline void transform(T* __restrict__ G, T __restrict__ coeff, 
                        T* __restrict__ fw0, T* __restrict__ fw1) {

    constexpr int nq = (P + 1) * (P + 1);

    for (int iq = 0; iq < nq; ++iq)
    {
      const T* _G = G + iq * 3;
      const T w0 = fw0[iq];
      const T w1 = fw1[iq];

      fw0[iq] = coeff * (_G[2] * w0 + _G[1] * w1);
      fw1[iq] = coeff * (_G[1] * w0 + _G[0] * w1);

    }
  }
}

template <typename T, int P>
class StiffnessSpectral2D {
public:
  StiffnessSpectral2D(std::shared_ptr<fem::FunctionSpace>& V) {

    // Create a map between polynomial degree and basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[2] = 3;
    Qdegree[3] = 4;
    Qdegree[4] = 6;
    Qdegree[5] = 8;
    Qdegree[6] = 10;
    Qdegree[7] = 12;
    Qdegree[8] = 14;
    Qdegree[9] = 16;
    Qdegree[10] = 18;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    Nc = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap and reorder
    dofmap_ = V->dofmap()->list().array();
    tensor_dofmap_.resize(dofmap_.size());
    reorder_dofmap(tensor_dofmap_, dofmap_, basix::cell::type::quadrilateral, P);

    // Tabulate quadrature points and weights
    auto [points, weights]
      = basix::quadrature::make_quadrature(
          basix::quadrature::type::gll,
          basix::cell::type::quadrilateral,
          Qdegree[P]);

    // Compute the scaled of the geometrical factor
    G_ = compute_scaled_geometrical_factor(mesh, points, weights);
  
    // Get the derivative data
    std::vector<double> basis = tabulate_1d(P, Qdegree[P], 1);
    dphi_ = basis.data() + (P + 1) * (P + 1);

    // Get the transpose of the basis
    dphiT_ = dphiT.data();
    transpose<T, N, N, 1, N>(dphi_, dphiT_);

  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs,
                  la::Vector<T, Alloc>& y) {
    
    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    T* fw0 = fw0_.data();
    T* fw1 = fw1_.data();

    for (std::int32_t c = 0; c < Nc; ++c) 
    {
      // Pack coefficients
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[tensor_dofmap_[c * Nd + i]];

      T1.fill(0.0);
      T2.fill(0.0);

      // Apply contraction in the x-direction
      fw0_.fill(0.0);
      contract<T, N, N, N>(x_.data(), dphi_, fw0);  // [i1, i2] x [i2, q2] -> [i1, q2]

      // Apply contraction in the y-direction
      fw1_.fill(0.0);
      transpose<T, N, N, 1, N>(x_.data(), T1.data());  // [i1, i2] -> [i2, i1]
      contract<T, N, N, N>(T1.data(), dphi_, T2.data());  // [i2, i1] x [i1, q1] -> [i2, q1]
      transpose<T, N, N, 1, N>(T2.data(), fw1);  // [i2, q1] -> [q1, i2]

      // Apply transform
      T* G = G_.data() + c * Nd * 3;
      stiffness::transform<T, P>(G, coeffs[c], fw0, fw1);

      T1.fill(0.0);
      T2.fill(0.0);

      // Apply contraction in the x-direction
      y0_.fill(0.0);
      contract<T, N, N, N>(fw0, dphiT_, y0_.data());  // [i1, q2] x [q2, i2] -> [i1, i2]

      // Apply contraction in the y-direction
      y1_.fill(0.0);
      transpose<T, N, N, 1, N>(fw1, T1.data()); // [q1, i2] -> [i2, q1]
      contract<T, N, N, N>(T1.data(), dphiT_, T2.data());  // [i2, q1] x [q1, i1] -> [i2, i1]
      transpose<T, N, N, 1, N>(T2.data(), y1_.data()); // [i1, i2]

      for (std::int32_t i = 0; i < Nd; ++i) {
        y_array[tensor_dofmap_[c * Nd + i]] += y0_[i] + y1_[i];
      }
    }
  }

private:
  // Number of dofs and quadrature points in 1D
  static constexpr int N = (P + 1);

  // Number of dofs and quadrature points per element
  static constexpr int Nd = N * N;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled geometrical factor
  std::vector<T> G_;

  // Derivative of the 1D basis functions
  const T* dphi_;
  
  // Transpose of derivative of the 1D basis functions
  std::array<T, N*N> dphiT;
  T* dphiT_;

  // Dofmap
  std::vector<std::int32_t> dofmap_;
  std::vector<std::int32_t> tensor_dofmap_;

  // Coefficients at quadrature point
  std::array<T, Nd> fw0_;
  std::array<T, Nd> fw1_;

  // Local input array
  std::array<T, Nd> x_;

  // Local output tensor
  std::array<T, Nd> y0_;
  std::array<T, Nd> y1_;

  // Working arrays
  std::array<T, Nd> T1;
  std::array<T, Nd> T2;

};