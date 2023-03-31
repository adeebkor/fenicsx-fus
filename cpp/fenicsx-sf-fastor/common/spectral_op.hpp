#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "Fastor/Fastor.h"

#include <cmath>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

using namespace Fastor;

// -------------- //
// Mass operators //
// -------------- //

namespace mass {
  template <typename T, int P, int Nq>
  inline void transform(T* __restrict__ detJ, T& __restrict__ coeff, T* __restrict__ fw) {
    
    for (int iq = 0; iq < Nq; ++iq)
      fw[iq] = coeff * fw[iq] * detJ[iq];
  }
}

/// 3D Spectral Mass operator
template <typename T, int P>
class MassSpectral3D {
public:
  MassSpectral3D(std::shared_ptr<fem::FunctionSpace>& V) {

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
    reorder_dofmap(tensor_dofmap_, dofmap_, basix::cell::type::hexahedron, P);

    // Tabulate quadrature points and weights
    auto [points, weights]
      = basix::quadrature::make_quadrature(
          basix::quadrature::type::gll,
          basix::cell::type::hexahedron,
          Qdegree[P]);

    // Compute the scaled of the Jacobian determinant
    detJ_ = compute_scaled_jacobian_determinant<T>(mesh, points, weights);
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

      T* sdetJ = detJ_.data() + c * Nd;
      mass::transform<T, P, Nd>(sdetJ, coeffs[c], x_.data());

      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[tensor_dofmap_[c * Nd + i]] += x_[i];
    }
  }

private:
  // Number of dofs and quadrature points in 1D
  static constexpr int N = (P + 1);

  // Number of dofs and quadrature points per element
  static constexpr int Nd = N * N * N;

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

// ------------------- //
// Stiffness operators //
// ------------------- //

namespace stiffness {
  template <typename T, int P, int Nq>
  inline void transform(T* __restrict__ G, T __restrict__ coeff,
                        T* __restrict__ fw0, T* __restrict__ fw1,
                        T* __restrict__ fw2) {

    for (int iq = 0; iq < Nq; ++iq)
    {
      const T* _G = G + iq * 6;
      const T w0 = fw0[iq];
      const T w1 = fw1[iq];
      const T w2 = fw2[iq];

      fw0[iq] = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
      fw1[iq] = coeff * (_G[1] * w0 + _G[3] * w1 + _G[4] * w2);
      fw2[iq] = coeff * (_G[2] * w0 + _G[4] * w1 + _G[5] * w2);
    }
  }
}  // namespace

enum
{
  a0,
  b0,
  a1,
  b1,
  a2,
  b2
};

template <typename T, int P>
class StiffnessSpectral3D {
public:
  StiffnessSpectral3D(std::shared_ptr<fem::FunctionSpace>& V) {

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
    reorder_dofmap(tensor_dofmap_, dofmap_, basix::cell::type::hexahedron, P);

    // Tabulate quadrature points and weights
    auto [points, weights]
      = basix::quadrature::make_quadrature(
          basix::quadrature::type::gll,
          basix::cell::type::hexahedron,
          Qdegree[P]);

    // Compute the scaled of the geometrical factor
    G_ = compute_scaled_geometrical_factor<T>(mesh, points, weights);
  
    // Get the derivative data
    std::vector<double> basis = tabulate_1d(P, Qdegree[P], 1);
    std::copy(basis.begin() + (P + 1) * (P + 1), basis.end(), dphi_.data());

    // Transpose derivative data
    dphiT_ = permute<Index<1, 0>>(dphi_);

  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, std::span<T> coeffs,
                  la::Vector<T, Alloc>& y) {

    std::span<const T> x_array = x.array();
    std::span<T> y_array = y.mutable_array();

    
    for (std::int32_t c = 0; c < Nc; ++c)
    {
      // Pack coefficients
      T* x_ = xi.data();
      for (std::int32_t i = 0; i < Nd; ++i)
        x_[i] = x_array[tensor_dofmap_[c * Nd + i]];

      // Apply contraction in the x-direction
      fw0_ = einsum<Index<a0, b0>, Index<b0, b1, b2>>(dphi_, xi);

      // Apply contraction in the y-direction
      fw1_ = permute<Index<1, 0, 2>>(xi);
      fw1_ = einsum<Index<a1, b1>, Index<b1, a0, b2>>(dphi_, fw1_);
      fw1_ = permute<Index<1, 0, 2>>(fw1_);

      // Apply contraction in the z-direction
      fw2_ = permute<Index<2, 0, 1>>(xi);
      fw2_ = einsum<Index<a2, b2>, Index<b2, a0, a1>>(dphi_, fw2_);
      fw2_ = permute<Index<1, 2, 0>>(fw2_);

      // Apply transform
      T* G = G_.data() + c * Nd * 6;
      stiffness::transform<T, P, Nd>(G, coeffs[c], 
                                     fw0_.data(), fw1_.data(), fw2_.data());

      // Apply contraction in the x-direction
      y0_ = einsum<Index<a0, b0>, Index<b0, b1, b2>>(dphiT_, fw0_);

      // Apply contraction in the y-direction
      y1_ = permute<Index<1, 0, 2>>(fw1_);
      y1_ = einsum<Index<a1, b1>, Index<b1, a0, b2>>(dphiT_, y1_);
      y1_ = permute<Index<1, 0, 2>>(y1_);

      // Apply contraction in the z-direction
      y2_ = permute<Index<2, 0, 1>>(fw2_);
      y2_ = einsum<Index<a2, b2>, Index<b2, a0, a1>>(dphiT_, y2_);
      y2_ = permute<Index<1, 2, 0>>(y2_);

      T* y0 = y0_.data();
      T* y1 = y1_.data();
      T* y2 = y2_.data();
      for (std::int32_t i = 0; i < Nd; ++i)
        y_array[tensor_dofmap_[c * Nd + i]] += y0[i] + y1[i] + y2[i];
    }
  }

private:
  // Number of dofs and quadrature points in 1D
  static constexpr int N = (P + 1);

  // Number of dofs and quadrature points per element
  static constexpr int Nd = N * N * N;

  // Number of cells in the mesh
  std::int32_t Nc;

  // Scaled geometrical factor
  std::vector<T> G_;

  // Derivative of the 1D basis functions
  Fastor::Tensor<T, N, N> dphi_;
  Fastor::Tensor<T, N, N> dphiT_;

  // Dofmap
  std::vector<std::int32_t> dofmap_;
  std::vector<std::int32_t> tensor_dofmap_;

  // Coefficients at quadrature point
  Fastor::Tensor<T, N, N, N> fw0_;
  Fastor::Tensor<T, N, N, N> fw1_;
  Fastor::Tensor<T, N, N, N> fw2_;

  // Local input tensor
  Fastor::Tensor<T, N, N, N> xi;

  // Local output tensor
  Fastor::Tensor<T, N, N, N> y0_;
  Fastor::Tensor<T, N, N, N> y1_;
  Fastor::Tensor<T, N, N, N> y2_;
};