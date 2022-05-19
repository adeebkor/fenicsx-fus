#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "Fastor/Fastor.h"

#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

using namespace Fastor;

namespace {
  template <typename T, int P>
  static inline void transform_fastor_spectral(
    T* __restrict__ G, T* __restrict__ in0, 
    T* __restrict__ in1, T* __restrict__ in2,
    T* __restrict__ out0, T* __restrict__ out1,
    T* __restrict__ out2) {
    double c0 = 1500.0;
    double coeff = - 1.0 * (c0 * c0);
    constexpr int nq = (P + 1) * (P + 1) * (P + 1);
    for (int iq = 0; iq < nq; iq++) {
      const double* _G = G + iq * 6;
      const T w0 = in0[iq];
      const T w1 = in1[iq];
      const T w2 = in2[iq];
      out0[iq] = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
      out1[iq] = coeff * (_G[1] * w0 + _G[3] * w1 + _G[4] * w2);
      out2[iq] = coeff * (_G[2] * w0 + _G[4] * w1 + _G[5] * w2);
    }
  }
}

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
class StiffnessSpectralFastor {
public:
  StiffnessSpectralFastor(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
    // Create map between basis degree and quadrature degree
    std::map<int, int> qdegree;
    qdegree[2] = 3;
    qdegree[3] = 4;
    qdegree[4] = 6;
    qdegree[5] = 8;
    qdegree[6] = 10;
    qdegree[7] = 12;
    qdegree[8] = 14;
    qdegree[9] = 16;
    qdegree[10] = 18;

    // Get mesh and mesh attributes
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    _num_cells = mesh->topology().index_map(tdim)->size_local();

    // Get dofmap and permute
    auto _dofmap = V->dofmap()->list().array();
    _perm_dofmap.reserve(_dofmap.size());
    reorder_dofmap(_perm_dofmap, _dofmap, P);

    // Tabulate quadrature points and weights
    auto cell_type = basix::cell::type::hexahedron;
    auto quad_type = basix::quadrature::type::gll;
    auto [points, weights]
      = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[P]);

    // Compute the scaled of the geometrical factor
    auto J = compute_jacobian(mesh, points);
    auto _detJ = compute_jacobian_determinant(J);
    _G = compute_geometrical_factor(J, _detJ, weights);

    // Tabulate the basis functions and clamped values
    auto dphi = tabulate_1d(P, qdegree[P], 1);
    std::copy_n(dphi.data(), dphi.size(), _dphi.data());

    // Transpose dphi
    _dphiT = permute<Index<1, 0>>(_dphi);

  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();

    for (std::int32_t cell = 0; cell < _num_cells; cell++) {

      // Pack coefficients
      T* _x = xi.data();
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[_perm_dofmap[cell * _num_dofs + i]];
      }

      // Apply contraction in the x-direction
      _fw0 = einsum<Index<a0, b0>, Index<b0, b1, b2>>(_dphi, xi);

      // Apply contraction in the y-direction
      _fw1 = permute<Index<1, 0, 2>>(xi);
      _fw1 = einsum<Index<a1, b1>, Index<b1, a0, b2>>(_dphi, _fw1);
      _fw1 = permute<Index<1, 0, 2>>(_fw1);

      // Apply contraction in the z-direction
      _fw2 = permute<Index<2, 0, 1>>(xi);
      _fw2 = einsum<Index<a2, b2>, Index<b2, a0, a1>>(_dphi, _fw2);
      _fw2 = permute<Index<1, 2, 0>>(_fw2);

      // Apply transform
      T* G = _G.data() + cell * _num_quads * 6;
      T* fw0 = _fw0.data();
      T* fw1 = _fw1.data();
      T* fw2 = _fw2.data(); 
      T* y0 = _y0.data();
      T* y1 = _y1.data();
      T* y2 = _y2.data();
      transform_fastor_spectral<T, P>(G, fw0, fw1, fw2, y0, y1, y2);

      // Apply contraction in the x-direction
      _y0 = einsum<Index<a0, b0>, Index<b0, b1, b2>>(_dphiT, _y0);

      // Apply contraction in the y-direction
      _y1 = permute<Index<1, 0, 2>>(_y1);
      _y1 = einsum<Index<a1, b1>, Index<b1, a0, b2>>(_dphiT, _y1);
      _y1 = permute<Index<1, 0, 2>>(_y1);

      // Apply contraction in the z-direction
      _y2 = permute<Index<2, 0, 1>>(_y2);
      _y2 = einsum<Index<a2, b2>, Index<b2, a0, a1>>(_dphiT, _y2);
      _y2 = permute<Index<1, 2, 0>>(_y2);

      for (std::size_t i = 0; i < _num_dofs; i++) {
        y_array[_perm_dofmap[cell * _num_dofs + i]] += y0[i] + y1[i] + y2[i];
      }
    }
  }

private:
  // Number of dofs in each direction
  static constexpr int N = P + 1;

  // Number of degrees of freedom per element
  static constexpr int _num_dofs = (P + 1) * (P + 1) * (P + 1);

  // Number of quadrature points per element
  static constexpr int _num_quads = (P + 1) * (P + 1) * (P + 1);

  // Number of cells in the mesh
  std::int32_t _num_cells;

  // Scaled geometrical factor
  // xt::xtensor<T, 4> _G;
  xt::xtensor<T, 3> _G;

  // Basis functions in 1D
  Fastor::Tensor<T, P + 1, P + 1> _dphi;
  Fastor::Tensor<T, P + 1, P + 1> _dphiT; // transpose

  // Tensors for the stiffness operators
  Fastor::Tensor<T, N, N, N> xi;
  Fastor::Tensor<T, N, N, N> _y0;
  Fastor::Tensor<T, N, N, N> _y1;
  Fastor::Tensor<T, N, N, N> _y2;
  Fastor::Tensor<T, N, N, N> _fw0;
  Fastor::Tensor<T, N, N, N> _fw1;
  Fastor::Tensor<T, N, N, N> _fw2;

  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<std::int32_t> _perm_dofmap;
};
