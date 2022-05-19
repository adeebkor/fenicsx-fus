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
  template <typename T, int Q>
  static inline void transform_fastor(
    T* __restrict__ G, T* __restrict__ fw0, 
    T* __restrict__ fw1, T* __restrict__ fw2) {
    double c0 = 1500.0;
    double coeff = - 1.0 * (c0 * c0);
    constexpr int nq = Q * Q * Q;
    for (int iq = 0; iq < nq; iq++) {
      const double* _G = G + iq * 6;
      const T w0 = fw0[iq];
      const T w1 = fw1[iq];
      const T w2 = fw2[iq];
      fw0[iq] = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
      fw1[iq] = coeff * (_G[1] * w0 + _G[3] * w1 + _G[4] * w2);
      fw2[iq] = coeff * (_G[2] * w0 + _G[4] * w1 + _G[5] * w2);
    }
  }
} // namespace

enum
{
  fa0,
  fb0,
  fa1,
  fb1,
  fa2,
  fb2
};

template <typename T, int P, int Q>
class StiffnessFastor {
public:
  StiffnessFastor(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
    // Create map between quadrature points to basix quadrature degree
    std::map<int, int> qdegree;
    qdegree[3] = 4;
    qdegree[4] = 5;
    qdegree[5] = 6;
    qdegree[6] = 8;
    qdegree[7] = 10;
    qdegree[8] = 12;
    qdegree[9] = 14;
    qdegree[10] = 16;

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
      = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[Q]);

    // Compute the scaled of the geometrical factor
    auto J = compute_jacobian(mesh, points);
    auto _detJ = compute_jacobian_determinant(J);
    _G = compute_geometrical_factor(J, _detJ, weights);

    // Tabulate the basis functions and clamped values
    auto phi = tabulate_1d(P, qdegree[Q], 0);
    auto dphi = tabulate_1d(P, qdegree[Q], 1);

    std::copy_n(phi.data(), phi.size(), _phi.data());
    std::copy_n(dphi.data(), dphi.size(), _dphi.data());

    // Transposes
    _phiT = permute<Index<1, 0>>(_phi);
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
      T0 = einsum<Index<fa0, fb0>, Index<fb0, fb1, fb2>>(_dphi, xi);
      T0_t = permute<Index<1, 0, 2>>(T0);
      T1 = einsum<Index<fa1, fb1>, Index<fb1, fa0, fb2>>(_phi, T0_t);
      T1_t = permute<Index<2, 0, 1>>(T1);
      T2 = einsum<Index<fa2, fb2>, Index<fb2, fa1, fa0>>(_phi, T1_t);
      _fw0 = permute<Index<2, 1, 0>>(T2);

      // Apply contraction in the y-direction
      T0 = einsum<Index<fa0, fb0>, Index<fb0, fb1, fb2>>(_phi, xi);
      T0_t = permute<Index<1, 0, 2>>(T0);
      T1 = einsum<Index<fa1, fb1>, Index<fb1, fa0, fb2>>(_dphi, T0_t);
      T1_t = permute<Index<2, 0, 1>>(T1);
      T2 = einsum<Index<fa2, fb2>, Index<fb2, fa1, fa0>>(_phi, T1_t);
      _fw1 = permute<Index<2, 1, 0>>(T2);

      // Apply contraction in the z-direction
      T0 = einsum<Index<fa0, fb0>, Index<fb0, fb1, fb2>>(_phi, xi);
      T0_t = permute<Index<1, 0, 2>>(T0);
      T1 = einsum<Index<fa1, fb1>, Index<fb1, fa0, fb2>>(_phi, T0_t);
      T1_t = permute<Index<2, 0, 1>>(T1);
      T2 = einsum<Index<fa2, fb2>, Index<fb2, fa1, fa0>>(_dphi, T1_t);
      _fw2 = permute<Index<2, 1, 0>>(T2);

      // Apply transform
      T* G = _G.data() + cell * _num_quads * 6;
      T* fw0 = _fw0.data();
      T* fw1 = _fw1.data();
      T* fw2 = _fw2.data(); 
      transform_fastor<T, Q>(G, fw0, fw1, fw2);

      // Apply contraction in the x-direction
      S0 = einsum<Index<fb0, fa0>, Index<fa0, fa1, fa2>>(_dphiT, _fw0);
      S0_t = permute<Index<1, 0, 2>>(S0);
      S1 = einsum<Index<fb1, fa1>, Index<fa1, fb0, fa2>>(_phiT, S0_t);
      S1_t = permute<Index<2, 0, 1>>(S1);
      S2 = einsum<Index<fb2, fa2>, Index<fa2, fb1, fb0>>(_phiT, S1_t);
      _y0 = permute<Index<2, 1, 0>>(S2);

      // Apply contraction in the y-direction
      S0 = einsum<Index<fb0, fa0>, Index<fa0, fa1, fa2>>(_phiT, _fw1);
      S0_t = permute<Index<1, 0, 2>>(S0);
      S1 = einsum<Index<fb1, fa1>, Index<fa1, fb0, fa2>>(_dphiT, S0_t);
      S1_t = permute<Index<2, 0, 1>>(S1);
      S2 = einsum<Index<fb2, fa2>, Index<fa2, fb1, fb0>>(_phiT, S1_t);
      _y1 = permute<Index<2, 1, 0>>(S2);

      // Apply contraction in the z-direction
      S0 = einsum<Index<fb0, fa0>, Index<fa0, fa1, fa2>>(_phiT, _fw2);
      S0_t = permute<Index<1, 0, 2>>(S0);
      S1 = einsum<Index<fb1, fa1>, Index<fa1, fb0, fa2>>(_phiT, S0_t);
      S1_t = permute<Index<2, 0, 1>>(S1);
      S2 = einsum<Index<fb2, fa2>, Index<fa2, fb1, fb0>>(_dphiT, S1_t);
      _y2 = permute<Index<2, 1, 0>>(S2);

      T* y0 = _y0.data();
      T* y1 = _y1.data();
      T* y2 = _y2.data();
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        y_array[_perm_dofmap[cell * _num_dofs + i]] += y0[i] + y1[i] + y2[i];
      }
    }
  }

private:
  // Number of dofs in each direction
  static constexpr int Nd = P + 1;

  // Number of degrees of freedom per element
  static constexpr int _num_dofs = (P + 1) * (P + 1) * (P + 1);

  // Number of quadrature points in each direction
  static constexpr int Nq = Q;

  // Number of quadrature points per element
  static constexpr int _num_quads = Q * Q * Q;

  // Number of cells in the mesh
  std::int32_t _num_cells;

  // Scaled geometrical factor
  xt::xtensor<T, 3> _G;

  // Basis functions in 1D
  Fastor::Tensor<T, Q, P + 1> _phi;
  Fastor::Tensor<T, P + 1, Q> _phiT;
  Fastor::Tensor<T, Q, P + 1> _dphi;
  Fastor::Tensor<T, P + 1, Q> _dphiT;

  // Tensors
  Fastor::Tensor<T, Nd, Nd, Nd> xi;
  Fastor::Tensor<T, Nd, Nd, Nd> _y0;
  Fastor::Tensor<T, Nd, Nd, Nd> _y1;
  Fastor::Tensor<T, Nd, Nd, Nd> _y2;
  Fastor::Tensor<T, Nq, Nq, Nq> _fw0;
  Fastor::Tensor<T, Nq, Nq, Nq> _fw1;
  Fastor::Tensor<T, Nq, Nq, Nq> _fw2;
  Fastor::Tensor<T, Nq, Nd, Nd> T0;
  Fastor::Tensor<T, Nd, Nq, Nd> T0_t;
  Fastor::Tensor<T, Nq, Nq, Nd> T1;
  Fastor::Tensor<T, Nd, Nq, Nq> T1_t;
  Fastor::Tensor<T, Nq, Nq, Nq> T2;
  Fastor::Tensor<T, Nd, Nq, Nq> S0;
  Fastor::Tensor<T, Nq, Nd, Nq> S0_t;
  Fastor::Tensor<T, Nd, Nd, Nq> S1;
  Fastor::Tensor<T, Nq, Nd, Nd> S1_t;
  Fastor::Tensor<T, Nd, Nd, Nd> S2;

  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<std::int32_t> _perm_dofmap;
};

