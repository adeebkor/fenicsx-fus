#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "Fastor/Fastor.h"

#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

using namespace Fastor;

namespace {
  template <typename T, int Q>
  inline void scale_coefficients_fastor(T* __restrict__ detJ, T* __restrict__ fw) {
    constexpr int nq = Q * Q * Q;
    for (int iq = 0; iq < nq; iq++) {
        fw[iq] = fw[iq] * detJ[iq];
    }
  }
} // namespace

enum
{
  i0,
  i1,
  i2,
  q0,
  q1,
  q2
};

template <typename T, int P, int Q>
class MassFastor {
public:
  MassFastor(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
    // Create a map between quadrature points to basix quadrature degree
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

    // Tabulate the 1D basis function
    auto phi = tabulate_1d(P, qdegree[Q], 0);
    std::copy_n(phi.data(), phi.size(), _phi.data());

    // Transpose phi
    _phiT = permute<Index<1, 0>>(_phi);

    // Compute the scaled of the Jacobian
    auto J = compute_jacobian(mesh, points);
    _detJ = compute_jacobian_determinant(J);
    for (std::size_t i = 0; i < _detJ.shape(0); i++) {
      for (std::size_t j = 0; j < _detJ.shape(1); j++) {
        _detJ(i, j) = _detJ(i, j) * weights[j];
      }
    }
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

      // Apply contractions
      T0 = einsum<Index<q0, i0>, Index<i0, i1, i2>>(_phi, xi);
      T1 = einsum<Index<q1, i1>, Index<q0, i1, i2>>(_phi, T0);
      fw = einsum<Index<q2, i2>, Index<q0, q1, i2>>(_phi, T1);

      // Scale coefficients
      // fw = coeff * detJ * weight
      double* detJ = _detJ.data() + cell * _num_quads;
      scale_coefficients_fastor<T, Q>(detJ, fw.data());

      // Apply contractions
      S0 = einsum<Index<i0, q0>, Index<q0, q1, q2>>(_phiT, fw);
      S1 = einsum<Index<i1, q1>, Index<i0, q1, q2>>(_phiT, S0);
      yi = einsum<Index<i2, q2>, Index<i0, i1, q2>>(_phiT, S1);

      T* _y = yi.data();
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        y_array[_perm_dofmap[cell * _num_dofs + i]] += _y[i];
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

  // Determinant of the Jacobian scaled by the quadrature weights
  xt::xtensor<T, 2> _detJ;

  // Basis functions in 1D
  Fastor::Tensor<T, Q, P + 1> _phi;
  Fastor::Tensor<T, P + 1, Q> _phiT;

  // Tensors
  Fastor::Tensor<T, Nd, Nd, Nd> xi;
  Fastor::Tensor<T, Nq, Nd, Nd> T0;
  Fastor::Tensor<T, Nq, Nq, Nd> T1;
  Fastor::Tensor<T, Nq, Nq, Nq> fw;
  Fastor::Tensor<T, Nd, Nq, Nq> S0;
  Fastor::Tensor<T, Nd, Nd, Nq> S1;
  Fastor::Tensor<T, Nd, Nd, Nd> yi;

  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<std::int32_t> _perm_dofmap;
};