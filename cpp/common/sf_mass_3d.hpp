#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "sum_factorisation.hpp"

#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace {
  template <typename T, int Q>
  inline void scale_coefficients_sf(T* __restrict__ detJ, T* __restrict__ fw) {
    constexpr int nq = Q * Q * Q;
    for (int iq = 0; iq < nq; iq++) {
      fw[iq] = fw[iq] * detJ[iq];
    }
  }
} // namespace

template <typename T, int P, int Q>
class MassSF {
public:
  MassSF(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
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

    // Tabulate the 1D basis functions
    _phi = tabulate_1d(P, qdegree[Q], 0);

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

    T* phi = _phi.data();
    T* fw = _fw.data();

    Buffer<T, Nq, Nd> buffer0;
    Buffer<T, Nd, Nq> buffer1;

    for (std::int32_t cell = 0; cell < _num_cells; cell++) {
      
      // Pack coefficients
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[_perm_dofmap[cell * _num_dofs + i]];
      }

      // Evaluate coefficients at quadrature points by apply 
      // three consecutive tensor contractions
      // coeff_q = \sum_i phi_qi * u_i
      apply_contractions<T, Nq, Nd, true>(phi, phi, phi, _x.data(), fw, buffer0);

      // Scale coefficients
      // fw = coeff * detJ * weight
      double* detJ = _detJ.data() + cell * _num_quads;
      scale_coefficients_sf<T, Q>(detJ, fw);

      // Accumulate contributions points by applying
      // three succesive tensor contractions
      // y_i = \sum_i phi_qi * fw_q
      apply_contractions<T, Nd, Nq, false>(phi, phi, phi, fw, _y.data(), buffer1);

      for (std::int32_t i = 0; i < _num_dofs; i++) {
        y_array[_perm_dofmap[cell * _num_dofs + i]] += _y[i];
      }
    }
  }

private:
  // Number of dofs in each direction
  static constexpr int Nd = P + 1;

  // Number of dofs per element
  static constexpr int _num_dofs = (P + 1) * (P + 1) * (P + 1);

  // Number of quadrature points in each direction
  static constexpr int Nq = Q;

  // Number of quadrature points per element
  static constexpr int _num_quads = Q * Q * Q;

  // Number of cells in the mesh
  std::int32_t _num_cells;

  // Determinant of the Jacobian scaled by the quadrature weights
  xt::xtensor<T, 2> _detJ;

  // Coefficients at quadrature point
  std::array<T, _num_quads> _fw;

  // Local input array
  std::array<T, _num_dofs> _x;

  // Local output array
  std::array<T, _num_dofs> _y;

  // Basis functions in 1D
  xt::xtensor_fixed<double, xt::fixed_shape<Q, P+1>> _phi;

  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<std::int32_t> _perm_dofmap;
};