#pragma once

#include "precompute.hpp"
#include "permute.hpp"

#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace {
  template <typename T, int Q>
  inline void transform(T* __restrict__ detJ, T* __restrict__ in, T* __restrict__ out) {
    constexpr int nq = Q * Q * Q;
    for (int iq = 0; iq < nq; iq++) {
        out[iq] = in[iq] * detJ[iq];
    }
  }
} // namespace

template <typename T, int P, int Q>
class SpectralMass {
public:
  SpectralMass(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
    // Create a map between basis degree and quadrature degree
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

    // Get tensor product order
    auto family = basix::element::family::P;
    auto cell_type = basix::cell::type::hexahedron;
    auto variant = basix::element::lagrange_variant::gll_warped;
    auto element = basix::create_element(family, cell_type, P, variant);
    auto perm = std::get<1>(element.get_tensor_product_representation()[0]);
    std::copy(perm.begin(), perm.end(), _perm.begin());

    // Tabulate quadrature points and weights
    auto quad_type = basix::quadrature::type::gll;
    auto [points, weights]
      = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[P]);

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
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[_perm_dofmap[cell * _num_dofs + i]];
      }

      std::fill(_y.begin(), _y.end(), 0.0);
      double* detJ = _detJ.data() + cell * _num_quads;
      transform<T, Q>(detJ, _x.data(), _y.data());

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

  // Permutations: from basix to tensor product order
  std::array<int, _num_dofs> _perm;

  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<int> _perm_dofmap;

  // Local input array
  std::array<T, _num_dofs> _x;

  // Local output array
  std::array<T, _num_dofs> _y;
};
