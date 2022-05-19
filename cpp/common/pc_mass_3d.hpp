#pragma once

#include "precompute.hpp"
#include "permute.hpp"

#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace {
  template <typename T, int P, int Q>
  inline void scale_coefficients(T* __restrict__ detJ, T* __restrict__ phi, T* __restrict__ in, T* __restrict__ out){
    constexpr int nq = Q * Q * Q;
    constexpr int nd = (P + 1) * (P + 1) * (P + 1);

    for (int iq = 0; iq < nq; iq++) {
      T w0 = 0.0;
      for (int id = 0; id < nd; id++) {
        w0 += in[id] * phi[iq * nd + id];
      }

      const double fw0 = w0 * detJ[iq];

      for (int i = 0; i < nd; i++) {
        out[i] += fw0 * phi[iq * nd + i];
      }
    }
  }
} // namespace

// namespace {
//   template <typename T, int P, int Q>
//   inline void transform(T* __restrict__ detJ, T* __restrict__ phi, T* __restrict__ in, T* __restrict__ out) {
//     constexpr int nq = Q * Q * Q;
//     constexpr int nd = (P + 1) * (P + 1) * (P + 1);

//     for (int i = 0; i < nd; i++) {
//       for (int iq = 0; iq < nq; iq++) {
//         T w0 = 0.0;
//         for (int id = 0; id < nd; id++) {
//           w0 += in[id] * phi[iq * nd + id]; // summing through columns
//         }

//         const double fw0 = w0 * detJ[iq];

//         out[i] += fw0 * phi[iq * nd + i]; // sum through rows
//       }
//     }
//   }
// }

template <typename T, int P, int Q>
class Mass {
public:
  Mass(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
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

    // Get dofmap
    _dofmap = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto cell_type = basix::cell::type::hexahedron;
    auto quad_type = basix::quadrature::type::gll;
    auto [points, weights]
      = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[Q]);

    // Tabulate the basis functions and clamp values
    auto family = basix::element::family::P;
    auto variant = basix::element::lagrange_variant::gll_warped;
    auto element = basix::create_element(family, cell_type, P, variant);
    xt::xtensor<double, 4> table = element.tabulate(0, points);
    _phi = xt::view(table, 0, xt::all(), xt::all());
    xt::filtration(_phi, xt::isclose(_phi, -1.0)) = -1.0;
    xt::filtration(_phi, xt::isclose(_phi, 0.0)) = 0.0;
    xt::filtration(_phi, xt::isclose(_phi, 1.0)) = 1.0;

    // Compute the scaled of the Jacobian
    auto J = compute_jacobian(mesh, points);
    _detJ = compute_jacobian_determinant(J);
    for (std::size_t i = 0; i < _detJ.shape(0); i++) {
      for (std::size_t j = 0; j < _detJ.shape(1); j++) {
        _detJ(i, j) = _detJ(i, j) * weights[j];
      }
    }

    assert(_phi.shape(0) == std::size_t(_num_quads));
    assert(_phi.shape(1) == std::size_t(_num_dofs));
  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();
      
    T* phi = _phi.data();

    for (std::int32_t cell = 0; cell < _num_cells; cell++) {
      auto cell_dofs = _dofmap.links(cell);

      // Pack coefficients
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[cell_dofs[i]];
      }

      std::fill(_y.begin(), _y.end(), 0.0);
      double* detJ = _detJ.data() + cell * _num_quads;
      scale_coefficients<T, P, Q>(detJ, phi, _x.data(), _y.data());

      for (std::int32_t i = 0; i < _num_dofs; i++) {
        y_array[cell_dofs[i]] += _y[i];
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

  // Local input array
  std::array<T, _num_dofs> _x;

  // Local output array
  std::array<T, _num_dofs> _y;

  // Basis functions
  xt::xtensor<T, 2> _phi;

  // Dofmap
  graph::AdjacencyList<std::int32_t> _dofmap;
};