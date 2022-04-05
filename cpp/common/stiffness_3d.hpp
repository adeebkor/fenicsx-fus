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
  inline void transform(T* __restrict__ G, T* __restrict__ dphi, T* __restrict__ in, T* __restrict out) {
    constexpr int nq = Q * Q * Q;
    constexpr int nd = (P + 1) * (P + 1) * (P + 1);
    double c0 = 1500.0;
    double coeff = - 1.0 * (c0 * c0);
    for (int iq = 0; iq < nq; iq++) {
      const double* _G = G + iq * 6;
      T w0 = 0.0;
      T w1 = 0.0;
      T w2 = 0.0;
      for (int id = 0; id < nd; id++) {
        w0 += in[id] * dphi[iq * nd + id]; // dx
        w1 += in[id] * dphi[nd*nq + iq*nd + id]; // dy
        w2 += in[id] * dphi[2*nd*nq + iq*nd + id]; // dz
      }
      const double fw0 = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
      const double fw1 = coeff * (_G[1] * w0 + _G[3] * w1 + _G[4] * w2);
      const double fw2 = coeff * (_G[2] * w0 + _G[4] * w1 + _G[5] * w2);
      for (int i = 0; i < nd; i++){
        out[i] += fw0 * dphi[iq*nd + i] + fw1 * dphi[nd*nq + iq*nd + i] 
                + fw2 * dphi[2*nd*nq + iq*nd + i];
      }
    }
  }
} // namespace

template <typename T, int P, int Q>
class Stiffness {
public:
  Stiffness(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
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
    _dofmap = V->dofmap()->list();

    // Tabulate quadrature points and weights
    auto cell_type = basix::cell::type::hexahedron;
    auto quad_type = basix::quadrature::type::gll;
    auto [points, weights]
      = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[P]);

    // Get basis functions and clamp values
    auto family = basix::element::family::P;
    auto variant = basix::element::lagrange_variant::gll_warped;
    auto element = basix::create_element(family, cell_type, P, variant);
    xt::xtensor<double, 4> table = element.tabulate(1, points);
    _dphi = xt::view(table, xt::range(1, tdim+1), xt::all(), xt::all(), 0);
    xt::filtration(_dphi, xt::isclose(_dphi, -1.0)) = -1.0;
    xt::filtration(_dphi, xt::isclose(_dphi, 0.0)) = 0.0;
    xt::filtration(_dphi, xt::isclose(_dphi, 1.0)) = 1.0;

    // Compute the scaled of the geometrical factor
    auto J = compute_jacobian(mesh, points);
    auto _detJ = compute_jacobian_determinant(J);
    _G = compute_geometrical_factor(J, _detJ, weights);
  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();

    for (std::int32_t cell = 0; cell < _num_cells; cell++) {
      auto cell_dofs = _dofmap.links(cell);
      
      // Pack coefficients
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[cell_dofs[i]];
      }

      std::fill(_y.begin(), _y.end(), 0.0);
      double* G = _G.data() + cell * _num_quads * 6;
      double* dphi = _dphi.data();
      transform<T, P, Q>(G, dphi, _x.data(), _y.data());

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

  // Scaled geometrical factor
  xt::xtensor<T, 3> _G;

  // Local input array
  std::array<T, _num_dofs> _x;

  // Local output array
  std::array<T, _num_dofs> _y;

  // Basis functions
  xt::xtensor<T, 3> _dphi;

  // Dofmap
  graph::AdjacencyList<std::int32_t> _dofmap;
};