#pragma once

#include "precompute.hpp"
#include "permute.hpp"
#include "sum_factorisation.hpp"

#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace {
  template <typename T, int P>
  static inline void transform_spectral(T* __restrict__ G, T* __restrict__ fw0, 
                                        T* __restrict__ fw1, T* __restrict__ fw2) {
    double c0 = 1500.0;
    double coeff = - 1.0 * (c0 * c0);
    constexpr int nq = (P + 1) * (P + 1) * (P + 1);
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
}

template <typename T, int P>
class StiffnessSpectral {
public:
  StiffnessSpectral(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {
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

    // Get dofmap
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
    _dphi = tabulate_1d(P, qdegree[P], 1);
  }

  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();

    const T* dphi = _dphi.data();
    T* fw0 = _fw0.data();
    T* fw1 = _fw1.data();
    T* fw2 = _fw2.data();

    // Reusable buffer for contraction
    Buffer<T, N, N> buffer;
    buffer.zero();


    for (std::int32_t cell = 0; cell < _num_cells; cell++) {
      
      // Pack coefficients
      for (std::int32_t i = 0; i < _num_dofs; i++) {
        _x[i] = x_array[_perm_dofmap[cell * _num_dofs + i]];
      }

      // Apply contraction in the x-direction
      buffer.zero();
      _fw0.fill(0.0);
      contract<T, N, N, N, N, true>(dphi, _x.data(), fw0);

      // Apply contraction in the y-direction
      buffer.zero();
      _fw1.fill(0.0);
      transpose<T, N, N, N, N, N*N, 1>(_x.data(), buffer.T0.data());
      contract<T, N, N, N, N, true>(dphi, buffer.T0.data(), buffer.T1.data());
      transpose<T, N, N, N, N, N*N, 1>(buffer.T1.data(), fw1);

      // Apply contraction in the z-direction
      buffer.zero();
      _fw2.fill(0.0);
      transpose<T, N, N, N, 1, N, N*N>(_x.data(), buffer.T0.data());
      contract<T, N, N, N, N, true>(dphi, buffer.T0.data(), buffer.T1.data());
      transpose<T, N, N, N, 1, N, N*N>(buffer.T1.data(), fw2);

      // Apply transform
      T* G = _G.data() + cell * _num_quads * 6;
      transform_spectral<T, P>(G, fw0, fw1, fw2);

      // Apply contraction in the x-direction
      buffer.zero();
      _y0.fill(0.0);
      contract<T, N, N, N, N, false>(dphi, fw0, _y0.data());
      
      // Apply contraction in the y-direction
      buffer.zero();
      _y1.fill(0.0);
      transpose<T, N, N, N, N, N*N, 1>(fw1, buffer.T0.data());
      contract<T, N, N, N, N, false>(dphi, buffer.T0.data(), buffer.T1.data());
      transpose<T, N, N, N, N, N*N, 1>(buffer.T1.data(), _y1.data());

      // Apply contraction in the z-direction
      buffer.zero();
      _y2.fill(0.0);
      transpose<T, N, N, N, 1, N, N*N>(fw2, buffer.T0.data());
      contract<T, N, N, N, N, false>(dphi, buffer.T0.data(), buffer.T1.data());
      transpose<T, N, N, N, 1, N, N*N>(buffer.T1.data(), _y2.data());

      for (std::int32_t i = 0; i < _num_dofs; i++) {
        y_array[_perm_dofmap[cell * _num_dofs + i]] += _y0[i] + _y1[i] + _y2[i];
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
  xt::xtensor<T, 3> _G;

  // Basis functions in 1D
  xt::xtensor_fixed<double, xt::fixed_shape<P+1, P+1>> _dphi;

  // Coefficients at quadrature point
  std::array<T, _num_quads> _fw0;
  std::array<T, _num_quads> _fw1;
  std::array<T, _num_quads> _fw2;

  // Local input array
  std::array<T, _num_dofs> _x;

  // Local output tensor
  std::array<T, _num_dofs> _y0;
  std::array<T, _num_dofs> _y1;
  std::array<T, _num_dofs> _y2;
  
  // Dofmap
  std::vector<std::int32_t> _dofmap;
  std::vector<std::int32_t> _perm_dofmap;
};