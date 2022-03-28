#pragma once

#include "precompute.hpp"
#include "precomputation.hpp"
#include "sum_factorisation.hpp"
#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>


std::pair<std::vector<int>, xt::xtensor<double, 4>>
tabulate_basis_and_permutation_hex(int p, int q){
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::hexahedron;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);

  xt::xtensor<double, 4> table = element.tabulate(1, points);
  std::vector<int> perm = std::get<1>(element.get_tensor_product_representation()[0]);

  // Clamp -1, 0, 1 values
  xt::filtration(table, xt::isclose(table, -1.0)) = -1.0;
  xt::filtration(table, xt::isclose(table, 0.0)) = 0.0;
  xt::filtration(table, xt::isclose(table, 1.0)) = 1.0;

  return {perm, table};
}

namespace {
  template <typename T>
  inline void mkernel(T* A, const T* w, const T* c, const double* detJ, int nq, int nd){
    for (int iq = 0; iq < nq; ++iq){
      A[iq] = w[iq] * detJ[iq];
    }
  }
} // namespace

template <typename T>
class MassOperator {
  public:
    MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim = mesh->topology().dim();
      _dofmap = V->dofmap()->list();
      _num_cells = mesh->topology().index_map(tdim)->size_local();
      _num_dofs = (bdegree + 1) * (bdegree + 1) * (bdegree + 1);
      _x.resize(_num_dofs);
      _y.resize(_num_dofs);

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

      // Get tensor product order
      auto family = basix::element::family::P;
      auto cell_type = basix::cell::type::hexahedron;
      auto variant = basix::element::lagrange_variant::gll_warped;
      auto element = basix::create_element(family, cell_type, bdegree, variant);
      _perm = std::get<1>(element.get_tensor_product_representation()[0]);

      // Tabulate quadrature points and weights
      auto quad_type = basix::quadrature::type::gll;
      auto [points, weights]
        = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[bdegree]);
      _num_quads = weights.size();

      // Compute the scaled determinant of the Jacobian
      J = compute_jacobian(mesh, points);
      _detJ = compute_jacobian_determinant(J);
      for (std::size_t i = 0; i < _detJ.shape(0); i++) {
          for (std::size_t j = 0; j < _detJ.shape(1); j++) {
              _detJ(i, j) = _detJ(i, j) * weights[j];
          }
      }
    }

    template <typename Alloc>
    void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
      xtl::span<const T> x_array = x.array();
      xtl::span<T> y_array = y.mutable_array();
      tcb::span<const int> cell_dofs;
      for (std::size_t cell = 0; cell < _num_cells; cell++){
        cell_dofs = _dofmap.links(cell);
        for (std::size_t i = 0; i < _num_dofs; i++) {
            _x[i] = x_array[cell_dofs[_perm[i]]];
        }

        std::fill(_y.begin(), _y.end(), 0.0);
        double* detJ_ptr = _detJ.data() + cell * _num_quads;
        mkernel<double>(_y.data(), _x.data(), nullptr, detJ_ptr, _num_quads, _num_dofs);
        for (std::size_t i = 0; i < _num_dofs; i++) {
            y_array[cell_dofs[_perm[i]]] += _y[i];
        }
      }
    }

  private:
    std::vector<T> _x;
    std::vector<T> _y;
    graph::AdjacencyList<std::int32_t> _dofmap;
    xt::xtensor<double, 4> J;
    xt::xtensor<double, 2> _detJ;
    std::vector<int> _perm;
    std::size_t _num_cells;
    std::size_t _num_dofs;
    std::size_t _num_quads;
};

namespace {
  template <typename T>
  inline void skernel(T* A, const T* w, std::map<std::string, double>& c, const double* G, const xt::xtensor<double, 3>& dphi, int nq, int nd){
    double c0 = 1.0;
    double coeff = 1.0 * c0 * c0;
    for (int iq = 0; iq < nq; iq++){
      const double* _G = G + iq * 9;
      double w0 = 0.0;
      double w1 = 0.0;
      double w2 = 0.0;
      for (int ic = 0; ic < nd; ic++){
        w0 += w[ic] * dphi(0, iq, ic); // dx
        w1 += w[ic] * dphi(1, iq, ic); // dy
        w2 += w[ic] * dphi(2, iq, ic); // dz
      }
      const double fw0 = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
      const double fw1 = coeff * (_G[3] * w0 + _G[4] * w1 + _G[5] * w2);
      const double fw2 = coeff * (_G[6] * w0 + _G[7] * w1 + _G[8] * w2);
      for (int i = 0; i < nd; i++){
        A[i] += fw0 * dphi(0, iq, i) + fw1 * dphi(1, iq, i) + fw2 * dphi(2, iq, i);
      }
    }
  }
}

template <typename T>
class StiffnessOperator {
  public:
    StiffnessOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree, std::map<std::string, double>& params) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim  = mesh->topology().dim();
      _dofmap = V->dofmap()->list();
      _num_cells = mesh->topology().index_map(tdim)->size_local();
      _num_dofs = (bdegree + 1) * (bdegree + 1) * (bdegree + 1);
      _x.resize(_num_dofs);
      _y.resize(_num_dofs);

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

      // Tabulate quadrature points and weights
      auto quad_type = basix::quadrature::type::gll;
      auto cell_type = basix::cell::type::hexahedron;
      auto [points, weights]
        = basix::quadrature::make_quadrature(quad_type, cell_type, qdegree[bdegree]);
      _num_quads = weights.size();

      // Tabulate the basis functions and clamped values
      auto family = basix::element::family::P;
      auto variant = basix::element::lagrange_variant::gll_warped;
      auto element = basix::create_element(family, cell_type, bdegree, variant);
      auto table = element.tabulate(1, points);
      _dphi = xt::view(table, xt::range(1, tdim+1), xt::all(), xt::all(), 0);
      xt::filtration(_dphi, xt::isclose(_dphi, -1.0)) = -1.0;
      xt::filtration(_dphi, xt::isclose(_dphi, 0.0)) = 0.0;
      xt::filtration(_dphi, xt::isclose(_dphi, 1.0)) = 1.0;

      // Compute the scaled of the geometrical factor
      auto J = compute_jacobian(mesh, points);
      auto _detJ = compute_jacobian_determinant(J);
      G = compute_geometrical_factor(J, _detJ, weights);

    }

    template <typename Alloc>
    void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
      xtl::span<const T> x_array = x.array();
      xtl::span<T> y_array = y.mutable_array();
      tcb::span<const int> cell_dofs;
      for (std::size_t cell = 0; cell < _num_cells; ++cell){
        cell_dofs = _dofmap.links(cell);
        for (std::size_t i = 0; i < _num_dofs; i++){
          _x[i] = x_array[cell_dofs[i]];
        }
        std::fill(_y.begin(), _y.end(), 0.0);
        double* G_cell = G.data() + cell * _num_quads * 9;
        skernel<double> (_y.data(), _x.data(), _params, G_cell, _dphi, _num_quads, _num_dofs);
        for (std::size_t i = 0; i < _num_dofs; i++){
          y_array[cell_dofs[i]] += _y[i];
        }
      }
    }

  private:
    std::vector<T> _x;
    std::vector<T> _y;
    std::size_t _num_cells;
    std::size_t _num_dofs;
    std::size_t _num_quads;
    graph::AdjacencyList<std::int32_t> _dofmap;
    xt::xtensor<double, 4> G;
    xt::xtensor<double, 3> _dphi;
    std::map<std::string, double> _params;
};

namespace {
template <typename T, int Q>
static inline void transform_coefficients(T* __restrict__ G, T* __restrict__ fw0,
                                          T* __restrict__ fw1, T* __restrict__ fw2) {
  double c0 = 1.0;
  double coeff = - 1.0 * (c0 * c0);
  constexpr int nquad = Q * Q * Q;
  for (int q = 0; q < nquad; q++) {
    const double* _G = G + q * 9;
    const T w0 = fw0[q];
    const T w1 = fw1[q];
    const T w2 = fw2[q];
    fw0[q] = coeff * (_G[0] * w0 + _G[1] * w1 + _G[2] * w2);
    fw1[q] = coeff * (_G[3] * w0 + _G[4] * w1 + _G[5] * w2);
    fw2[q] = coeff * (_G[6] * w0 + _G[7] * w1 + _G[8] * w2);
  }
}
}

template <typename T, int P, int Q>
class StiffnessOperatorSF {
  public:
    StiffnessOperatorSF(std::shared_ptr<fem::FunctionSpace>& V) : _dofmap(0) {

      // Get mesh and mesh attributes
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim = mesh->topology().dim();
      _num_cells = mesh->topology().index_map(tdim)->size_local();
      
      // Get dofmap
      _dofmap = V->dofmap()->list();

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

      // Compute the scaled of the geometrical factor
      auto J = compute_jacobian(mesh, points);
      auto _detJ = compute_jacobian_determinant(J);
      G = compute_geometrical_factor(J, _detJ, weights);

      // Tabulate the basis functions and clamped values
      _phi = tabulate_1d(P, qdegree[P], 0);
      _dphi = tabulate_1d(P, qdegree[P], 1);

      std::cout << _phi << std::endl;
      std::cout << _dphi << std::endl;
    }

    template <typename Alloc>
    void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
      y.set(0.0);
      xtl::span<const T> x_array = x.array();
      xtl::span<T> y_array = y.mutable_array();

      // Number of dofs in each direction
      constexpr int Nd = P + 1;

      // Number of quadrature points in each direction
      constexpr int Nq = Q;

      const T* phi = _phi.data();
      const T* dphi = _dphi.data();
      
      _fw0.fill(0.0);
      _fw1.fill(0.0);
      _fw2.fill(0.0);
      
      T* fw0 = _fw0.data();
      T* fw1 = _fw1.data();
      T* fw2 = _fw2.data();

      // Create reusable buffers for contraction
      Buffer<T, Nq, Nd> buffer0;
      Buffer<T, Nd, Nq> buffer1;

      for (std::size_t cell = 0; cell < _num_cells; cell++) {
        auto cell_dofs = _dofmap.links(cell);

        // Pack coefficients
        for (std::size_t i = 0; i < _num_dofs; i++) {
          _x[i] = x_array[cell_dofs[_perm[i]]];    
        }

        // Evaluate coefficients at quadrature points by applying
        // three successive tensor contractions
        // fw0 = \sum_i dphix_qi * u_i
        // fw1 = \sum_i dphiy_qi * u_i
        // fw2 = \sum_i dphiz_qi * u_i
        // apply_contractions<T, Nq, Nd, true>(dphi, phi, phi, _x.data(), fw0, buffer0);
        // apply_contractions<T, Nq, Nd, true>(phi, dphi, phi, _x.data(), fw1, buffer0);
        // apply_contractions<T, Nq, Nd, true>(phi, phi, dphi, _x.data(), fw2, buffer0);

        // double* G_cell = G.data() + cell * _num_quads * 9;
        // transform_coefficients<T, Q>(G_cell, fw0, fw1, fw2);

        // Accumulate contributions points by applying
        // three successive tensor contractions
        // y_i += \sum_i dphix_qi * fw0_q
        // y_i += \sum_i dphiy_qi * fw1_q
        // y_i += \sum_i dphiz_qi * fw2_q
        // apply_contractions<T, Nd, Nq, false>(dphi, phi, phi, fw0, _y0.data(), buffer1);
        // apply_contractions<T, Nd, Nq, false>(phi, dphi, phi, fw1, _y1.data(), buffer1);
        // apply_contractions<T, Nd, Nq, false>(phi, phi, dphi, fw2, _y2.data(), buffer1);

        common::Timer tadeeb("Adeeb time");

        tadeeb.start();
        apply_contraction_x<T, Nq, Nd, true>(dphi, _x.data(), fw0, buffer0);
        apply_contraction_y<T, Nq, Nd, true>(dphi, _x.data(), fw1, buffer0);
        apply_contraction_z<T, Nq, Nd, true>(dphi, _x.data(), fw2, buffer0);

        double* G_cell = G.data() + cell * _num_quads * 9;
        transform_coefficients<T, Q>(G_cell, fw0, fw1, fw2);

        apply_contraction_x<T, Nq, Nd, false>(dphi, fw0, _y0.data(), buffer1);
        apply_contraction_y<T, Nq, Nd, false>(dphi, fw1, _y1.data(), buffer1);
        apply_contraction_z<T, Nq, Nd, false>(dphi, fw2, _y2.data(), buffer1);
        for (int i = 0; i < 10; i++) {
            std::cout << _y2[i] << std::endl;
        }
        return;

        tadeeb.stop();

        std::cout << "Adeeb time: " << tadeeb.elapsed()[0] << std::endl;

        for (std::size_t i = 0; i < _num_dofs; i++) {
          y_array[cell_dofs[_perm[i]]] += _y0[i] + _y1[i] + _y2[i];
        }
      }
    }

  private:
    // Number of degrees of freedom per element
    static constexpr int _num_dofs = (P + 1) * (P + 1) * (P + 1);

    // Number of quadrature points per element
    static constexpr int _num_quads = Q * Q * Q;

    // Number of cells in the mesh
    std::size_t _num_cells;

    // Geometrical factor
    // [_num_cells]x[_num_quads]x[tdim]x[gdim]
    xt::xtensor<double, 4> G;

    // Local tensor input vector
    std::array<T, _num_dofs> _x;

    // Local tensor output vector
    std::array<T, _num_dofs> _y0;
    std::array<T, _num_dofs> _y1;
    std::array<T, _num_dofs> _y2;

    // Coefficients at quadrature point
    std::array<T, _num_quads> _fw0;
    std::array<T, _num_quads> _fw1;
    std::array<T, _num_quads> _fw2;

    // Permutations: from basix order to tensor product order
    std::array<int, _num_dofs> _perm;

    // Basis functions in 1D
    xt::xtensor_fixed<double, xt::fixed_shape<Q, P + 1>> _phi;
    xt::xtensor_fixed<double, xt::fixed_shape<Q, P + 1>> _dphi;

    // Dofmap
    graph::AdjacencyList<std::int32_t> _dofmap;
};