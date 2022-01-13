#pragma once

#include "precomputation.hpp"
#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>


std::pair<std::vector<int>, xt::xtensor<double, 4>>
tabulate_basis_and_permutation(int p, int q){
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::quadrilateral;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);

  xt::xtensor<double, 4> table = element.tabulate(1, points);
  std::vector<int> perm = std::get<1>(element.get_tensor_product_representation()[0]);

  return {perm, table};
}

namespace {
  template <typename T>
  inline void mkernel(T* A, const T* w, const T* c, const double* detJ, double* phi, int nq, int nd){
    for (int iq = 0; iq < nq; ++iq){
      A[iq] = w[iq] * detJ[iq];
    }
  }
} // namespace

template <typename T>
class MassOperator {
  private:
    std::vector<T> _x, _y;
    std::int32_t _ncells, _ndofs;
    graph::AdjacencyList<std::int32_t> _dofmap;
    xt::xtensor<double, 4> G, _table;
    xt::xtensor<double, 2> _detJ, _phi;
    std::vector<int> _perm;
  public:
    MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim = mesh->topology().dim();
      _dofmap = V->dofmap()->list();
      _ncells = mesh->topology().index_map(tdim)->size_local();
      _ndofs = (bdegree + 1) * (bdegree + 1);
      _x.resize(_ndofs);
      _y.resize(_ndofs);

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

      // Get the determinant and inverse of the Jacobian
      auto jacobian_data = precompute_jacobian_data_quad(mesh, bdegree);
      G = std::get<0>(jacobian_data);
      _detJ = std::get<1>(jacobian_data);
      auto table_perm = tabulate_basis_and_permutation(bdegree, qdegree[bdegree]);
      _perm = std::get<0>(table_perm);
      _table = std::get<1>(table_perm);
      xt::filtration(_table, xt::isclose(_table, 0.0)) = 0;
      xt::filtration(_table, xt::isclose(_table, 1.0)) = 1;
      xt::filtration(_table, xt::isclose(_table, -1.0)) = -1;
      _phi = xt::view(_table, 0, xt::all(), xt::all(), 0);

    }

    template <typename Alloc>
    void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
      xtl::span<const T> x_array = x.array();
      xtl::span<T> y_array = y.mutable_array();
      int nq = _detJ.shape(1);
      tcb::span<const int> cell_dofs;
      for (std::int32_t cell = 0; cell < _ncells; cell++){
        cell_dofs = _dofmap.links(cell);
        int _xdof = 0;
        for (auto& idx : _perm){
          _x[_xdof] = x_array[cell_dofs[idx]];
          _xdof++;
        }

        std::fill(_y.begin(), _y.end(), 0.0);
        double* detJ_ptr = _detJ.data() + cell * nq;
        mkernel<double>(_y.data(), _x.data(), nullptr, detJ_ptr, _phi.data(), nq, _ndofs);
        int _ydof = 0;
        for (auto& idx : _perm){
          y_array[cell_dofs[idx]] += _y[_ydof];
          _ydof++;
        }
      }
    }
};

namespace {
  template <typename T>
  inline void skernel(T* A, const T* w, const std::map<std::string, double>& c, const xt::xtensor<double, 3>& G, const xt::xtensor<double, 3>& dphi, int nq, int nd){
    for (int iq = 0; iq < nq; iq++){
      double w0 = 0.0;
      double w1 = 0.0;
      for (int ic = 0; ic < nd; ic++){
        w0 += w[ic] * dphi(0, iq, ic); // dx
        w1 += w[ic] * dphi(1, iq, ic); // dy
      }
      const double fw0 = std::pow(1486.0, 2) * (G(iq, 1, 0) * w0 + G(iq, 1, 1) * w1);
      const double fw1 = std::pow(1486.0, 2) * (G(iq, 0, 0) * w0 + G(iq, 0, 1) * w1);
      for (int i = 0; i < nd; i++){
        A[i] += fw0 * dphi(0, iq, i) + fw1 * dphi(1, iq, i);
      }
    }
  }
}

template <typename T>
class StiffnessOperator {
  private:
    std::vector<T> _x, _y;
    std::int32_t _ncells, _ndofs;
    graph::AdjacencyList<std::int32_t> _dofmap;
    xt::xtensor<double, 4> G, _table;
    xt::xtensor<double, 2> _detJ;
    xt::xtensor<double, 3> _dphi;
    std::vector<int> _perm;
    std::map<std::string, double> _params;

  public:
    StiffnessOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree, std::map<std::string, double>& params) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim  = mesh->topology().dim();
      _dofmap = V->dofmap()->list();
      _ncells = mesh->topology().index_map(tdim)->size_local();
      _ndofs = (bdegree + 1)*(bdegree + 1);
      _x.resize(_ndofs);
      _y.resize(_ndofs);
      _params = params;

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

      // Get the determinant and inverse of the Jacobian
      auto jacobian_data = precompute_jacobian_data_quad(mesh, bdegree);
      G = std::get<0>(jacobian_data);
      _detJ = std::get<1>(jacobian_data);
      auto table_perm = tabulate_basis_and_permutation(bdegree, qdegree[bdegree]);
      _perm = std::get<0>(table_perm);
      _table = std::get<1>(table_perm);
      xt::filtration(_table, xt::isclose(_table, 0.0)) = 0;
      xt::filtration(_table, xt::isclose(_table, 1.0)) = 1;
      xt::filtration(_table, xt::isclose(_table, -1.0)) = -1;
      _dphi = xt::view(_table, xt::range(1, tdim+1), xt::all(), xt::all(), 0);
    }

    template <typename Alloc>
    void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
      xtl::span<const T> x_array = x.array();
      xtl::span<T> y_array = y.mutable_array();
      int nq = _detJ.shape(1);
      tcb::span<const int> cell_dofs;
      xt::xtensor<double, 3> G_cell;
      for (std::int32_t cell = 0; cell < _ncells; ++cell){
        cell_dofs = _dofmap.links(cell);
        for (int i = 0; i < _ndofs; i++){
          _x[i] = x_array[cell_dofs[i]];
        }
        std::fill(_y.begin(), _y.end(), 0.0);
        G_cell = xt::view(G, cell, xt::all(), xt::all(), xt::all());
        skernel<double> (_y.data(), _x.data(), _params, G_cell, _dphi, nq, _ndofs);
        for (int i = 0; i < _ndofs; i++){
          y_array[cell_dofs[i]] += _y[i];
        }
      }
    }
};