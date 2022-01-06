#pragma once

#include "precomputation.hpp"
#include <cstdint>
#include <map>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
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
    xt::xtensor<double, 4> _J_inv, _table;
    xt::xtensor<double, 2> _detJ, _phi;
    std::vector<int> _perm;
  public:
    MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim = mesh->topology().dim();
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
      auto jacobian_data = precompute_jacobian_data(mesh, bdegree);
      _J_inv = std::get<0>(jacobian_data);
      _detJ = std::get<1>(jacobian_data);
      auto table_perm = tabulate_basis_and_permutation(bdegree, qdegree[bdegree]);
      _perm = std::get<0>(table_perm);
      _table = std::get<1>(table_perm);
      _phi = xt::view(_table, 0, xt::all(), xt::all(), 0);

      _dofmap = V->dofmap()->list();
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

template <typename T>
class StiffnessOperator {
  private:
    std::vector<T> _x, _y;
    std::int32_t _ncells, _ndofs;
    graph::AdjacencyList<std::int32_t> _dofmap;
    xt::xtensor<double, 4> _J_inv, _table;
    xt::xtensor<double, 2> _detJ;
    xt::xtensor<double, 3> _dphi;
    std::vector<int> _perm;

  public:
    StiffnessOperator(std::shared_ptr<fem::FunctionSpace>& V, int bdegree) : _dofmap(0) {
      std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
      int tdim  = mesh->topology().dim();
      _ncells = mesh->topology().index_map(tdim)->size_local();
      _ndofs = (bdegree + 1)*(bdegree + 1);
      _x.resize(_ndofs);
      _y.resize(_ndofs);

      
    }
}