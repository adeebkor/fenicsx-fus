#include "precompute.hpp"
#include <cstdint>
#include <xtensor/xio.hpp>

namespace {
template <typename T>
inline void kernel(T* A, const T* w, const T* c, const double* detJ, double* phi, int nq, int nd) {
  for (int iq = 0; iq < nq; ++iq) {
    T w0 = 0.0;
    for (int ic = 0; ic < nd; ++ic)
      w0 += w[ic] * phi[iq * nd + ic] * detJ[iq];
    for (int i = 0; i < nd; ++i)
      A[i] += w0 * phi[iq * nd + i];
  }
}
} // namespace

//-------------------------------------------------------//
template <typename T>
class MassOperator {
public:
  MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int P=4) : _dofmap(0) {
    dolfinx::common::Timer t0("~setup phase");
    std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
    int tdim = mesh->topology().dim();
    _ncells = mesh->topology().index_map(tdim)->size_local();
    _ndofs = (P + 1) * (P + 1);
    _x.resize(_ndofs);
    _y.resize(_ndofs);
    _detJ = precompute_jacobian(mesh, P + 2);
    _phi = tabulate_basis(P, P + 2);
    _dofmap = V->dofmap()->list();
  }
  // Compute y = Ax with matrix-free operator
  template <typename Alloc>
  void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y) {
    xtl::span<const T> x_array = x.array();
    xtl::span<T> y_array = y.mutable_array();
    int nq = _detJ.shape(1);
    for (std::int32_t cell = 0; cell < _ncells; ++cell) {
      auto cell_dofs = _dofmap.links(cell);
      for (int i = 0; i < _ndofs; i++) {
        _x[i] = x_array[cell_dofs[i]];
      }
      std::fill(_y.begin(), _y.end(), 0);
      double* detJ_ptr = _detJ.data() + cell * nq;
      kernel<double>(_y.data(), _x.data(), nullptr, detJ_ptr, _phi.data(), nq, _ndofs);
      for (int i = 0; i < _ndofs; i++) {
        y_array[cell_dofs[i]] += _y[i];
      }
    }
  }
private:
  std::vector<T> _x;
  std::vector<T> _y;
  std::int32_t _ncells, _ndofs;
  xt::xtensor<double, 2> _detJ;
  xt::xtensor<double, 2> _phi;
  graph::AdjacencyList<std::int32_t> _dofmap;
};