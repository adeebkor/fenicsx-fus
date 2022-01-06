#include "precompute_jacobian.hpp"
#include <cstdint>
#include <xtensor/xio.hpp>

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
        xt::xtensor<double, 4> _J, _basis;
        xt::xtensor<double, 2> _detJ, _phi;
        xt::xtensor<int, 1> _pidx;
    public:
        MassOperator(std::shared_ptr<fem::FunctionSpace>& V, int P=3) : _dofmap(0) {
            std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
            int tdim = mesh->topology().dim();
            _ncells = mesh->topology().index_map(tdim)->size_local();
            _ndofs = (P + 1)*(P + 1);
            _x.resize(_ndofs);
            _y.resize(_ndofs);

            std::pair<xt::xtensor<double, 4>, xt::xtensor<double, 2>> 
            p1 = precompute_jacobian(mesh, 4);
            _J = std::get<xt::xtensor<double, 4>>(p1);
            _detJ = std::get<xt::xtensor<double, 2>>(p1);

            std::pair<xt::xtensor<int, 1>, xt::xtensor<double, 4>> 
            p2 = tabulate_basis_and_permutation();
            _pidx = std::get<xt::xtensor<int, 1>>(p2);
            _basis = std::get<xt::xtensor<double, 4>>(p2);
            _phi = xt::view(_basis, 0, xt::all(), xt::all(), 0);

            _dofmap = V->dofmap()->list();
        }

        template <typename Alloc>
        void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
            xtl::span<const T> x_array = x.array();
            xtl::span<T> y_array = y.mutable_array();
            int nq = _detJ.shape(1);
            tcb::span<const int> cell_dofs;
            for (std::int32_t cell = 0; cell < _ncells; ++cell){
                cell_dofs = _dofmap.links(cell);
                int _xidx = 0;
                for (auto& idx : _pidx){
                    _x[_xidx] = x_array[cell_dofs[idx]];
                    _xidx++;
                }

                std::fill(_y.begin(), _y.end(), 0.0);
                double* detJ_ptr = _detJ.data() + cell * nq;
                mkernel<double>(_y.data(), _x.data(), nullptr, detJ_ptr, _phi.data(), nq, _ndofs);
                int _yidx = 0;
                for (auto& idx : _pidx){
                    y_array[cell_dofs[idx]] += _y[_yidx];
                    _yidx++;
                }
            }
        }
};