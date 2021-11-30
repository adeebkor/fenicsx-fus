#include "precompute_jacobian.hpp"
#include <cstdint>
#include <xtensor/xio.hpp>
#include <xtensor/xindex_view.hpp>

namespace {
    template <typename T>
    inline void skernel(T* A, const T* w, const T* c, const double* detJ, const xt::xtensor<double, 3>& J, const xt::xtensor<double, 3>& phi, int nq, int nd){
        const double weights_fdb[49] = { 0.0005668934240362788, 0.003295548182875772, 0.005139825966784068, 0.005804988662131499, 0.005139825966784057, 0.00329554818287577, 0.0005668934240362745, 0.003295548182875772, 0.01915816512445691, 0.02987959183673464, 0.03374641339264793, 0.02987959183673457, 0.0191581651244569, 0.003295548182875747, 0.005139825966784068, 0.02987959183673464, 0.04660101854901241, 0.05263181789986891, 0.0466010185490123, 0.02987959183673462, 0.005139825966784029, 0.005804988662131499, 0.03374641339264793, 0.05263181789986891, 0.05944308390022661, 0.05263181789986879, 0.03374641339264791, 0.005804988662131456, 0.005139825966784057, 0.02987959183673457, 0.0466010185490123, 0.05263181789986879, 0.04660101854901219, 0.02987959183673455, 0.005139825966784018, 0.00329554818287577, 0.0191581651244569, 0.02987959183673462, 0.03374641339264791, 0.02987959183673455, 0.01915816512445689, 0.003295548182875745, 0.0005668934240362745, 0.003295548182875747, 0.005139825966784029, 0.005804988662131456, 0.005139825966784018, 0.003295548182875745, 0.0005668934240362702 };
        for (int iq = 0; iq < nq; ++iq){
            double w0_d01 = 0.0;
            double w0_d10 = 0.0;
            for (int ic = 0; ic < nd; ++ic){
                w0_d01 += w[ic] * phi(1, iq, ic);
                w0_d10 += w[ic] * phi(0, iq, ic);
            }
            double sv_fdb[24];
            sv_fdb[0] = J(iq, 0, 0) * J(iq, 1, 1);
            sv_fdb[1] = J(iq, 0, 1) * J(iq, 1, 0);
            sv_fdb[2] = sv_fdb[0] + -1 * sv_fdb[1];
            sv_fdb[3] = J(iq, 0, 0) / sv_fdb[2];
            sv_fdb[4] = (-1 * J(iq, 0, 1)) / sv_fdb[2];
            sv_fdb[5] = w0_d01 * sv_fdb[3];
            sv_fdb[6] = w0_d10 * sv_fdb[4];
            sv_fdb[7] = sv_fdb[5] + sv_fdb[6];
            sv_fdb[8] = sv_fdb[7] * sv_fdb[3];
            sv_fdb[9] = sv_fdb[7] * sv_fdb[4];
            sv_fdb[10] = J(iq, 1, 1) / sv_fdb[2];
            sv_fdb[11] = (-1 * J(iq, 1, 0)) / sv_fdb[2];
            sv_fdb[12] = w0_d10 * sv_fdb[10];
            sv_fdb[13] = w0_d01 * sv_fdb[11];
            sv_fdb[14] = sv_fdb[12] + sv_fdb[13];
            sv_fdb[15] = sv_fdb[14] * sv_fdb[11];
            sv_fdb[16] = sv_fdb[14] * sv_fdb[10];
            sv_fdb[17] = sv_fdb[8] + sv_fdb[15];
            sv_fdb[18] = sv_fdb[16] + sv_fdb[9];
            sv_fdb[19] = (-1 * sv_fdb[17]) * pow(c[0], 2);
            sv_fdb[20] = (-1 * sv_fdb[18]) * pow(c[0], 2);
            sv_fdb[21] = fabs(sv_fdb[2]);
            sv_fdb[22] = sv_fdb[19] * sv_fdb[21];
            sv_fdb[23] = sv_fdb[20] * sv_fdb[21];
            const double fw0 = sv_fdb[23] * weights_fdb[iq];
            const double fw1 = sv_fdb[22] * weights_fdb[iq];
            for (int i = 0; i < nd; ++i)
                A[i] += fw0 * phi(0, iq, i) + fw1 * phi(1, iq, i);
            }
        }
}

template <typename T>
class StiffnessOperator {
    private:
        std::vector<T> _x, _y;
        std::int32_t _ncells, _ndofs;
        graph::AdjacencyList<std::int32_t> _dofmap;
        xt::xtensor<double, 4> _J, _basis;
        xt::xtensor<double, 2> _detJ;
        xt::xtensor<double, 3> _phi;
        xt::xtensor<int, 1> _pidx;
    public:
        StiffnessOperator(std::shared_ptr<fem::FunctionSpace>& V, int P=6) : _dofmap(0) {
            std::shared_ptr<const mesh::Mesh> mesh = V->mesh();
            int tdim = mesh->topology().dim();
            _ncells = mesh->topology().index_map(tdim)->size_local();
            _ndofs = (P + 1)*(P + 1);
            _x.resize(_ndofs);
            _y.resize(_ndofs);

            std::pair<xt::xtensor<double, 4>, xt::xtensor<double, 2>> 
            p1 = precompute_jacobian(mesh, 10);
            _J = std::get<xt::xtensor<double, 4>>(p1);
            _detJ = std::get<xt::xtensor<double, 2>>(p1);

            std::pair<xt::xtensor<int, 1>, xt::xtensor<double, 4>> 
            p2 = tabulate_basis_and_permutation();
            _pidx = std::get<xt::xtensor<int, 1>>(p2);
            _basis = std::get<xt::xtensor<double, 4>>(p2);
            _phi = xt::view(_basis, xt::range(1, 3), xt::all(), xt::all(), 0);

            auto idx0 = xt::argwhere(xt::isclose(_phi, 0.0));
            auto idx1 = xt::argwhere(xt::isclose(_phi, 1.0));
            auto idx2 = xt::argwhere(xt::isclose(_phi, -1.0));
            xt::index_view(_phi, idx0) = 0.0;
            xt::index_view(_phi, idx1) = 1.0;
            xt::index_view(_phi, idx2) = -1.0;

            // auto Jidx0 = xt::argwhere(xt::isclose(_J, 0.0));
            // auto Jidx1 = xt::argwhere(xt::isclose(_J, 1.0));
            // auto Jidx2 = xt::argwhere(xt::isclose(_J, -1.0));
            // xt::index_view(_J, Jidx0) = 0.0;
            // xt::index_view(_J, Jidx1) = 1.0;
            // xt::index_view(_J, Jidx2) = -1.0;

            // auto detJidx0 = xt::argwhere(xt::isclose(_detJ, 0.0));
            // auto detJidx1 = xt::argwhere(xt::isclose(_detJ, 1.0));
            // auto detJidx2 = xt::argwhere(xt::isclose(_detJ, -1.0));
            // xt::index_view(_detJ, detJidx0) = 0.0;
            // xt::index_view(_detJ, detJidx1) = 1.0;
            // xt::index_view(_detJ, detJidx2) = -1.0;

            // std::cout << xt::view(_detJ, 0, xt::all()) << std::endl;

            _dofmap = V->dofmap()->list();
        }

        template <typename Alloc>
        void operator()(const la::Vector<T, Alloc>& x, la::Vector<T, Alloc>& y){
            xtl::span<const T> x_array = x.array();
            xtl::span<T> y_array = y.mutable_array();
            int nq = _detJ.shape(1);
            tcb::span<const int> cell_dofs;
            xt::xtensor<double, 3> J;
            double* c = new double[1];
            c[0] = 1486.0;
            for (std::int32_t cell = 0; cell < _ncells; ++cell){
                cell_dofs = _dofmap.links(cell);
                for (int i = 0; i < _ndofs; i++){
                    _x[i] = x_array[cell_dofs[i]];
                }
                std::fill(_y.begin(), _y.end(), 0.0);
                double* detJ_ptr = _detJ.data() + cell * nq;
                J = xt::view(_J, cell, xt::all(), xt::all(), xt::all());
                skernel<double> (_y.data(), _x.data(), c, detJ_ptr, J, _phi, nq, _ndofs);
                for (int i = 0; i < _ndofs; i++){
                    y_array[cell_dofs[i]] += _y[i];
                }
            }
        }
};