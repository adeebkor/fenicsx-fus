// ===========================
// Computation of the Jacobian
// ===========================


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>
#include <dolfinx.h>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;

int main(int argc, char* argv[])
{
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    {
        int P = 4;  // Polynomial degree
        int Q = 9;  // Quadrature points

        // Map from quadrature points to basix quadrature degree
        std::map<int, int> qdegree;
        qdegree[2] = 3;
        qdegree[3] = 4;
        qdegree[4] = 5;
        qdegree[5] = 6;
        qdegree[6] = 8;
        qdegree[7] = 10;
        qdegree[8] = 12;
        qdegree[9] = 14;
        qdegree[10] = 16;

        // Create mesh
        const std::size_t N = 4;
        auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
        auto mesh = std::make_shared<mesh::Mesh>(
          mesh::create_rectangle(
            MPI_COMM_WORLD,
            {{{0.0, 0.0}, {1.0, 1.0}}},
            {N, N},
            mesh::CellType::quadrilateral,
            part));

        // Tabulate quadrature points and weights
        auto [points, weights]
          = basix::quadrature::make_quadrature(
              basix::quadrature::type::gll,
              basix::cell::type::quadrilateral,
              qdegree[Q]);

        // // Create mesh
        // const std::size_t N = 4;
        // auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
        // auto mesh = std::make_shared<mesh::Mesh>(
        //   mesh::create_box(
        //     MPI_COMM_WORLD,
        //     {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
        //     {N, N, N},
        //     mesh::CellType::hexahedron,
        //     part));

        // // Tabulate quadrature points and weights
        // auto [points, weights]
        //   = basix::quadrature::make_quadrature(
        //       basix::quadrature::type::gll,
        //       basix::cell::type::hexahedron,
        //       qdegree[Q]);

        std::size_t nq = weights.size();

        std::cout << "Number of quadrature points: " << nq << "\n";

        // Get geometry data
        const fem::CoordinateElement& cmap = mesh->geometry().cmap();
        const graph::AdjacencyList<std::int32_t>& x_dofmap 
            = mesh->geometry().dofmap();
        const std::size_t num_dofs_g = cmap.dim();
        std::span<const double> x_g = mesh->geometry().x();

        // Get dimensions
        const std::size_t tdim = mesh->topology().dim();
        const std::size_t gdim = mesh->geometry().dim();
        const std::size_t nc = mesh->topology().index_map(tdim)->size_local();

        // Tabulate basis functions at quadrature points
        std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
        std::vector<double> phi_b(
            std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
        cmdspan4_t phi(phi_b.data(), phi_shape);
        cmap.tabulate(1, points, {nq, gdim}, phi_b);

        // Create working arrays
        std::vector<double> coord_dofs_b(num_dofs_g * gdim);
        mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

        // Compute Jacobian and its determinant
        std::vector<double> J_b(nc * nq * tdim * gdim);
        mdspan4_t J(J_b.data(), nc, nq, tdim, gdim);
        std::vector<double> detJ_b(nc * nq);
        mdspan2_t detJ(detJ_b.data(), nc, nq);
        std::vector<double> det_scratch(2 * tdim * gdim);

        std::fill(J_b.begin(), J_b.end(), 0.0);
        for (std::int32_t c = 0; c < nc; ++c)
        {
            // Get cell geometry (coordinates dofs)
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                for (std::size_t j = 0; j < gdim; ++j)
                    coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
            }

            // Compute Jacobians and determinant for current cell
            for (std::size_t q = 0; q < nq; ++q)
            {
                auto dphi = stdex::submdspan(phi, std::pair(1, tdim+1), q, 
                                             stdex::full_extent, 0);
                auto _J = stdex::submdspan(J, c, q, stdex::full_extent,
                                           stdex::full_extent);
                cmap.compute_jacobian(dphi, coord_dofs, _J);
                detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);
            }
        }

        for (std::size_t i = 0; i < detJ_b.size(); ++i) {
            std::cout << detJ_b[i] << "\n";
        }
    }
}