// ========================================================
// Perform interpolation from a 3D geometry to a 2D surface
// ========================================================

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>
#include <dolfinx.h>

using T = float;

int main(int argc, char* argv[])
{
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    {
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

        const std::size_t N = 16;
        auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);

        // Create 2D mesh
        auto mesh_2d = std::make_shared<mesh::Mesh<T>>(
            mesh::create_rectangle<T>(
                MPI_COMM_WORLD,
                {{{0., 0.}, {1., 1.}}},
                {N, N},
                mesh::CellType::quadrilateral,
                part));

        // Create 3D mesh
        auto mesh_3d = std::make_shared<mesh::Mesh<T>>(
            mesh::create_box<T>(
                MPI_COMM_WORLD,
                {{{-1.0, -1.0, -1.0}, {2.0, 2.0, 2.0}}},
                {N, N, N},
                mesh::CellType::hexahedron,
                part));

        // Create function space for the 2D mesh
        constexpr int degreeOfBasis_2d = 1;
        basix::FiniteElement element_2d = basix::create_element<T>(
            basix::element::family::P, basix::cell::type::quadrilateral, degreeOfBasis_2d,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false
        );

        std::shared_ptr<fem::FunctionSpace<T>> V_2d 
            = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh_2d, element_2d));

        // Create function space for the 3D mesh
        constexpr int degreeOfBasis_3d = 3;
        basix::FiniteElement element_3d = basix::create_element<T>(
            basix::element::family::P, basix::cell::type::hexahedron, degreeOfBasis_3d,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false
        );

        std::shared_ptr<fem::FunctionSpace<T>> V_3d 
            = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh_3d, element_3d));

        // Create 2D function
        std::shared_ptr<fem::Function<T>> u_2d
            = std::make_shared<fem::Function<T>>(V_2d);

        // Create 3D function
        std::shared_ptr<fem::Function<T>> u_3d
            = std::make_shared<fem::Function<T>>(V_3d);

        // Interpolate 3D function
        u_3d->interpolate(
            [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
            {
                std::vector<T> u;
                for (std::size_t p = 0; p < x.extent(1); ++p)
                {
                    u.push_back(
                        0.83 * std::sin(2.0 * M_PI * (
                        1.3*x(0, p) + 0.7*x(1, p) + 1.9*x(2, p) + 1.25)) +
                        0.52 * std::sin(2.0 * M_PI * (
                        2.4*x(0, p) + 1.8*x(1, p) + 0.5*x(2, p) + 4.71)));
                }
            
            return {u, {u.size()}};
            }
        );
        u_3d->x()->scatter_fwd();

        io::VTXWriter<T> vtx(MPI_COMM_WORLD, "u_3d.bp", {u_3d}, "bp4");
        vtx.write(0);

        // Interpolate to 2D mesh (Same as sampling on a mesh)
        auto cell_map
            = mesh_2d->topology()->index_map(mesh_2d->topology()->dim());
        assert(cell_map);
        std::vector<std::int32_t> cells(
        cell_map->size_local() + cell_map->num_ghosts(), 0);
        std::iota(cells.begin(), cells.end(), 0);
        geometry::PointOwnershipData<T> interpolation_data
            = fem::create_interpolation_data<T>(
                u_2d->function_space()->mesh()->geometry(),
                *u_2d->function_space()->element(),
                *u_3d->function_space()->mesh(), std::span(cells), 1e-8);
        u_2d->interpolate(*u_3d, cells, interpolation_data);

        io::VTXWriter<T> vtx2d(MPI_COMM_WORLD, "u_2d.bp", {u_2d}, "bp4");
        vtx2d.write(0);
    }
    PetscFinalize();
    
    return 0;
}