#include "form.h"

#include <cmath>
#include <fstream>

#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/geometry/utils.h>

using namespace dolfinx;
using T = double;

int main(int argc, char* argv[]){
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    {
        std::cout.precision(10);
        std::cout << std::fixed;

        // Create mesh and function space
        auto part = mesh::create_cell_partitioner(
            mesh::GhostMode::shared_facet);
        std::shared_ptr<mesh::Mesh<T>> mesh = std::make_shared<mesh::Mesh<T>>(
            mesh::create_rectangle(MPI_COMM_WORLD, {{{-1.0, -1.0}, {1.0, 1.0}}},
            {32, 32}, mesh::CellType::quadrilateral, part));

        const int degreeOfBasis = 4;
        basix::FiniteElement element = basix::create_element<T>(
          basix::element::family::P, basix::cell::type::quadrilateral, degreeOfBasis,
          basix::element::lagrange_variant::gll_warped,
          basix::element::dpc_variant::unset, false
        );
        
        std::shared_ptr<fem::FunctionSpace<T>> V 
            = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(mesh, element));

        // Create input function
        std::shared_ptr<fem::Function<T>> u 
            = std::make_shared<fem::Function<T>>(V);
        u->interpolate(
            [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
            {
                std::vector<T> u;
                for (std::size_t p = 0; p < x.extent(1); ++p)
                {
                    u.push_back(std::sin(2.0 * M_PI * x(0, p)) * 
                                std::cos(2.0 * M_PI * x(1, p)));
                }
            
            return {u, {u.size()}};
            }
        );
        u->x()->scatter_fwd();

        // -------------------------------------------------------------------
        // Evaluate on a line
        const std::size_t num_points = 100;

        // Create evaluation point coordinates
        std::vector<T> point_coordinates(3*num_points);
        for (std::size_t i = 0; i < num_points; i++) {
            point_coordinates[3*i] = -1.0 + 2.0*i/(num_points-1);
            point_coordinates[3*i + 1] = 0.0;
            point_coordinates[3*i + 2] = 0.0;
        }

        const int tdim = mesh->topology()->dim();
        mesh->topology()->create_entities(tdim);
        auto map = mesh->topology()->index_map(tdim);
        const std::int32_t num_entities = map->size_local() + map->num_ghosts();
        std::vector<std::int32_t> entities(num_entities);
        std::iota(entities.begin(), entities.end(), 0);

        // Compute evaluation parameters
        auto bb_tree = geometry::BoundingBoxTree(*mesh, tdim, entities);
        auto cell_candidates = compute_collisions<T>(bb_tree, point_coordinates);
        auto colliding_cells = geometry::compute_colliding_cells<T>(
            *mesh, cell_candidates, point_coordinates);

        std::vector<std::int32_t> cells;
        std::vector<T> points_on_proc;

        for (std::size_t i = 0; i < num_points; ++i) {
            auto link = colliding_cells.links(i);
            if (link.size() > 0) {
                points_on_proc.push_back(point_coordinates[3*i]);
                points_on_proc.push_back(point_coordinates[3*i + 1]);
                points_on_proc.push_back(point_coordinates[3*i + 2]);
                cells.push_back(link[0]);
            }
        }

        std::size_t num_points_local = points_on_proc.size() / 3;

        // Evaluate function
        std::vector<T> u_eval(num_points_local);
        u->eval(points_on_proc, {num_points_local, 3}, cells, u_eval,
                {num_points_local, 1});

        // Print to file
        T * u_value = u_eval.data();
        T * p_value = points_on_proc.data();

        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        for (int i = 0; i < mpi_size; ++i) {
            if (mpi_rank == i) {
                std::ofstream MyFile("line_data.txt", std::ios_base::app);
                for (std::size_t i = 0; i < num_points_local; ++i) {
                    MyFile << *(p_value + 3*i) << "," 
                           << *(u_value + i) << std::endl;
                }
                MyFile.close();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    return 0;
}