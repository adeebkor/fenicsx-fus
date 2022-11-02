#include "form.h"

#include <cmath>
#include <fstream>

#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/geometry/utils.h>

using namespace dolfinx;

int main(int argc, char* argv[]){
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    {
        std::cout.precision(10);
        std::cout << std::fixed;

        // Create mesh and function space
        auto part = mesh::create_cell_partitioner(
            mesh::GhostMode::shared_facet);
        std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
            mesh::create_rectangle(MPI_COMM_WORLD, {{{-1.0, -1.0}, {1.0, 1.0}}},
            {32, 32}, mesh::CellType::quadrilateral, part));

        std::shared_ptr<fem::FunctionSpace> V 
            = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
                functionspace_form_form_a, "u", mesh));

        // Create input function
        std::shared_ptr<fem::Function<double>> u 
            = std::make_shared<fem::Function<double>>(V);
        u->interpolate(
            [](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
            {
                std::vector<double> u;
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
        std::vector<double> point_coordinates(3*num_points);
        for (std::size_t i = 0; i < num_points; i++) {
            point_coordinates[3*i] = -1.0 + 2.0*i/(num_points-1);
            point_coordinates[3*i + 1] = 0.0;
            point_coordinates[3*i + 2] = 0.0;
        }

        // Compute evaluation parameters
        auto bb_tree = geometry::BoundingBoxTree(*mesh, mesh->topology().dim());
        auto cell_candidates = compute_collisions(bb_tree, point_coordinates);
        auto colliding_cells = geometry::compute_colliding_cells(
            *mesh, cell_candidates, point_coordinates);

        std::vector<std::int32_t> cells;
        std::vector<double> points_on_proc;

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
        std::vector<double> u_eval(num_points_local);
        u->eval(points_on_proc, {num_points_local, 3}, cells, u_eval,
                {num_points_local, 1});

        // Print to file
        double * u_value = u_eval.data();
        double * p_value = points_on_proc.data();

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