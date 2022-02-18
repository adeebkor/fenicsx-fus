#include "form.h"

#include <cmath>
#include <fstream>

#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/geometry/utils.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[]){
    common::subsystem::init_logging(argc, argv);
    common::subsystem::init_mpi(argc, argv);
    {
        std::cout.precision(10);
        std::cout << std::fixed;

        // Create mesh and function space
        std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
            mesh::create_rectangle(MPI_COMM_WORLD, {{{-1.0, -1.0}, {1.0, 1.0}}},
            {32, 32}, mesh::CellType::quadrilateral, mesh::GhostMode::none));

        std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(functionspace_form_form_a, "u", mesh));

        // Create input function
        std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);
        u->interpolate([](auto& x) { return xt::sin(2.0 * M_PI * xt::row(x, 0)) * xt::cos(2.0 * M_PI * xt::row(x, 1)); });
        u->x()->scatter_fwd();

        // Output to XDMF
        // io::XDMFFile outfile(mesh->comm(), "u.xdmf", "w");
        // outfile.write_mesh(*mesh);
        // outfile.write_function(*u, 0.0);

        // Evaluate on a line
        int N = 100;
        double tol = 1e-6;
        xt::xarray<double> xp = xt::linspace<double>(-1.0+tol, 1.0-tol, N);
        xp.reshape({1, N});
        auto yzp = xt::zeros<double>({2, N});
        auto points = xt::vstack(xt::xtuple(xp, yzp));
        auto pointsT = xt::transpose(points);

        auto bb_tree = geometry::BoundingBoxTree(*mesh, mesh->topology().dim());
        auto cell_candidates = compute_collisions(bb_tree, pointsT);

        // Choose one of the cells that contains the point
        auto colliding_cells = geometry::compute_colliding_cells(*mesh, cell_candidates, pointsT);

        std::vector<int> cells;
        xt::xtensor<double, 2>::shape_type sh0 = {1, 3};
        auto points_on_proc = xt::empty<double>(sh0);

        std::cout << colliding_cells.str() << std::endl;
        int local_size = mesh->topology().index_map(2)->size_local();
        std::cout << local_size << std::endl;


        for (int i = 0; i < N; i++){
            auto link = colliding_cells.links(i);
            if (link.size() > 0){
                auto p = xt::view(pointsT, i, xt::newaxis(), xt::all());
                points_on_proc = xt::vstack(xt::xtuple(points_on_proc, p));
                // for (auto c : link){
                //     cells.push_back(c);
                // }
                cells.push_back(link[0]);
            }
        }

        points_on_proc = xt::view(points_on_proc, xt::drop(0), xt::all());
        int lsize = points_on_proc.shape(0);
        xt::xtensor<double, 2> u_eval({lsize, 1});

        u->eval(points_on_proc, cells, u_eval);

        double * uval = u_eval.data();
        double * pval = points_on_proc.data();
        
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        auto shape = points_on_proc.shape();
        std::cout << rank << " " << xt::adapt(shape) << std::endl;

        for (int i = 0; i < size; i++){
            if (rank == i){
                std::ofstream MyFile("filename.txt", std::ios_base::app);
                for (int i = 0; i < lsize; i++){
                    MyFile << *(pval + 3*i) << "," << *(uval + i) << std::endl;
                }
                MyFile.close();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // double* pval = points_on_proc.data();
        // double* uval = u_eval.data();

        // for (int i = 0; i < points_on_proc.shape(0); i++){
        //     std::cout << *(pval + 3*i) << std::endl;
        // }

        // double *rbuf0;
        // double *rbuf1;

        // if (rank == 0){
        //     rbuf0 = (double *)malloc(N*sizeof(double));
        //     rbuf1 = (double *)malloc(N*sizeof(double));
        // }

        // MPI_Gather(pval, lsize, MPI_DOUBLE, rbuf0, lsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Gather(uval, lsize, MPI_DOUBLE, rbuf1, lsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // if (rank == 0){
        //     std::ofstream MyFile("filename.txt");
        //     for (int i = 0; i < N; i++){
        //         MyFile << *(rbuf0 + 3*i) << "," << *(rbuf1 + i) << std::endl;
        //     }
        //     MyFile.close();
        // }
    }
}