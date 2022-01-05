#include "jacobian_precomputation.hpp"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using T = PetscScalar;

int main(int argc, char* argv[]){
	common::subsystem::init_logging(argc, argv);
	common::subsystem::init_mpi(argc, argv);
	{
		std::cout.precision(15); // Set print precision

		// Create mesh and function space
		auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
			MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {2, 2},
			mesh::CellType::quadrilateral, mesh::GhostMode::none));
			
		auto [p1, p2] = precompute_jacobian_data(mesh, 1);
	}
}