#include "form.h"
#include "precompute_jacobian.hpp"
#include "stiffness_operator.hpp"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

int main(int argc, char* argv[]){
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
	std::cout.precision(15);

	// Create mesh and function space
	std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
	  MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 4},
	  mesh::CellType::quadrilateral, mesh::GhostMode::none));

	std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(
	  fem::create_functionspace(functionspace_form_form_a, "u", mesh));

	// Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();

	// Create stiffness operator
	std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);
	xtl::span<double> _u = u->x()->mutable_array();
	std::fill(_u.begin(), _u.end(), 1.0);

	std::shared_ptr<StiffnessOperator<double>> stiffness_operator = std::make_shared<StiffnessOperator<double>>(V);
	std::shared_ptr<la::Vector<double>> s = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _s = s->mutable_array();
	std::fill(_s.begin(), _s.end(), 0.0);
	stiffness_operator->operator()(*u->x(), *s);

	for (int i = 0; i < 10; ++i){
      std::cout << s->mutable_array()[i] << std::endl;
    }
	std::getchar();
  }
}