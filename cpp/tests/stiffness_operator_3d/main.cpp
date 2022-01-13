#include "form.h"
#include "precomputation.hpp"
#include "operators_3d.hpp"
#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]){
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
	std::cout.precision(15);

	// Create mesh and function space
	std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
	  MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {4, 4, 4},
	  mesh::CellType::hexahedron, mesh::GhostMode::none));

	std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(
	  fem::create_functionspace(functionspace_form_form_a, "u", mesh));

	// Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();

	// Create stiffness operator
	std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);

	u->interpolate(
        [](auto& x) -> xt::xarray<PetscScalar>
        {
          auto dx = xt::square(xt::row(x, 0) - 0.5)
                    + xt::square(xt::row(x, 1) - 0.5);
          return 10e10 * xt::exp(-(dx) / 0.02);
        });

	std::map<std::string, double> params;
	params["c"] = 1486.0;
	std::shared_ptr<StiffnessOperator<double>> stiffness_operator = std::make_shared<StiffnessOperator<double>>(V, 3, params);
	std::shared_ptr<la::Vector<double>> s = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _s = s->mutable_array();
	std::fill(_s.begin(), _s.end(), 0.0);
	stiffness_operator->operator()(*u->x(), *s);

	double speedOfSound = params["c"];
	std::shared_ptr<fem::Constant<double>> c0 = std::make_shared<fem::Constant<double>>(speedOfSound);
	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {{"c0", c0}}, {}));
	std::shared_ptr<la::Vector<double>> s_ref = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _s_ref = s_ref->mutable_array();
	std::fill(_s_ref.begin(), _s_ref.end(), 0.0);
	fem::assemble_vector(_s_ref, *a);
	s_ref->scatter_rev(common::IndexMap::Mode::add);

	for (int i = 0; i < 10; ++i){
		std::cout << s->mutable_array()[i] 
				  << " " << s_ref->mutable_array()[i] 
		          << " " << (s->mutable_array()[i] - s_ref->mutable_array()[i]) / s_ref->mutable_array()[i]
				  << std::endl;
	}
  }

  return 0;
}
