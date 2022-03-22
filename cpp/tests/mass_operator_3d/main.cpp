#include "form.h"
#include "operators_3d.hpp"
#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]){
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    std::cout.precision(10);
    std::cout << std::fixed;

	// Create mesh and function space
	std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
	  mesh::create_box(MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {4, 4, 4},
	  mesh::CellType::hexahedron, mesh::GhostMode::none));

	std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(
	  fem::create_functionspace(functionspace_form_form_a, "u", mesh));

	// Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();

	// Create input function
	std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);
	u->interpolate([](auto& x) { return xt::row(x, 0)*xt::row(x, 1); });

	// Create mass operator
	MassOperator<double> mass_operator(V, 3);
	la::Vector<double> m(index_map, bs);
	mass_operator(*u->x(), m);
	m.scatter_rev(common::IndexMap::Mode::add);

	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
		fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {}, {}));
	la::Vector<double> m_ref(index_map, bs);
	fem::assemble_vector(m_ref.mutable_array(), *a);
	m_ref.scatter_rev(common::IndexMap::Mode::add);

    auto _m = m.array();
    auto _m_ref = m_ref.array();
    for (std::size_t i = 0; i < 10; ++i) {
      std::cout << _m[i] << "\t " << _m_ref[i] << "\t " << std::abs(_m[i] - _m_ref[i]) << std::endl;
	}
  }
  common::subsystem::finalize_mpi();
  return 0;
}