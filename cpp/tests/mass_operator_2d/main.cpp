#include "form.h"
#include "precomputation.hpp"
#include "operators_2d.hpp"
#include <cmath>
#include <dolfinx.h>

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

	// Create mass operator
	std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);
	xtl::span<double> _u = u->x()->mutable_array();
	std::fill(_u.begin(), _u.end(), 1.0);

	std::shared_ptr<MassOperator<double>> mass_operator = std::make_shared<MassOperator<double>>(V, 3);
	std::shared_ptr<la::Vector<double>> m = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _m = m->mutable_array();
	std::fill(_m.begin(), _m.end(), 0.0);
	mass_operator->operator()(*u->x(), *m);

	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {}, {}));
	std::shared_ptr<la::Vector<double>> m_ref = std::make_shared<la::Vector<double>>(index_map, bs);
	tcb::span<double> _m_ref = m_ref->mutable_array();
	std::fill(_m_ref.begin(), _m_ref.end(), 0.0);
	fem::assemble_vector(_m_ref, *a);
	m_ref->scatter_rev(common::IndexMap::Mode::add);

	for (int i = 0; i < 10; ++i){
		std::cout << m->mutable_array()[i] 
				  << " " << m_ref->mutable_array()[i] 
		          << " " << (m->mutable_array()[i] - m_ref->mutable_array()[i]) / m_ref->mutable_array()[i]
				  << std::endl;
	}
  }

  return 0;
}