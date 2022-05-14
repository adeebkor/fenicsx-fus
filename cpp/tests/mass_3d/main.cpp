#include "form.h"
#include "spectral_mass_3d.hpp"

#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]){
  
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  {
    std::cout.precision(10);
    std::cout << std::fixed;

	// Create mesh and function space
    const std::size_t N = 32;
	std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
	  mesh::create_box(MPI_COMM_WORLD, 
      {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, 
      {N, N, N},
	  mesh::CellType::hexahedron,
      mesh::GhostMode::none));

    std::shared_ptr<fem::FunctionSpace> V
      = std::make_shared<fem::FunctionSpace>(
          fem::create_functionspace(functionspace_form_form_a, "u", mesh));

	// Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();
    std::size_t ndofs = index_map->size_global();

    if (rank == 0) {
      std::cout << "Number of degress of freedom: " << ndofs << std::endl;
    }

	// Create input function
	std::shared_ptr<fem::Function<double>> u 
      = std::make_shared<fem::Function<double>>(V);
	u->interpolate([](auto& x) { return xt::row(x, 0)*xt::row(x, 1); });

    // Create dolfinx stiffness vector
	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
		fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {}, {}));
	la::Vector<double> m0(index_map, bs);
	fem::assemble_vector(m0.mutable_array(), *a);
	m0.scatter_rev(common::IndexMap::Mode::add);

	// Create mass operator
	SpectralMass<double, 3, 4> mass(V);
	la::Vector<double> m1(index_map, bs);
	mass(*u->x(), m1);
	m1.scatter_rev(common::IndexMap::Mode::add);

    std::size_t local_size = m0.mutable_array().size();
    auto m_dolfinx = xt::adapt(m0.mutable_array().data(), {local_size});
    auto m_spectral = xt::adapt(m1.mutable_array().data(), {local_size});

    assert(xt::allclose(m_dolfinx, m_spectral));
  }
  common::subsystem::finalize_mpi();
  return 0;
}