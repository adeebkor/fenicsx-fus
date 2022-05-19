#include "form.h"
#include "mass_3d.hpp"
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
    const std::size_t N = 64;
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

    // ------------------------------------------------------------------------
    // Create dolfinx mass vector
	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
		fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {}, {}));
	la::Vector<double> m0(index_map, bs);

    common::Timer t_dolfinx_assembly("~ dolfinx vector assembly");
    t_dolfinx_assembly.start();
	fem::assemble_vector(m0.mutable_array(), *a);
    t_dolfinx_assembly.stop();

	m0.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Create precompute mass operator
    common::Timer t_precompute_construct("~ construct precompute");
    t_precompute_construct.start();
    Mass<double, 4, 5, 6> mass_pc(V);
    t_precompute_construct.stop();
    la::Vector<double> m1(index_map, bs);

    common::Timer t_precompute_assembly("~ precompute vector assembly");
    t_precompute_assembly.start();
    mass_pc(*u->x(), m1);
    t_precompute_assembly.stop();

    m1.scatter_rev(common::IndexMap::mode::add);

    // ------------------------------------------------------------------------
	// Create mass operator
    common::Timer t_spectral_construct("~ construct spectral");
    t_spectral_construct.start();
	SpectralMass<double, 4, 5> mass(V);
    t_spectral_construct.stop();
	la::Vector<double> m2(index_map, bs);

    common::Timer t_spectral_assembly("~ spectral vector assembly");
    t_spectral_assembly.start();
	mass(*u->x(), m2);
    t_spectral_assembly.stop();

	m2.scatter_rev(common::IndexMap::Mode::add);

    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);

    // Check equality
    std::size_t local_size = m0.mutable_array().size();
    auto m_dolfinx = xt::adapt(m0.mutable_array().data(), {local_size});
    auto m_precompute = xt::adapt(m1.mutable_array().data(), {local_size});
    auto m_spectral = xt::adapt(m2.mutable_array().data(), {local_size});

    assert(xt::allclose(m_dolfinx, m_precompute));
    assert(xt::allclose(m_dolfinx, m_spectral));
  }
  common::subsystem::finalize_mpi();
  return 0;
}