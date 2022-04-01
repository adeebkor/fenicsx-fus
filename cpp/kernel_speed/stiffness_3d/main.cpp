#include "form.h"
#include "stiffness_3d.hpp"
#include "spectral_stiffness_3d.hpp"

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
        mesh::create_box(
            MPI_COMM_WORLD,
            {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
            {N, N, N},
            mesh::CellType::hexahedron,
            mesh::GhostMode::none));

	std::shared_ptr<fem::FunctionSpace> V 
      = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
          functionspace_form_form_a, "u", mesh));

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
    u->interpolate([](auto& x) { return xt::sin(xt::row(x, 0)); });

    // ------------------------------------------------------------------------
    // Create dolfinx stiffness vector
    double speedOfSound = 1500.0;
	std::shared_ptr<fem::Constant<double>> c0 
		= std::make_shared<fem::Constant<double>>(speedOfSound);
	std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
		fem::create_form<double>(
            *form_form_a, {V}, {{"u", u}}, {{"c0", c0}}, {}));
	la::Vector<double> s0(index_map, bs);

    common::Timer t_dolfinx_assembly("~ dolfinx vector assembly");
    t_dolfinx_assembly.start();
	fem::assemble_vector(s0.mutable_array(), *a);
    t_dolfinx_assembly.stop();

	s0.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
	// Create precomputed stiffness operator
	Stiffness<double, 4, 5> stiffness(V);
	la::Vector<double> s1(index_map, bs);

    common::Timer t_precomputed_assembly("~ stiffness vector assembly");
    t_precomputed_assembly.start();
	stiffness(*u->x(), s1);
    t_precomputed_assembly.stop();

    s1.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Create precomputed spectral stiffness operator
    common::Timer t_spectral_construct("~ construct spectral");
    t_spectral_construct.start();
    SpectralStiffness<double, 4, 5> spec_stiffness(V);
    t_spectral_construct.stop();
    la::Vector<double> s2(index_map, bs);

    common::Timer t_spectral_assembly("~ spectral vector assembly");
    t_spectral_assembly.start();
    spec_stiffness(*u->x(), s2);
    t_spectral_assembly.stop();

    s2.scatter_rev(common::IndexMap::Mode::add);

    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);

    // Check equality
    std::size_t local_size = s0.mutable_array().size();
    auto s_dolfinx = xt::adapt(s0.mutable_array().data(), {local_size});
    auto s_precomp = xt::adapt(s1.mutable_array().data(), {local_size});
    auto s_spectral = xt::adapt(s2.mutable_array().data(), {local_size});

    assert(xt::allclose(s_dolfinx, s_precomp));
    assert(xt::allclose(s_dolfinx, s_spectral));

  }
  common::subsystem::finalize_mpi();
  return 0;
}
