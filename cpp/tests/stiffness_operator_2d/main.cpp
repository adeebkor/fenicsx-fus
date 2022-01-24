#include "form.h"
#include "operators_2d.hpp"
#include "precomputation.hpp"
#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    std::cout.precision(15);

    // Create mesh and function space
    std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
        mesh::create_rectangle(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {2, 2},
                               mesh::CellType::quadrilateral, mesh::GhostMode::none));

    std::shared_ptr<fem::FunctionSpace> V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_form_a, "u", mesh));

    // Get index map and block size
    std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    // Create stiffness operator
    std::shared_ptr<fem::Function<double>> u = std::make_shared<fem::Function<double>>(V);

    u->interpolate(
        [](auto& x) -> xt::xarray<PetscScalar> { return xt::row(x, 0) * xt::row(x, 1); });

    std::map<std::string, double> params;
    params["c0"] = 1486.0;
    StiffnessOperator<double> stiffness_operator(V, 3, params);
    la::Vector<double> s(index_map, bs);
    stiffness_operator(*u->x(), s);

    std::shared_ptr<fem::Constant<double>> c0
        = std::make_shared<fem::Constant<double>>(params["c0"]);
    std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_form_a, {V}, {{"u", u}}, {{"c0", c0}}, {}));
    la::Vector<double> s_ref(index_map, bs);
    fem::assemble_vector(s_ref.mutable_array(), *a);
    s_ref.scatter_rev(common::IndexMap::Mode::add);

    auto _s = s.array();
    auto _s_ref = s_ref.array();
    for (std::size_t i = 0; i < _s.size(); ++i) {
      std::cout << _s[i] << "\t " << _s_ref[i] << "\t " << std::abs(_s[i] - _s_ref[i]) << std::endl;
    }
  }
  common::subsystem::finalize_mpi();
  return 0;
}
