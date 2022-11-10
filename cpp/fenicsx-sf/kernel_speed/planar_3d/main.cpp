//
// Kernel timing for the 3D planar problem
// =======================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "form.h"
#include "spectral_mass_3d.hpp"
#include "stiffness_3d.hpp"
#include "spectral_stiffness_3d.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

using namespace dolfinx;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    std::cout.precision(15); // Set print precision

    //-------------------------------------------------------------------------
    // MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // FE parameters
    const int P = 4;
    const int Q = P + 1;

    //-------------------------------------------------------------------------
    // Material parameters
    double speedOfSound = 1500.0;
  
    //-------------------------------------------------------------------------
    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::hexahedron, 1);
    io::XDMFFile xdmf(MPI_COMM_WORLD, 
      "/home/mabm4/rds/hpc-work/big_mesh/mesh.xdmf", "r");
    auto mesh
      = std::make_shared<mesh::Mesh>(
          xdmf.read_mesh(element, mesh::GhostMode::none, "planar3d"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "planar3d_boundaries"));

    //-------------------------------------------------------------------------
    // Get number of cells and tag facets
    std::int32_t num_cells = mesh->topology().index_map(3)->size_local();
    std::vector<std::int32_t> facets1_ = mt->find(1); // source facet
    std::vector<std::int32_t> facets2_ = mt->find(2); // ABC facet

    //-------------------------------------------------------------------------
    // Create function space
    std::shared_ptr<fem::FunctionSpace> V
      = std::make_shared<fem::FunctionSpace>(
          fem::create_functionspace(functionspace_form_form_a0, "u", mesh));

    //-------------------------------------------------------------------------
    // Get index map and block size
	std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
	int bs = V->dofmap()->index_map_bs();

    //-------------------------------------------------------------------------
    // Get degrees of freedom
    const xtl::span<const std::int32_t> facets1 
      = xtl::span<const std::int32_t>(facets1_);
    const xtl::span<const std::int32_t> facets2 
      = xtl::span<const std::int32_t>(facets2_);

    std::size_t ndofs = index_map->size_global();
    std::vector<std::int32_t> dof_facets1
      = fem::locate_dofs_topological(*V, 2, facets1);
    std::vector<std::int32_t> dof_facets2
      = fem::locate_dofs_topological(*V, 2, facets2);

    //-------------------------------------------------------------------------
    // Assemble forms

    // Create constant
    std::shared_ptr<fem::Constant<double>> c0 
      = std::make_shared<fem::Constant<double>>(speedOfSound);

    // Create coefficient
    std::shared_ptr<fem::Function<double>> u 
      = std::make_shared<fem::Function<double>>(V);
    std::shared_ptr<fem::Function<double>> v 
      = std::make_shared<fem::Function<double>>(V);
    std::shared_ptr<fem::Function<double>> g 
      = std::make_shared<fem::Function<double>>(V);
    std::shared_ptr<fem::Function<double>> u_n
      = std::make_shared<fem::Function<double>>(V);
    std::shared_ptr<fem::Function<double>> v_n
      = std::make_shared<fem::Function<double>>(V);

    // a0
    xtl::span<double> _u = u->x()->mutable_array();
    std::fill(_u.begin(), _u.end(), 1.0);

    std::shared_ptr<fem::Form<double>> a0
      = std::make_shared<fem::Form<double>>(
          fem::create_form<double>(*form_form_a0, {V}, {{"u", u}}, {}, {}));
    la::Vector<double> m0_dolfinx(index_map, bs);

    common::Timer t_dolfinx_m0("~ dolfinx a0 assembly");
    t_dolfinx_m0.start();
    fem::assemble_vector(m0_dolfinx.mutable_array(), *a0);
    t_dolfinx_m0.stop();
    m0_dolfinx.scatter_rev(common::IndexMap::Mode::add);

    
    SpectralMass<double, P, Q> mass(V);
    la::Vector<double> m0_spectral(index_map, bs);

    common::Timer t_spectral_m0("~ spectral a0 assembly");
    t_spectral_m0.start();
    mass(*u->x(), m0_spectral);
    t_spectral_m0.stop();
    m0_spectral.scatter_rev(common::IndexMap::Mode::add);

    std::size_t local_size = m0_dolfinx.mutable_array().size();
    auto m_dolfinx = xt::adapt(
      m0_dolfinx.mutable_array().data(), {local_size});
    auto m_spectral = xt::adapt(
      m0_spectral.mutable_array().data(), {local_size});

    assert(xt::allclose(m_dolfinx, m_spectral));

    // L0
    u_n->interpolate([](auto& x) { return xt::sin(xt::row(x, 0)); });

    std::shared_ptr<fem::Form<double>> L0
      = std::make_shared<fem::Form<double>>(fem::create_form<double>(
          *form_form_L0, {V}, {{"u_n", u_n}}, {{"c0", c0}}, {}));
    la::Vector<double> s0_dolfinx(index_map, bs);

    common::Timer t_dolfinx_s0("~ dolfinx L0 assembly");
    t_dolfinx_s0.start();
    fem::assemble_vector(s0_dolfinx.mutable_array(), *L0);
    t_dolfinx_s0.stop();
    s0_dolfinx.scatter_rev(common::IndexMap::Mode::add);

    Stiffness<double, P, Q> stiff(V);
    la::Vector<double> s0_precompute(index_map, bs);

    common::Timer t_precompute_s0("~ precompute L0 assembly");
    t_precompute_s0.start();
    stiff(*u_n->x(), s0_precompute);
    t_precompute_s0.stop();
    s0_precompute.scatter_rev(common::IndexMap::Mode::add);

    SpectralStiffness<double, P, Q> spec_stiff(V);
    la::Vector<double> s0_spectral(index_map, bs);

    common::Timer t_spectral_s0("~ spectral L0 assembly");
    t_spectral_s0.start();
    spec_stiff(*u_n->x(), s0_spectral);
    t_spectral_s0.stop();
    s0_spectral.scatter_rev(common::IndexMap::Mode::add);

    auto s0_dolfinx_ = xt::adapt(
        s0_dolfinx.mutable_array().data(), {local_size});
    auto s0_precomp_ = xt::adapt(
        s0_precompute.mutable_array().data(), {local_size});
    auto s0_spectral_ = xt::adapt(
        s0_spectral.mutable_array().data(), {local_size});

    assert(xt::allclose(s0_dolfinx_, s0_precomp_));
    assert(xt::allclose(s0_precomp_, s0_spectral_));

    // L1
    xtl::span<double> _g = g->x()->mutable_array();
    std::fill(_g.begin(), _g.end(), 0.0);

    std::shared_ptr<fem::Form<double>> L1 
      = std::make_shared<fem::Form<double>>(fem::create_form<double>(
          *form_form_L1, {V}, {{"g", g}}, {{"c0", c0}},
          {{dolfinx::fem::IntegralType::exterior_facet, &(*mt)}}));
    la::Vector<double> l1(index_map, bs);

    common::Timer t_l1("~ L1 assembly");
    t_l1.start();
    fem::assemble_vector(l1.mutable_array(), *L1);
    t_l1.stop();

    l1.scatter_rev(common::IndexMap::Mode::add);

    // L2
    v_n->interpolate([](auto& x) { return xt::sin(xt::row(x, 1)); });

    std::shared_ptr<fem::Form<double>> L2 
      = std::make_shared<fem::Form<double>>(fem::create_form<double>(
          *form_form_L2, {V}, {{"v_n", v_n}}, {{"c0", c0}},
          {{dolfinx::fem::IntegralType::exterior_facet, &(*mt)}}));
    la::Vector<double> l2(index_map, bs);

    common::Timer t_l2("~ L2 assembly");
    t_l2.start();
    fem::assemble_vector(l2.mutable_array(), *L2);
    t_l2.stop();

    l2.scatter_rev(common::IndexMap::Mode::add);


    //-------------------------------------------------------------------------
    // Print timings

    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);
  }
}