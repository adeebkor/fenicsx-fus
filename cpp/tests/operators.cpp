// Copyright (C) 2022 Adeeb Arif Kor
//
// SPDX-License-Identifier:    MIT
//
// Unit tests for the mass forms

#include "forms.h"
#include "pc_mass_3d.hpp"
#include "sf_mass_3d.hpp"
#include "spectral_mass_3d.hpp"
#include "pc_stiffness_3d.hpp"
#include "sf_stiffness_3d.hpp"
#include "spectral_stiffness_3d.hpp"

#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    const int P = 4;
    const int Q = 6;

    // Create mesh and function space
    const std::size_t N = 8;
    std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
      mesh::create_box(
        MPI_COMM_WORLD,
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
        {N, N, N},
        mesh::CellType::hexahedron,
        mesh::GhostMode::none));

    std::shared_ptr<fem::FunctionSpace> V 
      = std::make_shared<fem::FunctionSpace>(
          fem::create_functionspace(functionspace_form_forms_M, "u", mesh));

    // Get index map and block size
    std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    // Create input function
    std::shared_ptr<fem::Function<double>> u
      = std::make_shared<fem::Function<double>>(V);
    u->interpolate([](auto& x) { return xt::sin(xt::row(x, 0)); });

    // ------------------------------------------------------------------------
    // Create dolfinx mass vector
    std::shared_ptr<fem::Form<double>> a = std::make_shared<fem::Form<double>>(
      fem::create_form<double>(*form_forms_M, {V}, {{"u", u}}, {}, {}));
    
    la::Vector<double> m0(index_map, bs);
    fem::assemble_vector(m0.mutable_array(), *a);
    m0.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Compute precompute mass vector
    Mass<double, P, Q> mass_pc(V);
    
    la::Vector<double> m1(index_map, bs);
    mass_pc(*u->x(), m1);
    m1.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Compute sum factorisation mass vector
    MassSF<double, P, Q> mass_sf(V);
    
    la::Vector<double> m2(index_map, bs);
    mass_sf(*u->x(), m2);
    m2.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
	  // Compute spectral mass vector
	  MassSpectral<double, P> mass_spectral(V);
	
    la::Vector<double> m3(index_map, bs);
	  mass_spectral(*u->x(), m3);
	  m3.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Check mass vectors equality
    std::size_t local_size = m0.mutable_array().size();
    auto m_dolfinx = xt::adapt(m0.mutable_array().data(), {local_size});
    auto m_precompute = xt::adapt(m1.mutable_array().data(), {local_size});
    auto m_sf = xt::adapt(m2.mutable_array().data(), {local_size});
    auto m_spectral = xt::adapt(m3.mutable_array().data(), {local_size});
    
    assert(xt::allclose(m_dolfinx, m_precompute));
    assert(xt::allclose(m_dolfinx, m_sf));
    assert(xt::allclose(m_dolfinx, m_spectral));

    // ------------------------------------------------------------------------
    // Create dolfinx stiffness vector
    double speedOfSound = 1500.0;
	  std::shared_ptr<fem::Constant<double>> c0 
		  = std::make_shared<fem::Constant<double>>(speedOfSound);
	  std::shared_ptr<fem::Form<double>> L = std::make_shared<fem::Form<double>>(
		  fem::create_form<double>(
            *form_forms_L, {V}, {{"u", u}}, {{"c0", c0}}, {}));
	
    la::Vector<double> s0(index_map, bs);
	  fem::assemble_vector(s0.mutable_array(), *L);
	  s0.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
	  // Compute precomputed stiffness vector
	  Stiffness<double, P, Q> stiffness_pc(V);

	  la::Vector<double> s1(index_map, bs);
	  stiffness_pc(*u->x(), s1);
    s1.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
	  // Compute sum factorisation stiffness vector
    StiffnessSF<double, P, Q> stiffness_sf(V);
    
    la::Vector<double> s2(index_map, bs);
    stiffness_sf(*u->x(), s2);
    s2.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Create spectral stiffness vector
    StiffnessSpectral<double, P> stiffness_spectral(V);
    
    la::Vector<double> s3(index_map, bs);
    stiffness_spectral(*u->x(), s3);
    s3.scatter_rev(common::IndexMap::Mode::add);

    // ------------------------------------------------------------------------
    // Check stiffness vectors equality
    auto s_dolfinx = xt::adapt(s0.mutable_array().data(), {local_size});
    auto s_precompute = xt::adapt(s1.mutable_array().data(), {local_size});
    auto s_sf = xt::adapt(s2.mutable_array().data(), {local_size});
    auto s_spectral = xt::adapt(s3.mutable_array().data(), {local_size});
    
    assert(xt::allclose(s_dolfinx, s_precompute));
    assert(xt::allclose(s_dolfinx, s_sf));
    assert(xt::allclose(s_dolfinx, s_spectral));
  }
  PetscFinalize();
  return 0;
}