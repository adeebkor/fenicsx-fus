//
// The code to measure the time of assemble of each kernels
// ========================================================
// Copyright (C) 2024 Adeeb Arif Kor

#include "forms.h"
#include "precompute_op.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iomanip>
#include <iostream>

#define T_MPI MPI_DOUBLE
using T = double;

using namespace dolfinx;

template <typename T>
const T compute_diffusivity_of_sound(const T w0, const T c0, const T alpha) {
  const T diffusivity = 2.0 * alpha * c0 * c0 * c0 / w0 / w0;

  return diffusivity;
}

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const T sourceFrequency = 0.5e6; // (Hz)
    // const T sourceAmplitude = 60000;  // (Pa)
    // const T period = 1 / sourceFrequency;  // (s)
    const T angularFrequency = 2 * M_PI * sourceFrequency; // (rad/s)

    // Material parameters (Water)
    const T speedOfSoundWater = 1500.0; // (m/s)
    const T densityWater = 1000.0;      // (kg/m^3)
    const T nonlinearCoefficientWater = 3.5;
    const T diffusivityOfSoundWater = 0.0;

    // Material parameters (Skin)
    const T speedOfSoundSkin = 1610.0;           // (m/s)
    const T densitySkin = 1090.0;                // (kg/m^3)
    const T attenuationCoefficientdBSkin = 20.0; // (dB/m)
    const T nonlinearCoefficientSkin = 4.9;
    const T attenuationCoefficientNpSkin = attenuationCoefficientdBSkin / 20 * log(10);
    const T diffusivityOfSoundSkin = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundSkin, attenuationCoefficientNpSkin);

    // Material parameters (Cortical bone)
    const T speedOfSoundCortBone = 2800.0;            // (m/s)
    const T densityCortBone = 1850.0;                 // (kg/m^3)
    const T attenuationCoefficientdBCortBone = 400.0; // (dB/m)
    const T nonlinearCoefficientCortBone = 8.0;
    const T attenuationCoefficientNpCortBone = attenuationCoefficientdBCortBone / 20 * log(10);
    const T diffusivityOfSoundCortBone = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundCortBone, attenuationCoefficientNpCortBone);

    // Material parameters (Trabecular bone)
    const T speedOfSoundTrabBone = 2300.0;            // (m/s)
    const T densityTrabBone = 1700.0;                 // (kg/m^3)
    const T attenuationCoefficientdBTrabBone = 800.0; //(dB/m)
    const T nonlinearCoefficientTrabBone = 7.0;
    const T attenuationCoefficientNpTrabBone = attenuationCoefficientdBTrabBone / 20 * log(10);
    const T diffusivityOfSoundTrabBone = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundTrabBone, attenuationCoefficientNpTrabBone);

    // Material parameters (Brain)
    const T speedOfSoundBrain = 1560.0;           // (m/s)
    const T densityBrain = 1040.0;                // (kg/m^3)
    const T attenuationCoefficientdBBrain = 30.0; // (dB/m)
    const T nonlinearCoefficientBrain = 4.3;
    const T attenuationCoefficientNpBrain = attenuationCoefficientdBBrain / 20 * log(10);
    const T diffusivityOfSoundBrain = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundBrain, attenuationCoefficientNpBrain);

    // FE parameters
    const int degreeOfBasis = 4;
    const int numberOfQuadraturePoint = 5;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement<T>(mesh::CellType::hexahedron, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "/home/mabm4/rds/hpc-work/mesh/transducer_3d_6/mesh.xdmf",
                       "r");
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        fmesh.read_mesh(element, mesh::GhostMode::none, "transducer_3d_6"));
    mesh->topology()->create_connectivity(2, 3);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "transducer_3d_6_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "transducer_3d_6_facets"));

    // Mesh parameters
    const int tdim = mesh->topology()->dim();
    const int num_cell = mesh->topology()->index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<T> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<T>::iterator min_mesh_size_local
        = std::min_element(mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx = std::distance(mesh_size_local.begin(), min_mesh_size_local);
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Define DG function space for the physical parameters of the domain
    auto V_DG = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_a0, "c0", mesh));

    // Define cell functions
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<T>>(V_DG);
    auto beta0 = std::make_shared<fem::Function<T>>(V_DG);

    auto ncells = V_DG->dofmap()->index_map->size_global();

    auto cells_1 = mt_cell->find(1);
    auto cells_2 = mt_cell->find(2);
    auto cells_3 = mt_cell->find(3);
    auto cells_4 = mt_cell->find(4);
    auto cells_5 = mt_cell->find(5);
    auto cells_6 = mt_cell->find(6);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundWater; });
    std::for_each(cells_2.begin(), cells_2.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundSkin; });
    std::for_each(cells_3.begin(), cells_3.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundCortBone; });
    std::for_each(cells_4.begin(), cells_4.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundTrabBone; });
    std::for_each(cells_5.begin(), cells_5.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundCortBone; });
    std::for_each(cells_6.begin(), cells_6.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSoundBrain; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { rho0_[i] = densityWater; });
    std::for_each(cells_2.begin(), cells_2.end(), [&](std::int32_t& i) { rho0_[i] = densitySkin; });
    std::for_each(cells_3.begin(), cells_3.end(),
                  [&](std::int32_t& i) { rho0_[i] = densityCortBone; });
    std::for_each(cells_4.begin(), cells_4.end(),
                  [&](std::int32_t& i) { rho0_[i] = densityTrabBone; });
    std::for_each(cells_5.begin(), cells_5.end(),
                  [&](std::int32_t& i) { rho0_[i] = densityCortBone; });
    std::for_each(cells_6.begin(), cells_6.end(),
                  [&](std::int32_t& i) { rho0_[i] = densityBrain; });
    rho0->x()->scatter_fwd();

    std::span<T> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundWater; });
    std::for_each(cells_2.begin(), cells_2.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundSkin; });
    std::for_each(cells_3.begin(), cells_3.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundCortBone; });
    std::for_each(cells_4.begin(), cells_4.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundTrabBone; });
    std::for_each(cells_5.begin(), cells_5.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundCortBone; });
    std::for_each(cells_6.begin(), cells_6.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSoundBrain; });
    delta0->x()->scatter_fwd();

    std::span<T> beta0_ = beta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientWater; });
    std::for_each(cells_2.begin(), cells_2.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientSkin; });
    std::for_each(cells_3.begin(), cells_3.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientCortBone; });
    std::for_each(cells_4.begin(), cells_4.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientTrabBone; });
    std::for_each(cells_5.begin(), cells_5.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientCortBone; });
    std::for_each(cells_6.begin(), cells_6.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficientBrain; });
    beta0->x()->scatter_fwd();

    // Define function space
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(functionspace_form_forms_a0, "u", mesh));

    auto ndofs = V->dofmap()->index_map->size_global();

    // Define field functions
    auto index_map = V->dofmap()->index_map;
    auto bs = V->dofmap()->index_map_bs();

    auto u = std::make_shared<fem::Function<T>>(V);
    auto u_n = std::make_shared<fem::Function<T>>(V);
    auto v_n = std::make_shared<fem::Function<T>>(V);

    // Define source function
    auto g = std::make_shared<fem::Function<T>>(V);
    auto g_ = g->x()->mutable_array();
    auto dg = std::make_shared<fem::Function<T>>(V);
    auto dg_ = dg->x()->mutable_array();

    // Define forms
    std::span<T> u_ = u->x()->mutable_array();
    std::fill(u_.begin(), u_.end(), 1.0);

    std::span<T> u_n_ = u_n->x()->mutable_array();
    std::fill(u_n_.begin(), u_n_.end(), 2.0);

    // Compute exterior facets
    std::map<fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>> fd;
    auto facet_domains = fem::compute_integration_domains(
      fem::IntegralType::exterior_facet, *V->mesh()->topology_mutable(), 
      mt_facet->indices(), mesh->topology()->dim() - 1, mt_facet->values());
    for (auto& facet : facet_domains) {
      std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>> x;
      x.emplace_back(facet.first, std::span(facet.second.data(), facet.second.size()));
      fd.insert({fem::IntegralType::exterior_facet, std::move(x)});
    } 

    // ------------------------------------------------------------------------
    // Assembly of a0
    auto a0 = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_forms_a0, {V}, {{"u", u}, {"c0", c0}, {"rho0", rho0}}, {}, {}));

    auto m0 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto m0_ = m0->mutable_array();

    common::Timer m0_assembly("~ m0 assembly");
    m0_assembly.start();

    std::fill(m0_.begin(), m0_.end(), 0.0);
    fem::assemble_vector(m0_, *a0);
    m0->scatter_rev(std::plus<T>());

    m0_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of a0 (precompute)
    std::vector<T> a0_pc_coeffs(c0_.size());
    for (std::size_t i = 0; i < a0_pc_coeffs.size(); ++i)
      a0_pc_coeffs[i] = 1.0 / rho0_[i] / c0_[i] / c0_[i];

    Mass3D<T, degreeOfBasis, numberOfQuadraturePoint> a0_pc_operator(V);

    auto m0_pc = std::make_shared<la::Vector<T>>(index_map, bs);
    auto m0_pc_ = m0_pc->mutable_array();

    common::Timer m0_pc_assembly("~ m0_pc assembly");
    m0_pc_assembly.start();

    std::fill(m0_pc_.begin(), m0_pc_.end(), 0.0);
    a0_pc_operator(*u->x(), a0_pc_coeffs, *m0_pc);
    m0_pc->scatter_rev(std::plus<T>());

    m0_pc_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of a1
    auto a1 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_a1, {V}, {{"u", u}, {"c0", c0}, {"rho0", rho0}, {"delta0", delta0}}, {},
        fd));

    auto m1 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto m1_ = m1->mutable_array();

    common::Timer m1_assembly("~ m1 assembly");
    m1_assembly.start();

    std::fill(m1_.begin(), m1_.end(), 0.0);
    fem::assemble_vector(m1_, *a1);
    m1->scatter_rev(std::plus<T>());

    m1_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of a2
    auto a2 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_a2, {V}, {{"u", u}, {"c0", c0}, {"rho0", rho0}, {"beta0", beta0}, {"u_n", u_n}},
        {}, {}));

    auto m2 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto m2_ = m2->mutable_array();

    common::Timer m2_assembly("~ m2 assembly");
    m2_assembly.start();

    std::fill(m2_.begin(), m2_.end(), 0.0);
    fem::assemble_vector(m2_, *a2);
    m2->scatter_rev(std::plus<T>());

    m2_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of a2 (precompute)
    std::vector<T> a2_pc_coeffs(c0_.size());
    for (std::size_t i = 0; i < a2_pc_coeffs.size(); ++i)
      a2_pc_coeffs[i] = -2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] / c0_[i] / c0_[i] / c0_[i];

    Mass3D<T, degreeOfBasis, numberOfQuadraturePoint> a2_pc_operator(V);

    auto m2_pc = std::make_shared<la::Vector<T>>(index_map, bs);
    auto m2_pc_ = m2_pc->mutable_array();

    common::Timer m2_pc_assembly("~ m2_pc assembly");
    m2_pc_assembly.start();

    std::fill(m2_pc_.begin(), m2_pc_.end(), 0.0);
    a2_pc_operator(*u_n->x(), a2_pc_coeffs, *m2_pc);
    m2_pc->scatter_rev(std::plus<T>());

    m2_pc_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L0
    auto L0 = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_forms_L0, {V}, {{"u_n", u_n}, {"rho0", rho0}}, {}, {}));

    auto b0 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b0_ = b0->mutable_array();

    common::Timer b0_assembly("~ b0 assembly");
    b0_assembly.start();

    std::fill(b0_.begin(), b0_.end(), 0.0);
    fem::assemble_vector(b0_, *L0);
    b0->scatter_rev(std::plus<T>());

    b0_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L0 (precompute)

    std::vector<T> L0_pc_coeffs(c0_.size());
    for (std::size_t i = 0; i < L0_pc_coeffs.size(); ++i)
      L0_pc_coeffs[i] = -1.0 / rho0_[i];

    Stiffness3D<T, degreeOfBasis, numberOfQuadraturePoint> L0_pc_operator(V);

    auto b0_pc = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b0_pc_ = b0_pc->mutable_array();

    common::Timer b0_pc_assembly("~ b0_pc assembly");
    b0_pc_assembly.start();

    std::fill(b0_pc_.begin(), b0_pc_.end(), 0.0);
    L0_pc_operator(*u->x(), L0_pc_coeffs, *b0_pc);
    b0_pc->scatter_rev(std::plus<T>());

    b0_pc_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L1
    auto L1 = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_forms_L1, {V}, {{"g", g}, {"rho0", rho0}}, {},
                            fd));

    auto b1 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b1_ = b1->mutable_array();

    common::Timer b1_assembly("~ b1 assembly");
    b1_assembly.start();

    std::fill(b1_.begin(), b1_.end(), 0.0);
    fem::assemble_vector(b1_, *L1);
    b1->scatter_rev(std::plus<T>());

    b1_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L2
    auto L2 = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_forms_L2, {V}, {{"v_n", v_n}, {"rho0", rho0}, {"c0", c0}}, {},
                            fd));

    auto b2 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b2_ = b2->mutable_array();

    common::Timer b2_assembly("~ b2 assembly");
    b2_assembly.start();

    std::fill(b2_.begin(), b2_.end(), 0.0);
    fem::assemble_vector(b2_, *L2);
    b2->scatter_rev(std::plus<T>());

    b2_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L3
    auto L3 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_L3, {V}, {{"v_n", v_n}, {"rho0", rho0}, {"c0", c0}, {"delta0", delta0}}, {},
        {}));

    auto b3 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b3_ = b3->mutable_array();

    common::Timer b3_assembly("~ b3 assembly");
    b3_assembly.start();

    std::fill(b3_.begin(), b3_.end(), 0.0);
    fem::assemble_vector(b3_, *L3);
    b3->scatter_rev(std::plus<T>());

    b3_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L3 (precompute)

    std::vector<T> L3_pc_coeffs(c0_.size());
    for (std::size_t i = 0; i < L3_pc_coeffs.size(); ++i)
      L3_pc_coeffs[i] = -delta0_[i] / rho0_[i] / c0_[i] / c0_[i];

    Stiffness3D<T, degreeOfBasis, numberOfQuadraturePoint> L3_pc_operator(V);

    auto b3_pc = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b3_pc_ = b3_pc->mutable_array();

    common::Timer b3_pc_assembly("~ b3_pc assembly");
    b3_pc_assembly.start();

    std::fill(b3_pc_.begin(), b3_pc_.end(), 0.0);
    L3_pc_operator(*u->x(), L3_pc_coeffs, *b3_pc);
    b3_pc->scatter_rev(std::plus<T>());

    b3_pc_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L4
    auto L4 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_L4, {V}, {{"dg", dg}, {"rho0", rho0}, {"c0", c0}, {"delta0", delta0}}, {}, {}));

    auto b4 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b4_ = b4->mutable_array();

    common::Timer b4_assembly("~ b4 assembly");
    b4_assembly.start();

    std::fill(b4_.begin(), b4_.end(), 0.0);
    fem::assemble_vector(b4_, *L4);
    b4->scatter_rev(std::plus<T>());

    b4_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L5
    auto L5 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_L5, {V}, {{"v_n", v_n}, {"rho0", rho0}, {"c0", c0}, {"beta0", beta0}}, {}, {}));

    auto b5 = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b5_ = b5->mutable_array();

    common::Timer b5_assembly("~ b5 assembly");
    b5_assembly.start();

    std::fill(b5_.begin(), b5_.end(), 0.0);
    fem::assemble_vector(b5_, *L5);
    b5->scatter_rev(std::plus<T>());

    b5_assembly.stop();

    // ------------------------------------------------------------------------
    // Assembly of L5 (precompute)
    std::vector<T> L5_pc_coeffs(c0_.size());
    for (std::size_t i = 0; i < L5_pc_coeffs.size(); ++i)
      L5_pc_coeffs[i] = 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] / c0_[i] / c0_[i] / c0_[i];

    Mass3D<T, degreeOfBasis, numberOfQuadraturePoint> L5_pc_operator(V);

    auto b5_pc = std::make_shared<la::Vector<T>>(index_map, bs);
    auto b5_pc_ = b5_pc->mutable_array();

    common::Timer b5_pc_assembly("~ b5_pc assembly");
    b5_pc_assembly.start();

    std::fill(b5_pc_.begin(), b5_pc_.end(), 0.0);
    L5_pc_operator(*v_n->x(), L5_pc_coeffs, *b5_pc);
    b5_pc->scatter_rev(std::plus<T>());

    b5_pc_assembly.stop();

    // ------------------------------------------------------------------------
    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);

    if (mpi_rank == 0) {
      std::cout << "Degrees of freedom: " << ndofs << "\n";
      std::cout << "Number of cells: " << ncells << "\n";
    }
  }
  PetscFinalize();
}
