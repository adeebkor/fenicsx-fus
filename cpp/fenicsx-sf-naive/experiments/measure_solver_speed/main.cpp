//
// This code measure the simulation time of the solver.
// ====================================================
// Copyright (C) 2023 Adeeb Arif Kor

#include "Lossy.hpp"
#include "forms.h"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iomanip>
#include <iostream>

#define T_MPI MPI_DOUBLE
using T = double;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const T sourceFrequency = 0.5e6;                       // (Hz)
    const T sourceAmplitude = 60000;                       // (Pa)
    const T period = 1 / sourceFrequency;                  // (s)
    const T angularFrequency = 2 * M_PI * sourceFrequency; // (rad/s)

    // Material parameters (Water)
    const T speedOfSoundWater = 1500.0; // (m/s)
    const T densityWater = 1000.0;      // (kg/m^3)

    // Material parameters (Skin)
    const T speedOfSoundSkin = 1610.0;           // (m/s)
    const T densitySkin = 1090.0;                // (kg/m^3)
    const T attenuationCoefficientdBSkin = 20.0; // (dB/m)
    const T attenuationCoefficientNpSkin = attenuationCoefficientdBSkin / 20 * log(10);
    const T diffusivityOfSoundSkin = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundSkin, attenuationCoefficientNpSkin);

    // Material parameters (Cortical bone)
    const T speedOfSoundCortBone = 2800.0;            // (m/s)
    const T densityCortBone = 1850.0;                 // (kg/m^3)
    const T attenuationCoefficientdBCortBone = 400.0; //(dB/m)
    const T attenuationCoefficientNpCortBone = attenuationCoefficientdBCortBone / 20 * log(10);
    const T diffusivityOfSoundCortBone = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundCortBone, attenuationCoefficientNpCortBone);

    // Material parameters (Trabecular bone)
    const T speedOfSoundTrabBone = 2300.0;            // (m/s)
    const T densityTrabBone = 1700.0;                 // (kg/m^3)
    const T attenuationCoefficientdBTrabBone = 800.0; //(dB/m)
    const T attenuationCoefficientNpTrabBone = attenuationCoefficientdBTrabBone / 20 * log(10);
    const T diffusivityOfSoundTrabBone = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundTrabBone, attenuationCoefficientNpTrabBone);

    // Material parameters (Brain)
    const T speedOfSoundBrain = 1560.0;           // (m/s)
    const T densityBrain = 1040.0;                // (kg/m^3)
    const T attenuationCoefficientdBBrain = 30.0; // (dB/m)
    const T attenuationCoefficientNpBrain = attenuationCoefficientdBBrain / 20 * log(10);
    const T diffusivityOfSoundBrain = compute_diffusivity_of_sound(
        angularFrequency, speedOfSoundBrain, attenuationCoefficientNpBrain);

    // FE parameters
    const int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::hexahedron, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "/home/mabm4/rds/hpc-work/mesh/transducer_3d_8/mesh.xdmf",
                       "r");
    auto mesh = std::make_shared<mesh::Mesh>(
        fmesh.read_mesh(element, mesh::GhostMode::none, "transducer_3d_8"));
    mesh->topology().create_connectivity(2, 3);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(mesh, "transducer_3d_8_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(mesh, "transducer_3d_8_facets"));

    // Mesh parameters
    const int tdim = mesh->topology().dim();
    const int num_cell = mesh->topology().index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<double> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<double>::iterator min_mesh_size_local
        = std::min_element(mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx = std::distance(mesh_size_local.begin(), min_mesh_size_local);
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Define DG function space for the physical parameters of the domain
    auto V_DG = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<T>>(V_DG);

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
    std::for_each(cells_1.begin(), cells_1.end(), [&](std::int32_t& i) { delta0_[i] = 0.0; });
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

    // Temporal parameters
    const T CFL = 0.25;
    T timeStepSize
        = CFL * meshSizeMinGlobal / (speedOfSoundCortBone * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime
        = 200 * timeStepSize; // domainLength / speedOfSoundWater + 8.0 / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    // Model
    auto model = LossySpectral3D<T, degreeOfBasis>(
        mesh, mt_facet, c0, rho0, delta0, sourceFrequency, sourceAmplitude, speedOfSoundWater);

    auto nDofs = model.number_of_dofs();

    if (mpi_rank == 0) {
      std::cout << "Measure solve time!"
                << "\n";
      std::cout << "Polynomial basis degree: " << degreeOfBasis << "\n";
      std::cout << "Minimum mesh size: ";
      std::cout << std::setprecision(2) << meshSizeMinGlobal << "\n";
      std::cout << "Degrees of freedom: " << nDofs << "\n";
      std::cout << "CFL number: " << CFL << "\n";
      std::cout << "Time step size: " << timeStepSize << "\n";
      std::cout << "Number of steps per period: " << stepPerPeriod << "\n";
      std::cout << "Total number of steps: " << numberOfStep << "\n";
    }

    // Solve
    model.init();
    model.rk4(startTime, finalTime, timeStepSize);

    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);
  }
}
