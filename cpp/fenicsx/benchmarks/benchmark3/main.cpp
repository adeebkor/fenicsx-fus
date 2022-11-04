//
// Heterogenous 3D viscoelastic wave problem
// - circular planar source
// - first-order Sommerfeld ABC
// =========================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "Lossy.hpp"
#include "forms.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {

    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const double sourceFrequency = 0.5e6;  // (Hz)
    const double sourceAmplitude = 60000;  // (Pa)
    const double period = 1 / sourceFrequency;  // (s)
    const double angularFrequency = 2 * M_PI * sourceFrequency;  // (rad/s)

    // Material parameters
    const double speedOfSoundWater = 1500.0;  // (m/s)
    const double speedOfSoundCortBone = 2800.0;  // (m/s)
    const double densityWater = 1000.0;  // (kg/m^3)
    const double densityCortBone = 1850.0;  // (kg/m^3)

    // Compute attenuation parameter
    const double attenuationCoefficientdBCortBone = 400.0;  // (m/s)
    const double attenuationCoefficientNpCortBone
      = attenuationCoefficientdBCortBone / 20 * log(10);
    const double diffusivityOfSoundCortBone = compute_diffusivity_of_sound(
      angularFrequency, speedOfSoundCortBone, 
      attenuationCoefficientNpCortBone);
    
    // Domain parameters
    const double domainLength = 0.12;  // (m)

    // FE parameters
    const int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::hexahedron, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(
      fmesh.read_mesh(element, mesh::GhostMode::none, "planar_3d_1"));
    mesh->topology().create_connectivity(2, 3);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "planar_3d_1_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "planar_3d_1_facets"));

    // Mesh parameters
    const int tdim = mesh->topology().dim();
    const int num_cell = mesh->topology().index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<double> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<double>::iterator min_mesh_size_local = std::min_element(
      mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx = std::distance(
      mesh_size_local.begin(), min_mesh_size_local);
    double meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    double meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, MPI_DOUBLE, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Define DG function space for the physical parameters of the domain
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<double>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<double>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<double>>(V_DG);

    auto cells_1 = mt_cell->find(1);
    auto cells_2 = mt_cell->find(2);

    std::span<double> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { c0_[i] = speedOfSoundWater; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { c0_[i] = speedOfSoundCortBone; });
    c0->x()->scatter_fwd();

    std::span<double> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { rho0_[i] = densityWater; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { rho0_[i] = densityCortBone; });
    rho0->x()->scatter_fwd();

    std::span<double> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { delta0_[i] = 0.0; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { delta0_[i] = diffusivityOfSoundCortBone; });
    delta0->x()->scatter_fwd();

    // Temporal parameters
    const double CFL = 0.22;
    double timeStepSize = CFL * meshSizeMinGlobal / 
      (speedOfSoundCortBone * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const double startTime = 0.0;
    // const double finalTime = 100*timeStepSize;
    const double finalTime = 0.03 / speedOfSoundWater + 2.0 / sourceFrequency;  
    // const double finalTime = domainLength / speedOfSoundWater + 8.0 / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0){
      std::cout << "Benchmark: 3" << "\n";
      std::cout << "Polynomial basis degree: " << degreeOfBasis << "\n";
      std::cout << "Minimum mesh size: ";
      std::cout << std::setprecision(2) << meshSizeMinGlobal << "\n";
      std::cout << "CFL number: " << CFL << "\n";
      std::cout << "Time step size: " << timeStepSize << "\n";
      std::cout << "Number of steps per period: " << stepPerPeriod << "\n";
      std::cout << "Total number of steps: " << numberOfStep << "\n";
    }
    
    // Model
    auto model = LossyGLL<double, 4>(
      mesh, mt_facet, c0, rho0, delta0, sourceFrequency, sourceAmplitude,
      speedOfSoundWater);

    // Solve
    common::Timer tsolve("Solve time");

    model.init();

    tsolve.start();
    model.rk4(startTime, finalTime, timeStepSize);
    tsolve.stop();

    if (mpi_rank == 0) {
      std::cout << "Solve time: " << tsolve.elapsed()[0] << std::endl;
      std::cout << "Time per step: " 
                << tsolve.elapsed()[0] / numberOfStep << std::endl;
    }

  }
}