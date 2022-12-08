//
// Linear solver for the 2D planewave problem with attenuation
// - structured mesh
// - first-order Sommerfeld ABC
// - different attenuation between 2 medium (x < 0.06 m, x > 0.06 m)
// =================================================================
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
    const double speedOfSound = 1500;  // (m/s)
    const double density = 1000;  // (kg/m^3)
    const double attenuationCoefficientdB = 50.0;  // (dB/m)
    const double attenuationCoefficientNp = attenuationCoefficientdB / 20 * log(10);
    const double diffusivityOfSound = compute_diffusivity_of_sound(
      angularFrequency, speedOfSound, attenuationCoefficientNp);

    // Domain parameters
    const double domainLength = 0.12;  // (m)

    // FE parameters
    const int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(
      fmesh.read_mesh(element, mesh::GhostMode::none, "planewave_2d_4"));
    mesh->topology().create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "planewave_2d_4_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "planewave_2d_4_facets"));

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
      [&](std::int32_t &i) { c0_[i] = speedOfSound; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { c0_[i] = speedOfSound; });
    c0->x()->scatter_fwd();

    std::span<double> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { rho0_[i] = density; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { rho0_[i] = density; });
    rho0->x()->scatter_fwd();

    std::span<double> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { delta0_[i] = 0.0; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { delta0_[i] = diffusivityOfSound; });
    delta0->x()->scatter_fwd();

    // Temporal parameters
    const double CFL = 0.4;
    double timeStepSize = CFL * meshSizeMinGlobal / 
      (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const double startTime = 0.0;
    const double finalTime = domainLength / speedOfSound + 4.0 
                             / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0){
      std::cout << "Problem type: Planewave 2D (with attenuation)" << "\n";
      std::cout << "Speed of sound: " << speedOfSound << "\n";
      std::cout << "Density: " << density << "\n";
      std::cout << "Diffusivity of sound: " << diffusivityOfSound << "\n";
      std::cout << "Source frequency: " << sourceFrequency << "\n";
      std::cout << "Source amplitude: " << sourceAmplitude << "\n";
      std::cout << "Domain length: " << domainLength << "\n";
      std::cout << "Polynomial basis degree: " << degreeOfBasis << "\n";
      std::cout << "Minimum mesh size: ";
      std::cout << std::setprecision(2) << meshSizeMinGlobal << "\n";
      std::cout << "CFL number: " << CFL << "\n";
      std::cout << "Time step size: " << timeStepSize << "\n";
      std::cout << "Number of steps per period: " << stepPerPeriod << "\n";
      std::cout << "Total number of steps: " << numberOfStep << "\n";
    }

    // Model
    auto model = LossySpectral2D<double, 4>(
      mesh, mt_facet, c0, rho0, delta0, sourceFrequency, sourceAmplitude,
      speedOfSound);

    // Solve
    model.init();
    model.rk4(startTime, finalTime, timeStepSize);

    // Final solution
    auto u_n = model.u_sol();

    // Output to VTX
    dolfinx::io::VTXWriter u_out(mesh->comm(), "output_final.bp", {u_n});
    u_out.write(0.0);

  }
}