//
// Westervelt solver for 2D planewave problem with attenuation
// - structured mesh
// - first-order Sommerfeld ABC
// ===========================================================
// Copyright (C) 2023 Adeeb Arif Kor

#include "Westervelt.hpp"
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

    // Material parameters
    const T speedOfSound = 1500; // (m/s)
    const T density = 1000;      // (kg/m^3)
    const T nonlinearCoefficient = 300.0;
    const T attenuationCoefficientdB = 50.0; // (dB/m)
    const T attenuationCoefficientNp = attenuationCoefficientdB / 20 * log(10);
    const T diffusivityOfSound
        = compute_diffusivity_of_sound(angularFrequency, speedOfSound, attenuationCoefficientNp);

    // Domain parameters
    const T domainLength = 0.12; // (m)

    // FE parameters
    const int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement<T>(mesh::CellType::quadrilateral, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        fmesh.read_mesh(element, mesh::GhostMode::none, "planewave_2d_1"));
    mesh->topology()->create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "planewave_2d_1_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "planewave_2d_1_facets"));

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
        fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<T>>(V_DG);
    auto beta0 = std::make_shared<fem::Function<T>>(V_DG);

    auto cells_1 = mt_cell->find(1);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(), [&](std::int32_t& i) { c0_[i] = speedOfSound; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(), [&](std::int32_t& i) { rho0_[i] = density; });
    rho0->x()->scatter_fwd();

    std::span<T> beta0_ = beta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { beta0_[i] = nonlinearCoefficient; });
    beta0->x()->scatter_fwd();

    std::span<T> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { delta0_[i] = diffusivityOfSound; });
    delta0->x()->scatter_fwd();

    // Temporal parameters
    const T CFL = 0.8;
    T timeStepSize = CFL * meshSizeMinGlobal / (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime = domainLength / speedOfSound + 4.0 / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0) {
      std::cout << "Problem type: Planewave 2D (with attenuation)"
                << "\n";
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
    auto model = WesterveltSpectral2D<T, 4>(mesh, mt_facet, c0, rho0, delta0, beta0,
                                                 sourceFrequency, sourceAmplitude, speedOfSound);

    // Solve
    common::Timer tsolve("Solve time");

    model.init();

    tsolve.start();
    model.rk4(startTime, finalTime, timeStepSize);
    tsolve.stop();

    if (mpi_rank == 0) {
      std::cout << "Solve time: " << tsolve.elapsed()[0] << std::endl;
      std::cout << "Time per step: " << tsolve.elapsed()[0] / numberOfStep << std::endl;
    }

    // Final solution
    auto u_n = model.u_sol();

    // Output to VTX
    dolfinx::io::VTXWriter<T> u_out(mesh->comm(), "output_final.bp", {u_n}, "BP4");
    u_out.write(0.0);
  }
}