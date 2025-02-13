//
// Linear solver for the 2D circular penetrable scatterer
// - circular scatterer
// - wavelength < scatterer radius
// ======================================================
// Copyright (C) 2024 Adeeb Arif Kor

#include "Linear.hpp"
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

    // Material parameters
    const T speedOfSound1 = 1500; // (m/s)
    const T density1 = 1000;      // (kg/m^3)
    const T speedOfSound2 = 3500; // (m/s)
    const T density2 = 1900;      // (kg/m^3)

    // Source parameters
    const T sourceFrequency = 2000;      // (Hz)
    const T sourceSpeed = 1.0;  // (m/s)
    const T sourceAmplitude = density1 * speedOfSound1 * sourceSpeed; // (Pa)
    const T period = 1 / sourceFrequency; // (s)

    // Domain parameters
    const T wavelength = speedOfSound1 / sourceFrequency;
    const T scattererRadius = 1.0 * wavelength;
    const T domainScale = 10.0;
    const T domainLength =  1.5 * domainScale * scattererRadius + 2 * scattererRadius; // (m)
    
    // FE parameters
    const int degreeOfBasis = 4;

    // Read mesh and mesh tags
    int geom_order = 2;
    auto coord_element = fem::CoordinateElement<T>(mesh::CellType::quadrilateral, geom_order);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        fmesh.read_mesh(coord_element, mesh::GhostMode::none, "circular_scatterer_penetrable_2d_1"));
    mesh->topology()->create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "circular_scatterer_penetrable_2d_1_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "circular_scatterer_penetrable_2d_1_facets"));

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

    // Finite element
    basix::FiniteElement element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::quadrilateral, degreeOfBasis,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false
    );

    // Define DG function space for the physical parameters of the domain
    basix::FiniteElement element_DG = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::quadrilateral, 0,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, true
    );
    auto V_DG = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(mesh, element_DG));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

    auto cells_1 = mt_cell->find(1);
    auto cells_2 = mt_cell->find(2);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSound1; });
    std::for_each(cells_2.begin(), cells_2.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSound2; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { rho0_[i] = density1; });
    std::for_each(cells_2.begin(), cells_2.end(),
                  [&](std::int32_t& i) { rho0_[i] = density2; });
    rho0->x()->scatter_fwd();

    // Temporal parameters
    const T CFL = 0.9;
    T timeStepSize = CFL * meshSizeMinGlobal / (speedOfSound2 * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime = domainLength / speedOfSound1 + 8.0 / sourceFrequency;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0) {
      std::cout << "Problem type: Planewave 2D"
                << "\n";
      std::cout << "Scatterer radius: " << scattererRadius << "\n";
      std::cout << "Wavelength: " << wavelength << "\n";
      std::cout << "Speed of sound (1): " << speedOfSound1 << "\n";
      std::cout << "Speed of sound (2): " << speedOfSound2 << "\n";
      std::cout << "Density (1): " << density1 << "\n";
      std::cout << "Density (2): " << density2 << "\n";
      std::cout << "Source frequency: " << sourceFrequency << "\n";
      std::cout << "Source amplitude: " << sourceAmplitude << "\n";
      std::cout << "Domain length: " << domainLength << "\n";
      std::cout << "Polynomial basis degree: " << degreeOfBasis << "\n";
      std::cout << "Minimum mesh size: ";
      std::cout << std::setprecision(2) << meshSizeMinGlobal << "\n";
      std::cout << "CFL number: " << CFL << "\n";
      std::cout << "Time step size: " << timeStepSize << "\n";
      std::cout << "Final time: " << finalTime << "\n";
      std::cout << "Number of steps per period: " << stepPerPeriod << "\n";
      std::cout << "Total number of steps: " << numberOfStep << "\n";
    }

    // Model
    auto model = LinearSpectral2D<T, degreeOfBasis>(element, mesh, mt_facet, c0, rho0, sourceFrequency,
                                                    sourceAmplitude, speedOfSound1);

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
    dolfinx::io::VTXWriter<T> u_out(mesh->comm(), "output_final.bp", {u_n}, "bp5");
    u_out.write(0.0);
  }
}