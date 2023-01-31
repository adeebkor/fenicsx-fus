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

#define T_MPI MPI_DOUBLE
using T = double;

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
    const T sourceFrequency = 0.5e6;  // (Hz)
    const T sourceAmplitude = 60000;  // (Pa)
    const T period = 1 / sourceFrequency;  // (s)
    const T angularFrequency = 2 * M_PI * sourceFrequency;  // (rad/s)

    // Material parameters
    const T speedOfSound = 1500;  // (m/s)
    const T density = 1000;  // (kg/m^3)
    const T attenuationCoefficientdB = 100.0;  // (dB/m)
    const T attenuationCoefficientNp = attenuationCoefficientdB / 20 * log(10);
    const T diffusivityOfSound = compute_diffusivity_of_sound(
      angularFrequency, speedOfSound, attenuationCoefficientNp);

    // Domain parameters
    const T domainLength = 0.12;  // (m)

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
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Define DG function space for the physical parameters of the domain
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<T>>(V_DG);

    auto cells_1 = mt_cell->find(1);
    auto cells_2 = mt_cell->find(2);
    
    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { c0_[i] = speedOfSound; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { c0_[i] = speedOfSound; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { rho0_[i] = density; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { rho0_[i] = density; });
    rho0->x()->scatter_fwd();

    std::span<T> delta0_ = delta0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
      [&](std::int32_t &i) { delta0_[i] = 0.0; });
    std::for_each(cells_2.begin(), cells_2.end(),
      [&](std::int32_t &i) { delta0_[i] = diffusivityOfSound; });
    delta0->x()->scatter_fwd();

    // Temporal parameters
    const T CFL = 0.4;
    T timeStepSize = CFL * meshSizeMinGlobal / 
      (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime = domainLength / speedOfSound + 4.0 
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
    auto model = LossySpectral2D<T, degreeOfBasis>(
      mesh, mt_facet, c0, rho0, delta0, sourceFrequency, sourceAmplitude,
      speedOfSound);

    // Solve
    model.init();

    common::Timer solve_time("~ SOLVE TIME");
    solve_time.start();

    model.rk4(startTime, finalTime, timeStepSize);

    solve_time.stop();

    // Final solution
    auto u_n = model.u_sol();

    // Output to VTX
    // dolfinx::io::VTXWriter u_out(mesh->comm(), "output_final.bp", {u_n});
    // u_out.write(0.0);

    // ------------------------------------------------------------------------
    // Computing function evaluation parameters

    std::string fname;

    // Grid parameters
    const std::size_t Nr = 241;
    const std::size_t Nz = 141;

    // Create evaluation point coordinates
    std::vector<double> point_coordinates(3 * Nr * Nz);
    for (std::size_t i = 0; i < Nz; ++i) {
      for (std::size_t j = 0; j < Nr; ++j) {
        point_coordinates[3*j + 3*i*Nr] = i * 0.12 / (Nz - 1);
        point_coordinates[3*j + 3*i*Nr + 1] = j * 0.07 / (Nr - 1) - 0.035;
        point_coordinates[3*j + 3*i*Nr + 2] = 0.0;
      }
    }

    // Compute evaluation parameters
    auto bb_tree = geometry::BoundingBoxTree(*mesh, mesh->topology().dim());
    auto cell_candidates = compute_collisions(bb_tree, point_coordinates);
    auto colliding_cells = geometry::compute_colliding_cells(
      *mesh, cell_candidates, point_coordinates);

    std::vector<std::int32_t> cells;
    std::vector<double> points_on_proc;

    for (std::size_t i = 0; i < Nr*Nz; ++i) {
      auto link = colliding_cells.links(i);
      if (link.size() > 0) {
        points_on_proc.push_back(point_coordinates[3*i]);
        points_on_proc.push_back(point_coordinates[3*i + 1]);
        points_on_proc.push_back(point_coordinates[3*i + 2]);
        cells.push_back(link[0]);
      }
    }

    std::size_t num_points_local = points_on_proc.size() / 3;
    std::vector<T> u_eval(num_points_local);

    T* u_value = u_eval.data();
    double* p_value = points_on_proc.data();

    // Evaluate function
    u_n->eval(points_on_proc, {num_points_local, 3}, cells, u_eval,
              {num_points_local, 1});
    u_value = u_eval.data();

    // Write evaluation from each process to a single text file
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < mpi_size; ++i) {
      if (mpi_rank == i) {
        fname = "/home/mabm4/data/pressure_on_xz_plane_.txt";
        std::ofstream txt_file(fname, std::ios_base::app);
        for (std::size_t i = 0; i < num_points_local; ++i) {
          txt_file << *(p_value + 3 * i) << ","
                   << *(p_value + 3 * i + 1) << "," 
                   << *(u_value + i) << std::endl;
        }
        txt_file.close();
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    // ------------------------------------------------------------------------

    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);
  }
}