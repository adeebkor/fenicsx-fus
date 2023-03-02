//
// The code to collect data for determining the fraction of peak performance
// =========================================================================
// Copyright (C) 2023 Adeeb Arif Kor

#include "forms.h"
#include "spectral_op.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>

using T = float;


template <typename T>
const T compute_diffusivity_of_sound(const T w0, const T c0, const T alpha){
  const T diffusivity = 2*alpha*c0*c0*c0/w0/w0;

  return diffusivity;
}


int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Material parameters (Water)
    const T speedOfSoundWater = 1500.0;  // (m/s)
    const T densityWater = 1000.0;  // (kg/m^3)
    
    // Set polynomial degree
    const int P = 6;

    // Set number of elements
    std::size_t N;
    if (P == 2) {
      N = 168;
    }
    if (P == 3) {
      N = 112;
    }
    if (P == 4) {
      N = 84;
    }
    if (P == 5) {
      N = 68;
    }
    if (P == 6) {
      N = 56;
    }

    // Create mesh
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh>(
      mesh::create_box(
        MPI_COMM_WORLD,
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
        {N, N, N},
        mesh::CellType::hexahedron,
        part));

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "u", mesh));

    auto ndofs = V->dofmap()->index_map->size_global();

    // Create input function
    auto u = std::make_shared<fem::Function<T>>(V);
    u->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> u(x.extent(1));
        
        for (std::size_t p = 0; p < x.extent(1); ++p)
          u[p] = std::sin(x(0, p)) * std::cos(std::numbers::pi * x(1, p));

        return {u, {u.size()}};
      });

    // Create DG functions
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "c0", mesh));

    auto ncells = V_DG->dofmap()->index_map->size_global();
    
    // Define cell functions
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::span<T> rho0_ = rho0->x()->mutable_array();
    
    std::fill(c0_.begin(), c0_.end(), speedOfSoundWater);
    std::fill(rho0_.begin(), rho0_.end(), densityWater);

    // ------------------------------------------------------------------------
    // Stiffness coefficients
    std::vector<T> s_coeffs(c0_.size());
    for (std::size_t i = 0; i < s_coeffs.size(); ++i)
      s_coeffs[i] = - 1.0 / rho0_[i];

    // ------------------------------------------------------------------------
    // Compute spectral stiffness vector
    StiffnessSpectral3D<T, P> stiffness_spectral(V);

    auto stiff = std::make_shared<fem::Function<T>>(V);

    common::Timer stiff_assembly("~ stiffness spectral");
    stiff_assembly.start();
    stiffness_spectral(*u->x(), s_coeffs, *stiff->x());
    stiff_assembly.stop();
    stiff->x()->scatter_rev(std::plus<T>());

    // ------------------------------------------------------------------------
    // List timings
    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);

    if (mpi_rank == 0) {
      std::cout << "Polynomial degree: " << P << "\n";
      std::cout << "Degrees of freedom: " << ndofs << "\n";
      std::cout << "Number of cells: " << ncells << "\n";
    }

  }
  PetscFinalize();
}