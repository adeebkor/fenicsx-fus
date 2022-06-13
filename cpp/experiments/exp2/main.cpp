//
// Linear solver for the 3D planar problem with inhomogenous domain
// ================================================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "LinearGLLPenetrable.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

using namespace dolfinx;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    std::cout.precision(15); // Set print precision

    // MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Material parameters
    constexpr double speedOfSoundWater = 1500.0;  // (m/s)
    constexpr double speedOfSoundBone = 2800.0;   // (m/s)

    // Source parameters
    constexpr double sourceFrequency = 0.5e6;      // (Hz)
    constexpr double pressureAmplitude = 60000.0;  // (Pa)
    constexpr double period = 1 / sourceFrequency; // (s)

    // Domain parameters
    constexpr double domainLength = 0.12; // (m)

    // FE parameters
    constexpr int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::hexahedron, 1);
    io::XDMFFile xdmf(MPI_COMM_WORLD, "/home/mabm4/rds/hpc-work/big_mesh/mesh.xdmf", "r");
    auto mesh
        = std::make_shared<mesh::Mesh>(xdmf.read_mesh(element, mesh::GhostMode::none, "planar3d"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "planar3d_boundaries"));

    // Get smallest mesh size
    int tdim = mesh->topology().dim();
    int num_cell = mesh->topology().index_map(tdim)->size_local();
    std::vector<int> v(num_cell);
    std::iota(v.begin(), v.end(), 0);
    std::vector<double> hmin = mesh::h(*mesh, v, tdim);
    std::vector<double>::iterator result = std::min_element(hmin.begin(), hmin.end());
    int idx = std::distance(hmin.begin(), result);
    double h = hmin.at(idx);
    double meshSize;
    MPI_Reduce(&h, &meshSize, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "Problem type: planar" << std::endl;
      std::cout << "Polynomial basis degree: " << degreeOfBasis << std::endl;
      std::cout << "Minimum mesh size: " << meshSize << std::endl; 
    }

    // Temporal parameters
    double CFL = 0.64;
    double timeStepSize = CFL * meshSize / (speedOfSoundBone * pow(degreeOfBasis, 2));
    double startTime = 0.0;
    double finalTime = 20 * timeStepSize; // domainLength / speedOfSound + 8.0 / sourceFrequency;
    int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    int numberOfStep = finalTime / timeStepSize + 1;
    if (rank == 0) {
      std::cout << "Number of step per period: " << stepPerPeriod << std::endl;
      std::cout << "dt = " << timeStepSize << std::endl;
    }

    // Model


    // Solve
    
  }
  PetscFinalize();
  return 0;
}