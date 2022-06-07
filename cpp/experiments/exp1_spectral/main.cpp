//
// Linear solver for the 3D planar problem (spectral)
// ==================================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "LinearGLLSpectral.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    std::cout.precision(15); // Set print precision

    // MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Material parameters
    double speedOfSound = 1500.0; // (m/s)

    // Source parameters
    double sourceFrequency = 0.5e6;      // (Hz)
    double pressureAmplitude = 60000;    // (Pa)
    double period = 1 / sourceFrequency; // (s)

    // Domain parameters
    double domainLength = 0.12; // (m)

    // FE parameters
    const int degreeOfBasis = 4;

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
    double CFL = 0.62;
    double timeStepSize = CFL * meshSize / (speedOfSound * pow(degreeOfBasis, 2));
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
    common::Timer instantiation("~ Instantiate model");
    instantiation.start();
    auto eqn = LinearGLLSpectral<degreeOfBasis>(mesh, mt, speedOfSound, sourceFrequency,
                                                pressureAmplitude);
    instantiation.stop();

    if (rank == 0) {
      std::cout << "CFL number: " << CFL << std::endl;
      std::cout << "Number of steps: " << numberOfStep << std::endl;
      std::cout << "Degrees of freedom: " << eqn.num_dofs() << std::endl;
    }

    // Solve
    eqn.init();

    common::Timer rksolve("~ RK solve time");
    rksolve.start();
    eqn.rk4(startTime, finalTime, timeStepSize);
    rksolve.stop();

    if (rank == 0) {
      std::cout << "Solve time: " << rksolve.elapsed()[0] << std::endl;
      std::cout << "Time per step: " << rksolve.elapsed()[0] / numberOfStep << std::endl;
    }

    list_timings(MPI_COMM_WORLD, {TimingType::wall}, Table::Reduction::min);
  }
  PetscFinalize();
  return 0;
}