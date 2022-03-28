//
// Linear solver (optimize) for the 3D planar problem
// ==================================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "LinearGLLOpt.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    std::cout.precision(15); // Set print precision

    // MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Material parameters
    double speedOfSound = 1500.0;    // (m/s)
    double densityOfMedium = 1000.0; // (kg/m^3)
    double coeffOfNonlinearity = 3.5;
    double diffusivityOfSound = 4.33e-6;

    // Source parameters
    double sourceFrequency = 0.5e6;                           // (Hz)
    double angularFrequency = 2.0 * M_PI * sourceFrequency; // (rad/s)
    double pressureAmplitude = 60000;                     // (Pa)
    double period = 1 / sourceFrequency;                    // (s)

    // Domain parameters
    double shockFormationDistance = densityOfMedium * pow(speedOfSound, 3) / coeffOfNonlinearity
                                    / pressureAmplitude / angularFrequency; // (m)
    double domainLength = 0.1;                                              // (m)

    // Physical parameters
    double wavelength = speedOfSound / sourceFrequency; // (m)
    double wavenumber = 2.0 * M_PI / wavelength;        // (m^-1)

    // FE parameters
    int degreeOfBasis = 4;
    double numberOfWaves = domainLength / wavelength;

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

    // Temporal parameters
    double CFL = 0.5;
    double timeStepSize = CFL * meshSize / (speedOfSound * pow(degreeOfBasis, 2));
    double startTime = 0.0;
    double finalTime = domainLength / speedOfSound + 8.0 / sourceFrequency;
    int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    if (rank == 0){
        std::cout << "Number of step per period: " << stepPerPeriod << std::endl;
        std::cout << "dt = " << timeStepSize << std::endl;
    }

    int nstep = (finalTime - startTime) / timeStepSize + 1;

    // Model
    LinearGLLOpt eqn(mesh, mt, degreeOfBasis, speedOfSound, sourceFrequency, pressureAmplitude);

    if (rank == 0) {
      std::cout << "Number of steps: " << nstep << std::endl;
      std::cout << "Degrees of freedom: " << eqn.V->dofmap()->index_map->size_global() << std::endl;
    }

    // Solve
    eqn.init();

    common::Timer tsolve("Solve time");

    tsolve.start();
    eqn.rk4(startTime, finalTime, timeStepSize);
    tsolve.stop();

    if (rank == 0){
      std::cout << "Solve time: " << tsolve.elapsed()[0] << std::endl;
    }

  }
  common::subsystem::finalize_mpi();
  return 0;
}
