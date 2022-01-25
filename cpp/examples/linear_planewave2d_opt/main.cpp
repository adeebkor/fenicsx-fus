#include "LinearGLLOpt.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <iostream>

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);
  {
    std::cout.precision(15); // Set print precision

    // Material parameters
    double speedOfSound = 1.0;    // (m/s)
    double densityOfMedium = 1.0; // (kg/m^3)
    double coeffOfNonlinearity = 0.01;
    double diffusivityOfSound = 0.001;

    // Source parameters
    double sourceFrequency = 10.0;                                                 // (Hz)
    double angularFrequency = 2.0 * M_PI * sourceFrequency;                        // (rad/s)
    double velocityAmplitude = 1.0;                                                // (m/s)
    double pressureAmplitude = densityOfMedium * speedOfSound * velocityAmplitude; // (Pa)

    // Domain parameters
    double shockFormationDistance = densityOfMedium * pow(speedOfSound, 3) / coeffOfNonlinearity
                                    / pressureAmplitude / angularFrequency; // (m)
    double domainLength = 1.0;                                              // (m)

    // Physical parameters
    double wavelength = speedOfSound / sourceFrequency; // (m)
    double wavenumber = 2.0 * M_PI / wavelength;        // (m^-1)

    // FE parameters
    int degreeOfBasis = 4;

    // Mesh parameters
    int elementPerWavelength = 4;
    double numberOfWaves = domainLength / wavelength;
    int numberOfElement = elementPerWavelength * numberOfWaves + 1;
    double meshSize = sqrt(2 * pow(domainLength / numberOfElement, 2));

    // Temporal parameters
    double CFL = 0.8;
    double timeStepSize = CFL * meshSize / (speedOfSound * pow(degreeOfBasis, 2));
    double startTime = 0.0;
    double finalTime = domainLength / speedOfSound + 10.0 / sourceFrequency;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    io::XDMFFile xdmf(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh
        = std::make_shared<mesh::Mesh>(xdmf.read_mesh(element, mesh::GhostMode::none, "mesh"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "edges"));

    // Model
    LinearGLLOpt eqn(mesh, mt, degreeOfBasis, speedOfSound, sourceFrequency, pressureAmplitude);

    std::cout << "Degrees of freedom: " << eqn.V->dofmap()->index_map->size_global() << std::endl;

    // RK solve
    eqn.init();
    eqn.rk4(startTime, finalTime, timeStepSize);
  }
  common::subsystem::finalize_mpi();
  return 0;
}