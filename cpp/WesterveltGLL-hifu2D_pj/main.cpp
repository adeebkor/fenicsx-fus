#include "WesterveltGLL.hpp"
#include "precompute_jacobian.hpp"

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

    // MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Material parameters
    double speedOfSound = 1486.0;    // (m/s)
    double densityOfMedium = 998.0; // (kg/m^3)
    double coeffOfNonlinearity = 3.5;
    double diffusivityOfSound = 4.33e-6;

    // Source parameters
    double sourceFrequency = 1.0e6;                                                 // (Hz)
    double angularFrequency = 2.0 * M_PI * sourceFrequency;                        // (rad/s)
    // double velocityAmplitude = 1.0;                                                // (m/s)
    double pressureAmplitude = 1.0e6; // (Pa)

    // Domain parameters
    double shockFormationDistance = densityOfMedium * pow(speedOfSound, 3) / coeffOfNonlinearity
                                    / pressureAmplitude / angularFrequency; // (m)
    double domainLength = 0.12;                                              // (m)

    // Physical parameters
    double wavelength = speedOfSound / sourceFrequency; // (m)
    double wavenumber = 2.0 * M_PI / wavelength;        // (m^-1)

    // FE parameters
    int degreeOfBasis = 4;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    io::XDMFFile xdmf(MPI_COMM_WORLD, "../../../mesh/rectangle_dolfinx.xdmf", "r");
    auto mesh
        = std::make_shared<mesh::Mesh>(xdmf.read_mesh(element, mesh::GhostMode::none, "rectangle"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "rectangle_edge"));

    {
      precompute_jacobian(mesh, 6);
      permutation_vector(degreeOfBasis, 6);
      std::getchar();
    }

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
    double CFL = 0.8;
    double timeStepSize = CFL * meshSize / (speedOfSound * pow(degreeOfBasis, 2));
    double startTime = 0.0;
    double finalTime = domainLength / speedOfSound + 2.0 / sourceFrequency;

    // Model
    WesterveltGLL eqn(mesh, mt, degreeOfBasis, speedOfSound, sourceFrequency, pressureAmplitude, diffusivityOfSound, coeffOfNonlinearity, densityOfMedium);

    if (rank == 0){
      std::cout << "Degrees of freedom: " << eqn.V->dofmap()->index_map->size_global() << std::endl;
    }

    // RK solve
    eqn.init();
    eqn.rk4(startTime, finalTime, timeStepSize);
  }
  common::subsystem::finalize_mpi();
  return 0;
}