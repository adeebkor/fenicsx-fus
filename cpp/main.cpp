#include "forms.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

int main(int argc, char* argv[])
{
    common::subsystem::init_logging(argc, argv);
    common::subsystem::init_petsc(argc, argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    


}