//
// Westervelt solver for the 2D planewave problem
// - structured mesh
// - first-order Sommerfeld ABC
// ==============================================
// Copyright (C) 2024 Adeeb Arif Kor

#include "Westervelt.hpp"
#include "forms.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // MPI

  }
}