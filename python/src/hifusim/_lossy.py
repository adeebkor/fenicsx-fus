import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl_wrapper
from dolfinx.fem import FunctionSpace, Function, form
from dolfinx.petsc import assemble_matrix, assemble_vector
from dolfinx.io import VTKFile
from dolfinx.mesh import locate_entities
from ufl import TestFunction, TrialFunction, Measure, inner, grad, dx


class LossyGLL:
    pass
