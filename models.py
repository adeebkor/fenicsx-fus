import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import FunctionSpace, Function
from dolfinx.fem import assemble_matrix, assemble_vector
from ufl import (FiniteElement, TrialFunction, TestFunction, Measure, inner,
                 grad, dx)

class Wave:
    """
    Base class for wave models.
    """

    def __init__(self, mesh, meshtags, fe, k, c0, freq0, p0):
        FE = FiniteElement("Lagrange", mesh.cell_type(), k, variant=fe)
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        self.ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        u.x.array[:] = 0.0
        v.x.array[:] = 0.0

        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        
        return v.copy(result=result)


        