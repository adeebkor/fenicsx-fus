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
        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant=fe)
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


class LinearEquispaced(Wave):
    """
    Solver for linear second order wave equation.

    The model uses an equispaced lattice and Gauss quadrature to compute
    the mass matrix. A direct solver is use to solve Ax=b.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, windowing=True):
        super().__init__(mesh, meshtags, "equispaced", k, c0, freq0, p0)
        self.u = TrialFunction(self.V)
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4

        # Define variational formulation
        self.a = inner(self.u, self.v) * dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2*(- inner(grad(self.u_n), grad(self.v)) * dx
                             + inner(self.g, self.v) * self.ds(1)
                             - 1/self.c0*inner(self.v_n, self.v) * self.ds(2))

        # Linear solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.M)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            else:
                window = 1.0
        else:
            window = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
                            * np.cos(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(b, result)

        return result



