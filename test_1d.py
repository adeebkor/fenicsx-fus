import numpy as np
from dolfinx import FunctionSpace, Function, IntervalMesh
from dolfinx.fem import assemble_matrix, assemble_vector
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import TrialFunction, TestFunction, Measure, inner, grad, dx
from mpi4py import MPI
from petsc4py import PETSc

import runge_kutta_methods as rk

class Model1:
    """
    Model that consider only diffraction
    """

    def __init__(self, length, nx, degree, c0, freq0, p0):
        # Generate mesh
        mesh = IntervalMesh(
            MPI.COMM_WORLD,
            nx,
            [0, length]
        )

        # Locate boundary facets
        tdim = mesh.topology.dim

        facet0 = locate_entities_boundary(mesh, tdim-1,
                                          lambda x: x[0] < \
                                          np.finfo(float).eps)
        facet1 = locate_entities_boundary(mesh, tdim-1,
                                          lambda x: x[0] > \
                                          1 - np.finfo(float).eps)
        indices, pos = np.unique(np.hstack((facet0, facet1)),
                                 return_index=True)
        values = np.hstack((np.full(facet0.shape, 1, np.intc),
                            np.full(facet1.shape, 2, np.intc)))
        marker = MeshTags(mesh, tdim-1, indices, values[pos])
        ds = Measure('ds', subdomain_data=marker, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Define basis function
        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(- inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2))

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

            # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.sin(self.w0 * t))

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


# Choose model
model = "Model 1"

# Domain parameters
L = 0.0205 * 3

# Physical parameters
c0 = 1500
u0 = 1
rho0 = 1000
f0 = 5E6
delta = 1e-4
beta = 3.5

p0 = rho0*c0*u0
lmbda = c0/f0
k = 2 * np.pi / lmbda

# FE parameters
epw = 100
nw = L / lmbda
nx = int(epw * nw)
degree = 1

# Instantiate model
if model == "Model 1":
    eqn = Model1(L, nx, degree, c0, f0, p0)

# Temporal parameters
t = 0.0
T = L / c0 + 2.0 / f0
CFL = 0.9
dt = CFL * lmbda / epw / (c0 * (2 * degree + 1))
nstep = int(T / dt)
print("Final time:", T)
print("Total step:", nstep)

# RK4
fname = "{}-1d".format(model.lower().replace(" ", ""))

rk.solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4, fname)