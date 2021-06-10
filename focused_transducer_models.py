import numpy as np
from dolfinx.generation import RectangleMesh
from dolfinx import FunctionSpace, Function
from dolfinx.cpp.geometry import select_colliding_cells
from dolfinx.fem import assemble_matrix, assemble_vector
from dolfinx.geometry import BoundingBoxTree, compute_collisions_point
from dolfinx.io import XDMFFile
from ufl import TrialFunction, TestFunction, Measure, inner, grad, dx
from mpi4py import MPI
from petsc4py import PETSc

import runge_kutta_methods as rk
from utils import get_hmin, get_eval_params

class Model1:
    """
    Model that consider only diffraction
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
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


class Model2:
    """
    Model that consider diffraction + absorption
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx
                          + inner(self.dg, self.v)*ds(1))

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

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(self.p0*self.w0**2/self.c0 * np.cos(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
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


class Model3:
    """
    Model for the Westervelt equation, i.e. diffraction + absorption +
    nonlinearity.
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta, beta, rho0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0

        # Define variational formulation
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        self.a = inner(self.u, self.v)*dx \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2) \
            - 2*self.beta/self.rho0/self.c0**2*self.u_n \
            * inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx
                          + inner(self.dg, self.v)*ds(1)) \
            + 2*self.beta/self.rho0/self.c0**2 \
            * inner(self.v_n*self.v_n, self.v)*dx

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

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(self.p0*self.w0**2/self.c0 * np.cos(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        self.solver.setOperators(self.M)

        # Solve
        self.solver.solve(b, result)

        return result


# Read mesh
with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/piston2d.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="piston2d")
    tdim = mesh.topology.dim
    fdim = tdim-1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mt = xdmf.read_meshtags(mesh, name="facets")

# mesh = RectangleMesh(
#     MPI.COMM_WORLD,
#     [np.array([0., 0., 0.]), np.array([1., 1., 0])],
#     [10, 10]
# )

# Choose model
model = "Model 3"
dimension = "2d"

# Set parameters
c0 = 1482  # m/s
f0 = 2e6  # Hz
p0 = 4.3e5  # Pa
delta = 1e-4
beta = 1e-1
rho0 = 1.0
k = 2

# Instantiate model
if model == "Model 1":
    eqn = Model1(mesh, mt, k, c0, f0, p0)
elif model == "Model 2":
    eqn = Model2(mesh, mt, k, c0, f0, p0, delta)
elif model == "Model 3":
    eqn = Model3(mesh, mt, k, c0, f0, p0, delta, beta, rho0)

# Temporal parameters
t = 0.0  # start time
T = 0.08 / c0 + 2.0 / f0 # 0.5e-5  # final time
PETSc.Sys.syncPrint("Final time:", T)
CFL = 0.9
hmin = get_hmin(mesh)
dt = CFL * hmin / (c0 * (2 * k + 1))
nstep = int(T / dt)
PETSc.Sys.syncPrint("Total steps:", nstep)

# RK4
fname = "solution/2d/{}_{}".format(
    model.lower().replace(" ", ""),
    dimension
)
# rk.ode452(eqn.f0, eqn.f1, *eqn.init(), t, T, fname)
# rk.solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4, fname)

npts = 10000
x0 = np.linspace(-0.05, 0.03, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = get_eval_params(mesh, points)
rk.solve2_eval(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4, x, cells,
               "test_eval_model3_p2")
