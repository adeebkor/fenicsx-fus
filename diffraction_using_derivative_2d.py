import numpy as np
from dolfinx import FunctionSpace, Function
from dolfinx.fem import (assemble_matrix, assemble_vector, apply_lifting,
                         locate_dofs_topological, LinearProblem)
from dolfinx.io import XDMFFile
from ufl import (TrialFunction, TestFunction, Measure, inner, grad, dx,
                 Circumradius)
from mpi4py import MPI
from petsc4py import PETSc

import runge_kutta_methods as rk


class DiffractionProblem:
    def __init__(self, mesh, meshtag, k):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Set boundary condition
        tdim = mesh.topology.dim
        fdim = tdim-1
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = 1

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx \
                             + inner(self.g, self.v)*ds(1) \
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
    
    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) -> PETSc.Vec:
        """For du/dt = f(t, u, v), return f"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) -> PETSc.Vec:
        """For dv/dt = f(t, u, v), return f"""

        # Update boundary condition
        p0 = 10
        f0 = 100
        w0 = 2 * np.pi * f0
        with self.g.vector.localForm() as g_local:
            g_local.set(p0*w0/self.c0 * np.cos(w0 * t))

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


# # Read mesh
# with XDMFFile(MPI.COMM_WORLD, "rectangle.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name="rectangle")
#     tdim = mesh.topology.dim
#     fdim = tdim-1
#     mesh.topology.create_connectivity(fdim, tdim)
#     mesh.topology.create_connectivity(fdim, 0)
#     mt = xdmf.read_meshtags(mesh, name="edges")

# Read mesh
# with XDMFFile(MPI.COMM_WORLD, "piston2d.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name="piston2d")
#     tdim = mesh.topology.dim
#     fdim = tdim-1
#     mesh.topology.create_connectivity(fdim, tdim)
#     mesh.topology.create_connectivity(fdim, 0)
#     mt = xdmf.read_meshtags(mesh, name="edges")

# Read mesh
with XDMFFile(MPI.COMM_WORLD, "piston2d_new.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="piston2d_new")
    tdim = mesh.topology.dim
    fdim = tdim-1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mt = xdmf.read_meshtags(mesh, name="edges")

# Create model
k = 1
eqn = DiffractionProblem(mesh, mt, k)

# Temporal parameters
t = 0  # start time
T = 0.1  # final time

# RK4
dt = 0.002
num_steps = int(T / dt)
fname = "diffraction_using_derivative_2d_cos"
# tstart = time.time()
# rk.solve2(eqn.f0, eqn.f1, *eqn.init(), dt=dt, num_steps=num_steps, rk_order=4, filename=fname)
rk.ode452(eqn.f0, eqn.f1, *eqn.init(), t, T, fname)
# telapsed = time.time() - tstart
# print("Solve time: ", telapsed)