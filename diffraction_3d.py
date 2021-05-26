import numpy as np
from dolfinx import DirichletBC, FunctionSpace, Function
from dolfinx.fem import (assemble_matrix, assemble_vector, apply_lifting,
                         locate_dofs_topological, set_bc, LinearProblem)
from dolfinx.io import XDMFFile
from ufl import (TrialFunction, TestFunction, Measure, inner, grad, dx,
                 Circumradius)
from mpi4py import MPI
from petsc4py import PETSc

import runge_kutta_methods as rk

# Define boundary condition class
class Source:
    def __init__(self, t, c0):
        self.t = t
        self.c0 = c0

    def __call__(self, x):
        p0 = 0.7E6
        f0 = 1.7E6
        w0 = 2 * np.pi * f0
        c0 = self.c0
        F = 0.05
        return p0*np.sin(w0*(self.t+(x[0]**2+x[1]**2)/2/c0/F))


class DiffractionProblem:
    def __init__(self, mesh, meshtag, k):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)

        # Set boundary condition
        tdim = mesh.topology.dim
        fdim = tdim - 1
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = 1482

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx + \
                             1/self.c0*inner(self.v_n, self.v)*ds(2))

        # Define source
        self.u_source = Function(self.V)
        self.source = Source(0.0, self.c0)

        # Get source dofs
        self.source_facets = meshtag.indices[meshtag.values==1]
        self.source_dofs = locate_dofs_topological(self.V, fdim,
                                                   self.source_facets)
        
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

        # Update source
        self.source.t = t
        self.u_source.interpolate(self.source)

        # Update boundary condition
        bcs = [DirichletBC(self.u_source, self.source_dofs)]

        # Update fields that f depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        
        # Assemble RHS
        b = assemble_vector(self.L)
        apply_lifting(b, [self.a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        set_bc(b, bcs)

        # Solve
        self.solver.solve(b, result)

        return result


def hmin(mesh):
    cr = Circumradius(mesh)
    V = FunctionSpace(mesh, ("DG", 0))
    u, v = TrialFunction(V), TestFunction(V)
    a = u*v*dx
    L = cr*v*dx
    lp = LinearProblem(a, L)
    r = lp.solve()
    min_distance = MPI.COMM_WORLD.allreduce(min(r.vector.array), op=MPI.MIN)
    return min_distance


# Read mesh
with XDMFFile(MPI.COMM_WORLD, "piston.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="piston")
    tdim = mesh.topology.dim
    fdim = tdim-1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mt = xdmf.read_meshtags(mesh, name="surfaces")

# Create model
eqn = DiffractionProblem(mesh, mt, 2)

# Temporal parameters
# h = hmin(mesh)
# CFL = 0.9
dt = 2E-9
tstart = 0.0
tend = 0.5E-6
num_steps = int(tend/dt)
print("Step:", num_steps)

# Solve
fname = "test"
# rk.ode452(eqn.f0, eqn.f1, *eqn.init(), tstart, tend, fname)
rk.solve2(eqn.f0, eqn.f1, *eqn.init(), dt=dt, num_steps=num_steps, rk_order=4, filename=fname)