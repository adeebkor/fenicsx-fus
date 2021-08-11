import sys
import json

import numpy as np
from scipy.integrate import RK45
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import IntervalMesh, FunctionSpace, Function
from dolfinx.fem import assemble_scalar
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx, grad

from models import LinearEquispaced


class LinearSciPy(LinearEquispaced):
    """
    Model of the linear second order wave equation.

    This model creates a mass matrix using equispaced points and Gauss
    quadrature. It then uses a direct solver to solve for the mass matrix.

    Uses SciPy ODE solver for time stepping.

    I've only tested this in serial
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0, windowing=True):
        super().__init__(mesh, meshtags, k, c0, freq0, p0, windowing)

        # Define helper vector
        self.un = Function(self.V)
        self.vn = Function(self.V)

    def f(self, t: float, y: np.array):
        self.un.x.array[:] = y[:y.shape[0]//2]
        self.vn.x.array[:] = y[y.shape[0]//2:]
        vdot = self.f1(t, self.un.vector, self.vn.vector, self.un.vector)

        return np.concatenate([self.vn.x.array[:], vdot.array[:]])


# Material properties
c0 = 1.0  # speed of sound (m/s)
rho0 = 1.0  # density of medium (kg / m^3)

# Source parameters
f0 = 10.0  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = int(sys.argv[1])  # degree of basis function

# Mesh parameters
epw = int(sys.argv[2])  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx

PETSc.Sys.syncPrint("Element size:", h)

# Generate mesh
mesh = IntervalMesh(
    MPI.COMM_WORLD,
    nx,
    [0, L]
)

# Tag boundaries
tdim = mesh.topology.dim

facets0 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] > L - np.finfo(float).eps)

indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = MeshTags(mesh, tdim-1, indices, values[pos])

# Temporal parameters
tstart = 0.0  # simulation start time (s)
tend = L / c0 + 16 / f0  # simulation final time (s)

PETSc.Sys.syncPrint("Final time:", tend)

# Instantiate model
eqn = LinearSciPy(mesh, mt, degree, c0, f0, p0)
dof = eqn.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms: ", dof)

# Solve
tol = float(sys.argv[3])
y0 = np.zeros((2*dof,))
problem = RK45(eqn.f, tstart, y0, tend, rtol=tol, atol=tol)

step = 0
while problem.t < tend:
    problem.step()
    step += 1
    if step % 100 == 0:
        print(step, problem.t)

# Get solution at tend
u = Function(eqn.V)
u.x.array[:] = problem.y[:dof]
tf = problem.t


# Calculate error
class Analytical:
    def __init__(self, c0, f0, p0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.sin(self.w0 * (self.t - x[0]/self.c0)) * \
              np.heaviside(self.t-x[0]/self.c0, 0)

        return val


u_ba = Function(eqn.V)
u_ba.interpolate(Analytical(c0, f0, p0, tf))

V_e = FunctionSpace(mesh, ("Lagrange", degree+3))
u_e = Function(V_e)
u_e.interpolate(Analytical(c0, f0, p0, tf))

# L2 error
diff_fe = u - u_e
L2_diff_fe = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(diff_fe, diff_fe) * dx), op=MPI.SUM)

diff_ba = u_ba - u_e
L2_diff_ba = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(diff_ba, diff_ba) * dx), op=MPI.SUM)

L2_exact = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(u_e, u_e) * dx), op=MPI.SUM)

L2_error_fe = abs(np.sqrt(L2_diff_fe) / np.sqrt(L2_exact))
print("Relative L2 error of FEM solution:", L2_error_fe)

L2_error_ba = abs(np.sqrt(L2_diff_ba) / np.sqrt(L2_exact))
print("Relative L2 error of BA solution:", L2_error_ba)

# H1 error
H1_diff_fe = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(grad(diff_fe), grad(diff_fe)) * dx), op=MPI.SUM)

H1_diff_ba = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(grad(diff_ba), grad(diff_ba)) * dx), op=MPI.SUM)

H1_exact = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(grad(u_e), grad(u_e)) * dx), op=MPI.SUM)

H1_error_fe = abs(np.sqrt(H1_diff_fe) / np.sqrt(H1_exact))
print("Relative H1 error of FEM solution:", H1_error_fe)

H1_error_ba = abs(np.sqrt(H1_diff_ba) / np.sqrt(H1_exact))
print("Relative H1 error of BA solution:", H1_error_ba)

# if MPI.COMM_WORLD.rank == 0:
#     with open("data/simulation_data.json") as file:
#         data = json.load(file)

#     data["Type"].append("GLLv-SciPy")
#     data["Dimension"].append(1)
#     data["Linear solver"].append("Diagonal")
#     data["RK level"].append("RK45")
#     data["Window"].append(True)
#     data["Tolerance"].append(tol)
#     data["Time step"].append('nan')
#     data["Total step"].append(step)
#     data["Final time"].append(tf)
#     data["Basis degree"].append(degree)
#     data["Number of element per wavelength"].append(epw)
#     data["Degrees of freedom"].append(dof)
#     data["Element size"].append(h)
#     data["L2 error (FE)"].append(L2_error_fe)
#     data["L2 error (BA)"].append(L2_error_ba)
#     data["H1 error (FE)"].append(H1_error_fe)
#     data["H1 error (BA)"].append(H1_error_ba)

#     with open("data/simulation_data.json", "w") as file:
#         json.dump(data, file)
