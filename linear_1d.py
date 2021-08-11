import sys

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import IntervalMesh
from dolfinx.mesh import locate_entities_boundary, MeshTags

from models import LinearEquispaced, LinearGLL
from rk import solve_ibvp
from utils import get_eval_params

# Settings
linear_solver = "Diagonal"
rk_order = "Heun3"

# Material parameters
c0 = 1  # speed of sound (m/s)
rho0 = 1  # density of medium (kg / m^3)

# Source parameters
f0 = 10  # source frequency (Hz)
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
tspan = [tstart, tend]

CFL = float(sys.argv[3])
dt = CFL * h / (c0 * degree**2)

PETSc.Sys.syncPrint("Final time:", tend)

# Instantiate model (Equispaced)
eqn_eq = LinearEquispaced(mesh, mt, degree, c0, f0, p0)
dof_eq = eqn_eq.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms: ", dof_eq)

# Solve (Equispaced)
u_eq, tf_eq, nstep_eq = solve_ibvp(eqn_eq.f0, eqn_eq.f1, *eqn_eq.init(), dt,
                                   tspan, rk_order)
u_eq.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
PETSc.Sys.syncPrint("tf:", tf_eq)
PETSc.Sys.syncPrint("Number of steps:", nstep_eq)

# Instantiate model (GLL)
eqn_gll = LinearGLL(mesh, mt, degree, c0, f0, p0)
dof_gll = eqn_gll.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms: ", dof_gll)

# Solve (GLL)
u_gll, tf_gll, nstep_gll = solve_ibvp(eqn_gll.f0, eqn_gll.f1, *eqn_gll.init(),
                                      dt, tspan, rk_order)
u_gll.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
PETSc.Sys.syncPrint("tf:", tf_gll)
PETSc.Sys.syncPrint("Number of steps:", nstep_gll)

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
idx, x, cells = get_eval_params(mesh, points)

u_eval_eq = u_eq.eval(x, cells).flatten()
u_eval_gll = u_gll.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval_eq, x.T[0], u_eval_gll, '--')
plt.legend(["Equispaced", "GLL"], loc='upper left')
plt.savefig("linear_1d_p{}_epw{}.png".format(degree, epw))
