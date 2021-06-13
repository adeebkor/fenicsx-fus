import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.io import XDMFFile

from utils import get_hmin, get_eval_params
from models import Westervelt
from runge_kutta_methods import solve2_eval

# Read mesh
with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/domain1d.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="domain1d")
    tdim = mesh.topology.dim
    fdim = tdim-1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mt = xdmf.read_meshtags(mesh, name="facets")


# Physical parameters
c0 = 1481.44  # m/s
mu0 = 1.0016E-3  # Pa s
rho0 = 999.6  # kg/m^3
f0 = 0.1E6  # Hz
w0 = 2 * np.pi * f0  # rad/s
p0 = 5E6  # Pa
delta = 4*mu0/3/rho0  # m^2/s
beta = 10

# FE parameters
degree = 1

# Temporal parameters
Lmax = max(MPI.COMM_WORLD.allgather(max(mesh.geometry.x[:, 0])))
tstart = 0.0  # s
tend = Lmax / c0 + 2.0 / f0  # s
CFL = 0.9
hmin = get_hmin(mesh)
dt = CFL * hmin / (c0 * (2 * degree + 1))
nstep = int(tend / dt)

PETSc.Sys.syncPrint("Final time:", tend)
PETSc.Sys.syncPrint("Number of steps:", nstep)

# Instantiate model
eqn = Westervelt(mesh, mt, degree, c0, f0, p0, delta, beta, rho0)

# Solve
npts = 10000
x0 = np.linspace(0, Lmax, npts)
points = np.zeros((3, npts))
points[0] = x0
idx, x, cells = get_eval_params(mesh, points)

outfilename = "test-model"

solve2_eval(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4,
            npts, idx, x, cells,
            outfilename)
