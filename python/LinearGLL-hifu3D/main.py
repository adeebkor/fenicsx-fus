import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp
from dolfinx.io import XDMFFile

from LinearGLL import LinearGLL

# Material parameters
c0 = 1486  # speed of sound (m/s)
rho0 = 998  # density of medium (kg / m^3)
beta = 3.5  # coefficient of nonlinearity
delta = 4.33e-6  # diffusivity of sound

# Source parameters
f0 = 1e6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
# u0 = 1  # velocity amplitude (m / s)
p0 = 0.75e6  # pressure amplitude (Pa)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 0.12  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 6

# Read mesh and meshtags
with XDMFFile(MPI.COMM_WORLD, "../../mesh/hifu_mesh_3d.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="hifu")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt = xdmf.read_meshtags(mesh, "hifu_surface")

# Mesh parameters
tdim = mesh.topology.dim
num_cells = num_cells = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh, tdim, range(num_cells)).min()])
h = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, h, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(h, root=0)

# Temporal parameters
CFL = 0.45
dt = CFL * h / c0 / degree**2
t0 = 0.0
tf = L/c0 + 4.0/f0

# Model
eqn = LinearGLL(mesh, mt, degree, c0, f0, p0)

PETSc.Sys.syncPrint("Degrees of freedom:", eqn.V.dofmap.index_map.size_global)

# Solve
eqn.init()
eqn.rk4(t0, tf, dt)
