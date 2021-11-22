import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.io import XDMFFile

from LinearGLL import LinearGLL

# Material parameters
c0 = 1  # speed of sound (m/s)
rho0 = 1  # density of medium (kg / m^3)
beta = 0.01  # coefficient of nonlinearity
delta = 0.001  # diffusivity of sound

# Source parameters
f0 = 10  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 6

# Mesh parameters
epw = 4  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = np.sqrt(2 * (L/nx)**2)

# Temporal parameters
CFL = 0.8
dt = CFL * h / c0 / degree**2
t0 = 0.0
tf = L/c0 + 10.0/f0

# Read mesh and meshtags

with XDMFFile(MPI.COMM_WORLD,
              "../../mesh/rectangle_dolfinx.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="rectangle")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt = xdmf.read_meshtags(mesh, "rectangle_edge")

# Model
eqn = LinearGLL(mesh, mt, degree, c0, f0, p0)
PETSc.Sys.syncPrint("Degrees of freedom:", eqn.V.dofmap.index_map.size_global)

# Solve
eqn.init()
eqn.rk4(t0, tf, dt)
