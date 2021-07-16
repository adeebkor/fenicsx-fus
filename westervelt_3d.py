import time
import numpy as np
from scipy.special import jv
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import BoxMesh, FunctionSpace, Function
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx

from models import WesterveltGLLv
from runge_kutta_methods import solve2

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density of medium (kg / m^3)
beta = 3.5  # coefficient of nonlinearity

# Source parameters
f0 = 5E6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 0.9 * xsh  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 3  # degree of basis function

# Mesh parameters
epw = 4  # number of element per wavelength
nw = L / lmbda  # number of waves
n = int(epw * nw + 1)  # total number of elements
h = np.sqrt(3*(L / n)**2)

PETSc.Sys.syncPrint("Element size:", h)

# Generate mesh
mesh = BoxMesh(
	MPI.COMM_WORLD,
	[np.array([0., 0., 0.,]), np.array([L, L, L])],
	[n, n, n],
	CellType.tetrahedron
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
tend = L / c0 + 2 / f0  # simulation final time (s)
CFL = 0.7
dt = CFL * h / (c0 * (2 * degree + 1))

nstep = int(tend / dt)