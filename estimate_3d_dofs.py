import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import BoxMesh, FunctionSpace, Function
from dolfinx.cpp.mesh import CellType

# Material properties
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density of medium (kg / m^3)
beta = 3.5  # coefficient of nonlinearity

# Source parameters
f0 = 1E6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)

# Domain parameters
L = 0.25  # domain length (m)

# Physical parameters
lmbda = c0 / f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 4  # degree of basis function

# Mesh parameters
epw = 16  # number of element per wavelength
nw = (L+2*lmbda) / lmbda  # number of waves
n = int(epw * nw + 1)  # total number of elements
h = np.sqrt(3*(L / n)**2)

# Generate mesh
mesh = BoxMesh(
	MPI.COMM_WORLD,
	[np.array([0., 0., 0.,]), np.array([L, L, L])],
	[n, n, n],
	CellType.hexahedron
)

# Define function space
V = FunctionSpace(mesh, ("Lagrange", degree))
dofs = V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms:", dofs)
