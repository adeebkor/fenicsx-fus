#
# .. _linear_planewave2d:
#
# Linear solver for the 2D planewave problem
# ==========================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.io import XDMFFile
from dolfinx.mesh import (MeshTags, CellType, create_box,
                          locate_entities_boundary)

from hifusim import Linear

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density of medium (kg / m^3)
beta = 0.01  # coefficient of nonlinearity
delta = 0.001  # diffusivity of sound

# Source parameters
f0 = 1e6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 2*lmbda  # domain length (m)

# FE parameters
degree = 3

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

# Create box mesh
mesh = create_box(
    MPI.COMM_WORLD,
    ((0.0, 0.0, 0.0), (L, L, L)),
    (nx, nx, nx),
    CellType.hexahedron)

# Tag boundaries
facets0 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 0.0))
facets1 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], L))
indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = MeshTags(mesh, 2, indices, values[pos])

# Model
eqn = Linear(mesh, mt, degree, c0, f0, p0)
PETSc.Sys.syncPrint("Degrees of freedom:", eqn.V.dofmap.index_map.size_global)

# Solve
eqn.init()
eqn.rk4(t0, tf, dt)

with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as f:
    f.write_mesh(eqn.mesh)
    f.write_function(eqn.u_n)
