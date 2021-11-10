import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import RectangleMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, MeshTags
from dolfinx.cpp.mesh import CellType

# Material parameters
c0 = 1.0  # speed of sound (m/s)

# Source parameters
f0 = 10.0  # source frequency (Hz)
w0 = 2.0 * np.pi * f0  # angular frequency (rad/s)

# Domain parameters
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2.0 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 4  # degree of basis function

# Mesh parameters
epw = 4  # number of element per wavelength
nw = L / lmbda  # number of waves in the domain
nx = int(epw * nw + 1)  # total number of element in x-direction

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, L, 0])],
    [nx, nx],
    CellType.quadrilateral)
mesh.name = "rectangle"

h = np.sqrt(2 * (L/nx)**2)

tdim = mesh.topology.dim

facets0 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = locate_entities_boundary(
    mesh, tdim-1,
    lambda x: x[0] > L - np.finfo(float).eps)
indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = MeshTags(mesh, tdim-1, indices, values[pos])
mt.name = "rectangle_edge"

with XDMFFile(
    MPI.COMM_WORLD, "Mesh/rectangle_dolfinx.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    xdmf.write_meshtags(mt)
