import numpy as np
from dolfinx import IntervalMesh
from dolfinx.mesh import locate_entities_boundary, MeshTags
from dolfinx.io import XDMFFile
from mpi4py import MPI

# Domain parameters
L = 0.1 * 4.5  # domain length (m)

# Physical parameters
c0 = 1481.44  # speed of sound (m/s)
mu0 = 1.0016E-3  # viscosity of medium () 
rho0 = 999.6  # density of medium (kg/m^3)
f0 = 0.1E6  # source frequency (Hz)
delta = 4*mu0/3/rho0  # diffusivity of sound ()
beta = 10

p0 = 5E6  # pressure amplitude (Pa)
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 1

# Mesh parameters
epw = 60
nw = L / lmbda
nx = int(epw * nw)

# Generate mesh
mesh = IntervalMesh(
    MPI.COMM_WORLD,
    nx,
    [0, L]
)
mesh.name = "domain1d"

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
mt.name = "facets"

with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/domain1d.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='{}']/Geometry".format(mesh.name))