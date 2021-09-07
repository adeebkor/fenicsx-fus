import sys

import numpy as np
import matplotlib.pyplot as plt
import gmsh
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp
from dolfinx.cpp.io import perm_gmsh
from dolfinx.io import (extract_gmsh_geometry, ufl_mesh_from_gmsh,
                        extract_gmsh_topology_and_markers)

from dolfinx.mesh import create_mesh, locate_entities_boundary, MeshTags

from models import LinearEquispaced, LinearGLL
from rk import solve_ibvp

from utils import get_eval_params


def build_mesh2d(h, nx, L, transfinite=False):
    """
        Build a 2D rectangular mesh with [0, L] x [0, L].
        If transfinite is true, a structured mesh is build.
        h: target mesh size
        nx: number of nodes on the boundaries
        L: length of the domain
    """

    gmsh.initialize()
    model = gmsh.model()
    model.add("Rectangle")

    lc = h

    model.geo.addPoint(0, 0, 0, lc, 1)
    model.geo.addPoint(L, 0, 0, lc, 2)
    model.geo.addPoint(L, L, 0, lc, 3)
    model.geo.addPoint(0, L, 0, lc, 4)

    model.geo.addLine(1, 2, 1)
    model.geo.addLine(2, 3, 2)
    model.geo.addLine(3, 4, 3)
    model.geo.addLine(4, 1, 4)

    if transfinite:
        model.geo.mesh.setTransfiniteCurve(1, nx+1, "Progression", 1.03)
        model.geo.mesh.setTransfiniteCurve(2, nx+1)
        model.geo.mesh.setTransfiniteCurve(3, nx+1, "Progression", 1.03)
        model.geo.mesh.setTransfiniteCurve(4, nx+1)

    model.geo.addCurveLoop([1, 2, 3, 4], 1)
    model.geo.addPlaneSurface([1], 1)

    model.geo.synchronize()
    model.addPhysicalGroup(2, [1])
    model.setPhysicalName(2, 1, "Domain")

    if transfinite:
        model.geo.mesh.setTransfiniteSurface(1)

    model.geo.mesh.setRecombine(2, 1)

    model.geo.synchronize()
    model.mesh.generate(2)

    x = extract_gmsh_geometry(model, model_name="Rectangle")
    gmsh_cell_id = model.mesh.getElementType("quadrangle", 1)
    topologies = extract_gmsh_topology_and_markers(model, "Rectangle")
    cells = topologies[gmsh_cell_id]["topology"]
    gmsh_quad4 = perm_gmsh(cpp.mesh.CellType.quadrilateral, 4)
    cells = cells[:, gmsh_quad4]
    domain = ufl_mesh_from_gmsh(gmsh_cell_id, 2)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :2], domain)
    mesh.name = "rectangle"

    return mesh


# RK setting
rk_type = "Heun3"

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

# Generate mesh
mesh = build_mesh2d(h, nx, L)

# Tag boundaries
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
hmin = min(cpp.mesh.h(mesh, tdim, range(num_cells)))

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
dt = CFL * hmin / (c0 * degree**2)

PETSc.Sys.syncPrint("Final time:", tend)

# Instantiate model (Equispaced)
eqn_eq = LinearEquispaced(mesh, mt, degree, c0, f0, p0)
dof_eq = eqn_eq.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms: ", dof_eq)

# Solve (Equispaced)
u_eq, tf_eq, nstep_eq = solve_ibvp(eqn_eq.f0, eqn_eq.f1, *eqn_eq.init(), dt,
                                   tspan, rk_type)
u_eq.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
PETSc.Sys.syncPrint("tf:", tf_eq)
PETSc.Sys.syncPrint("Number of steps:", nstep_eq)

# Instantiate model (GLL)
eqn_gll = LinearGLL(mesh, mt, degree, c0, f0, p0, vectorised=True)
dof_gll = eqn_gll.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms: ", dof_gll)

# Solve (GLL)
u_gll, tf_gll, nstep_gll = solve_ibvp(eqn_gll.f0, eqn_gll.f1, *eqn_gll.init(),
                                      dt, tspan, rk_type)
u_gll.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
PETSc.Sys.syncPrint("tf:", tf_gll)
PETSc.Sys.syncPrint("Number of steps:", nstep_gll)

# Plot solution
npts = degree * (nx+1)
x0, x1 = np.mgrid[0:L:1j*npts, 0:L:1j*npts]
points = np.zeros((3, npts*npts))
points[0, :] = x0.flatten()
points[1, :] = x1.flatten()
idx, x, cells = get_eval_params(mesh, points)

u_eval_eq = u_eq.eval(x, cells).reshape(npts, npts, order='F')
u_eval_gll = u_gll.eval(x, cells).reshape(npts, npts, order='F')

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(u_eval_eq, aspect='equal', origin='lower',
             extent=(0, L, 0, L), interpolation='bicubic', cmap='RdBu')
ax[0].set_title("Equispaced")
ax[1].imshow(u_eval_gll, aspect='equal', origin='lower',
             extent=(0, L, 0, L), interpolation='bicubic', cmap='RdBu')
ax[1].set_title("GLL")
plt.savefig("linear_2d_p{}_epw{}.png".format(degree, epw), bbox_inches='tight')
