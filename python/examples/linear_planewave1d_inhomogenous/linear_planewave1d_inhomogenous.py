#
# .. _linear_planewave1d_inhomogeneoues:
#
# Linear solver for the 1D inhomogeneous media
# ============================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import create_interval, locate_entities_boundary, meshtags
from ufl import inner, dx

from hifusim import LinearInhomogenousGLL
from hifusim.utils import compute_eval_params

# Material parameters
c0 = 1  # speed of sound (m/s)
rho0 = 1  # density of medium (kg / m^3)
ri = 0.5  # refractive index

# Source parameters
f0 = 10  # source frequency (Hz)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)

# FE parameters
degree = 4

# Mesh parameters
epw = 8
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx

# Generate mesh
mesh = create_interval(MPI.COMM_WORLD, nx, [0, L])

# Tag boundaries
tdim = mesh.topology.dim

facets0 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] > L - np.finfo(float).eps)

indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = meshtags(mesh, tdim-1, indices, values[pos])

# Temporal parameters
tstart = 0.0  # simulation start time (s)
tend = L / c0 + 16 / f0  # simulation final time (s)

CFL = 0.9
dt = CFL * h / (c0 * degree**2)

print("Final time:", tend)

# Instantiate model
eqn = LinearInhomogenousGLL(mesh, mt, degree, c0, f0, p0, ri)
eqn.alpha = 4
print("Degree of freedoms: ", eqn.V.dofmap.index_map.size_global)

# Solve
eqn.init()
u_e, _, _, _ = eqn.rk4(tstart, tend, dt)

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

u_eval = u_e.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval)
plt.savefig("u_e.png")
plt.close()
