#
# .. _linear_planewave1d_1_exp:
#
# Linear solver for the 1D planewave problem
# - structured mesh
# - first-order Sommerfeld ABC
# - homogenous medium
# - implicit Newmark (beta = 1/4, gamma = 1/2)
# ==========================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import (create_interval, locate_entities,
                          locate_entities_boundary, meshtags)
from ufl import inner, dx

from fenicsxfus import LinearSpectralNewmark
from fenicsxfus.utils import compute_eval_params

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density (kg / m^3)

# Source parameters
f0 = 0.5e6  # source frequency (Hz)
u0 = 0.04  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
L = 0.12  # domain length (m)

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

# Define DG functions to specify different medium
cells0 = locate_entities(
    mesh, tdim, lambda x: x[0] < L / 2)
cells1 = locate_entities(
    mesh, tdim, lambda x: x[0] >= L / 2 - h)

V_DG = FunctionSpace(mesh, ("DG", 0))
c = Function(V_DG)
c.x.array[:] = c0

rho = Function(V_DG)
rho.x.array[:] = rho0

# Temporal parameters
CFL = 0.5
dt = CFL * h / (c0 * degree * 2)

tstart = 0.0  # simulation start time (s)
tend = L / c0 + 16 / f0  # simulation final time (s)

# Model
model = LinearSpectralNewmark(mesh, mt, degree, c, rho, f0, p0, c0, dt)

# Solve
model.init()
u_e, v_e, w_e, tf = model.newmark(tstart, tend)

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

u_eval = u_e.eval(x, cells).flatten()


# Best approximation
class Analytical:
    """ Analytical solution """

    def __init__(self, c0, f0, p0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.sin(self.w0 * (self.t - x[0]/self.c0)) * \
            np.heaviside(self.t-x[0]/self.c0, 0)

        return val


V_ba = FunctionSpace(mesh, ("Lagrange", degree))
u_ba = Function(V_ba)
u_ba.interpolate(Analytical(c0, f0, p0, tf))

u_ba_eval = u_ba.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval)
plt.plot(x.T[0], u_ba_eval, "--")
plt.savefig("u_e.png")
plt.close()

# L2 error
diff = u_e - u_ba
L2_diff = mesh.comm.allreduce(
    assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = mesh.comm.allreduce(
    assemble_scalar(form(inner(u_ba, u_ba) * dx)), op=MPI.SUM)

L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

print(f"L2 relative error: {L2_error:5.5}")
