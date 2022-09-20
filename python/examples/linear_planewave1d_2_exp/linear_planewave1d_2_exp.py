#
# .. _linear_planewave1d_2:
#
# Linear solver for the 1D planewave problem
# - structured mesh
# - first-order Sommerfeld
# - two different medium (x < 0.5, x >= 0.5)
# ============================================================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import (create_interval, locate_entities,
                          locate_entities_boundary, meshtags)

from hifusim import LinearGLLExplicit
from hifusim.utils import compute_eval_params

# Material parameters
c0 = 1500  # medium 1 speed of sound (m/s)
c1 = 2800  # medium 2 speed of sound (m/s)
rho0 = 1000  # medium 1 density (kg / m^3)
rho1 = 1850  # medium 2 density (kg / m^3)

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

# RK parameter
rk = 4

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
c.x.array[cells1] = c1

rho = Function(V_DG)
rho.x.array[:] = rho0
rho.x.array[cells1] = rho1

# Temporal parameters
CFL = 0.9
dt = CFL * h / (c0 * degree**2)

tstart = 0.0  # simulation start time (s)
tend = L / c0 + 16 / f0  # simulation final time (s)

# Model
model = LinearGLLExplicit(mesh, mt, degree, c, rho, f0, p0, c0, rk, dt)

# Solve
model.init()
u_e, _, tf, = model.rk(tstart, tend)

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

u_eval = u_e.eval(x, cells).flatten()


# Best approximation
class Wave:
    """ Analytical solution """

    def __init__(self, c1, c2, rho1, rho2, f, p, t):
        self.r1 = c1*rho1
        self.r2 = c2*rho2

        self.ratio = self.r2 / self.r1

        self.R = (self.ratio - 1) / (self.ratio + 1)
        self.T = 2 * self.ratio / (self.ratio + 1)

        self.f = f
        self.w = 2 * np.pi * f
        self.p = p
        self.k1 = self.w / c1
        self.k2 = self.w / c2

        self.t = t

    def field(self, x):
        x0 = x[0] + 0.j  # need to plus 0.j because piecewise return same type
        val = np.piecewise(
            x0, [x0 < L / 2, x0 >= L / 2],
            [lambda x: self.R * self.p * np.exp(
                1j * (self.w * self.t - self.k1 * (x - L / 2))),
             lambda x: self.T * self.p * np.exp(
                1j * (self.w * self.t - self.k2 * (x - L / 2)))])

        return val.imag


V_ba = FunctionSpace(mesh, ("Lagrange", degree))
u_ba = Function(V_ba)
wave = Wave(c0, c1, rho0, rho1, f0, p0, tf)
u_ba.interpolate(wave.field)

u_ba_eval = u_ba.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval)
plt.plot(x.T[0], u_ba_eval, "--")
plt.savefig("u_e.png")
plt.close()
