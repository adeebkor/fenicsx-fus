#
# .. _mendousse:
#
# Westervelt solver for a 1D plane wave problem
# - Objective:
#   To compare with mendousse solution
# =============================================
# Copyright (C) 2023 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpi4py import MPI

from dolfinx.common import Timer
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import (create_interval, locate_entities,
                          locate_entities_boundary, meshtags)

from hifusim import WesterveltSpectralExplicit
from hifusim.utils import compute_eval_params, compute_diffusivity_of_sound

# Source parameters
f0 = 1e6  # source frequency (Hz)
w0 = 2 * np.pi * f0
p0 = 5e6  # pressure amplitude (Pa)

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density (kg / m^3)
beta0 = 4.8  # nonlinearity coefficient
alphadB = 25
alphaNp = alphadB / 20 * np.log(10)
delta0 = compute_diffusivity_of_sound(
    w0, c0, alphadB)

# Domain parameters
sigma = np.array([0.1, 0.5, 1.0, 1.5, 3.0])
sensor = sigma * rho0 * c0**3 / beta0 / p0 / 2 / np.pi / f0
xsh = rho0*c0**3/beta0/p0/w0  # shock formation distance (m)
print(f"Sensor location {sensor}, Shock formation distance: {xsh}")
L = 0.09

# Physical parameters
lmbda = c0/f0  # wavelength (m)

# FE parameters
degree = 6

# RK parameter
rk = 4

# Mesh parameters
epw = 10
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

delta = Function(V_DG)
delta.x.array[:] = delta0

beta = Function(V_DG)
beta.x.array[:] = beta0

# Temporal parameters
CFL = 1.0
dt = CFL * h / (c0 * degree**2)

tstart = 0.0  # simulation start time (s)
tend = L / c0 + 8 / f0  # simulation final time (s)
print("Time step size:", dt)
print("Start time:", 0.09 / c0)
numStepPerPeriod = int(1.0 / f0 / dt)
print("Final time:", 0.09 / c0 + numStepPerPeriod*dt)

# Model
model = WesterveltSpectralExplicit(
    mesh, mt, degree, c, rho, delta, beta, f0, p0, c0, rk, dt,
    sensor)

# Solve
model.init()
with Timer() as t_solve:
    u_e, _, tf, = model.rk(tstart, tend)

print("Solver time: ", t_solve.elapsed()[0])
print("Domain length: ", L)
print("Final time:", tf)

class Lossy:
    """
    This is the analytical solution of the linear wave equation with
    attenuation.

    """

    def __init__(self, c0, f0, p0, a0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.a0 = a0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.sin(self.w0 * (self.t - x[0]/self.c0)) * \
            np.heaviside(self.t-x[0]/self.c0, 0) * \
            np.exp(-self.a0*x[0])

        return val


# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

u_eval = u_e.eval(x, cells).flatten()

V_ba = FunctionSpace(mesh, ("Lagrange", degree))
u_linear = Function(V_ba)
u_linear.interpolate(Lossy(c0, f0, p0, alphaNp, tf))
u_linear_eval = u_linear.eval(x, cells).flatten()

plt.figure(figsize=(14, 8))
plt.plot(x.T[0], u_eval)
plt.plot(x.T[0], u_linear_eval)
plt.xlim([xsh - 2 * lmbda, xsh + 2 * lmbda])
plt.savefig("u.png", bbox_inches="tight")
plt.close()
