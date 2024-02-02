#
# .. _westervelt_planewave1d_1_exp:
#
# Westervelt solver for the 1D planewave problem
# - structured mesh
# - first-order Sommerfeld ABC
# - homogenous medium
# - explicit RK solver
# ==============================================
# Copyright (C) 2023 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
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
p0 = 125000  # pressure amplitude (Pa)

# Material parameters
c0 = 1482.32  # speed of sound (m/s)
rho0 = 998.2  # density (kg / m^3)
beta0 = 3.5  # nonlinearity coefficient
alphadB = 0.217  # attenuation (dB/m)
alphaNp = alphadB / 20 * np.log(10)
delta0 = compute_diffusivity_of_sound(
    w0, c0, alphadB)

# Domain parameters
L = 0.12  # domain length (m)
xsh = rho0*c0**3/beta0/p0/w0  # shock formation distance (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)

# FE parameters
degree = 4

# RK parameter
rk = 4

# Mesh parameters
epw = 3
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
CFL = 0.8
dt = CFL * h / (c0 * degree**2)

tstart = 0.0  # simulation start time (s)
tend = L / c0 + 16 / f0  # simulation final time (s)

# Model
model = WesterveltSpectralExplicit(
    mesh, mt, degree, c, rho, delta, beta, f0, p0, c0, rk, dt)

# Solve
model.init()
with Timer() as t_solve:
    u_e, _, tf, = model.rk(tstart, tend)

print("Solver time: ", t_solve.elapsed()[0])
print("Shock formation distance: ", xsh)
print("Domain length: ", L)

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0, L, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

u_eval = u_e.eval(x, cells).flatten()


class Nonlinear:
    """
    This solution is not the analytical solution for the Burgers' equation.
    It is the solution multiply by the attenuation term.

    """

    def __init__(self, c0, f0, p0, rho0, a0, beta0, t):
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.p0 = p0
        self.u0 = p0 / rho0 / c0
        self.rho0 = rho0
        self.a0 = a0
        self.beta0 = beta0
        self.t = t

    def __call__(self, x):
        xsh = self.c0**2 / self.w0 / self.beta0 / self.u0
        sigma = (x[0]+0.0000001) / xsh

        val = np.zeros(sigma.shape[0])
        for term in range(1, 100):
            val += 2/term/sigma * jv(term, term*sigma) * \
                np.sin(term*self.w0*(self.t - x[0]/self.c0)) * \
                np.exp(-self.a0*x[0])

        return self.p0 * val


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


V_ba = FunctionSpace(mesh, ("Lagrange", degree))
u_nonlinear = Function(V_ba)
u_nonlinear.interpolate(Nonlinear(c0, f0, p0, rho0, alphaNp, beta0, tf))

u_nonlinear_eval = u_nonlinear.eval(x, cells).flatten()

u_linear = Function(V_ba)
u_linear.interpolate(Lossy(c0, f0, p0, alphaNp, tf))
u_linear_eval = u_linear.eval(x, cells).flatten()

plt.figure(figsize=(14, 8))
plt.plot(x.T[0], u_eval, label="FEniCSx")
plt.plot(x.T[0], u_nonlinear_eval, "--", label="Nonlinear")
plt.plot(x.T[0], u_linear_eval, label="Lossy")
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig("u_1.png", bbox_inches="tight")
plt.close()

data_range = [0.06, 0.06 + 2 * lmbda]
idx = np.argwhere(np.logical_and(
    x.T[0] > data_range[0], x.T[0] < data_range[1]))
plt.figure(figsize=(16, 8))
plt.plot(x.T[0][idx], u_eval[idx], label="FEniCSx")
plt.plot(x.T[0][idx], u_nonlinear_eval[idx], "--", label="Nonlinear")
plt.plot(x.T[0][idx], u_linear_eval[idx], "--", label="Lossy")
plt.xlim([data_range[0], data_range[1]])
plt.tick_params(left=False, right=False, labelleft=False,
                labelbottom=True, bottom=True)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title(f"Attenuation = {alphadB}")
plt.savefig(f"u_2_{str(alphadB).zfill(3)}.png", bbox_inches="tight")
plt.close()
