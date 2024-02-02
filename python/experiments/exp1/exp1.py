#
# .. exp1:
#
# Experiment 1: 1D linear planewave
# =================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import create_interval, locate_entities_boundary, meshtags
from ufl import inner, dx

from fenicsxfus import LinearSpectralExplicit
from fenicsxfus.utils import compute_eval_params

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Hz)
period = 1 / sourceFrequency  # (s)

# Material parameters
speedOfSound = 1500  # (m/s)
density = 1000  # (kg/m^3)

# Physical parameters
lmbda = speedOfSound/sourceFrequency  # wavelength (m)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 3

# RK parameter
rkOrder = 4

# Mesh parameters
elementPerWavelength = 8
numberOfWaves = domainLength / lmbda
numberOfElements = int(elementPerWavelength * numberOfWaves + 1)
meshSize = domainLength / numberOfElements

# Generate mesh
mesh = create_interval(MPI.COMM_WORLD, numberOfElements, [0, domainLength])

# Define a DG function space for the physical parameters of the domain
V_DG = FunctionSpace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speedOfSound

rho0 = Function(V_DG)
rho0.x.array[:] = density

# Tag boundaries
tdim = mesh.topology.dim

facets0 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] > domainLength - np.finfo(float).eps)

indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = meshtags(mesh, tdim-1, indices, values[pos])

# Temporal parameters
CFL = 0.9
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis**2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod  # adjust time step size
startTime = 0.0
finalTime = domainLength / speedOfSound + 6 / sourceFrequency
numberOfStep = int(finalTime / timeStepSize + 1)

# Model
model = LinearSpectralExplicit(
    mesh, mt, degreeOfBasis, c0, rho0, sourceFrequency, sourceAmplitude,
    speedOfSound, rkOrder, timeStepSize)
model.alpha = 4.0

# Solve
model.init()
uh, vh, tf = model.rk(startTime, finalTime)


# Compute accuracy
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


V_e = FunctionSpace(mesh, ("Lagrange", degreeOfBasis+3))
u_e = Function(V_e)
u_e.interpolate(Analytical(speedOfSound, sourceFrequency, sourceAmplitude, tf))

# L2 error
diff = uh - u_e
L2_diff = mesh.comm.allreduce(
    assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = mesh.comm.allreduce(
    assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM)

L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))


# Plot the solution
npts = 1024
x0 = np.linspace(0, domainLength, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

uh_eval = uh.eval(x, cells).flatten()
ue_eval = u_e.eval(x, cells).flatten()


plt.plot(x.T[0], uh_eval, "-C0", lw=2, label="FEniCSx")
plt.plot(x.T[0], ue_eval, "--C1", label="Analytical")
plt.xlim([0.12 - 4 * lmbda, 0.12])
plt.legend()
plt.savefig("sol.png")
plt.close()

if mpi_rank == 0:
    print("Problem type: Planewave 1D (Homogenous)", flush=True)
    print(f"Speed of sound: {speedOfSound}", flush=True)
    print(f"Source frequency: {sourceFrequency}", flush=True)
    print(f"Source amplitude: {sourceAmplitude}", flush=True)
    print(f"Domain length: {domainLength}", flush=True)
    print(f"Polynomial basis degree: {degreeOfBasis}", flush=True)
    print(
        f"Number of elements per wavelength: {elementPerWavelength}",
        flush=True)
    print(f"Number of elements: {numberOfElements}", flush=True)
    print(f"Minimum mesh size: {meshSize:4.4}", flush=True)
    print(f"CFL number: {CFL}", flush=True)
    print(f"Time step size: {timeStepSize:4.4}", flush=True)
    print(f"Number of step per period: {stepPerPeriod}", flush=True)
    print(f"Number of steps: {numberOfStep}", flush=True)
    print(f"L2 error: {L2_error}", flush=True)
    print("Relative L2:",
          np.linalg.norm(uh_eval - ue_eval) /
          np.linalg.norm(ue_eval), flush=True)
