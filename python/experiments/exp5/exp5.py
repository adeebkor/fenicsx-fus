#
# .. exp5:
#
# Experiment 5: 1D lossy planewave
# =================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import (create_interval, locate_entities_boundary, meshtags)

from fenicsxfus import LossySpectralExplicit
from fenicsxfus.utils import compute_diffusivity_of_sound, compute_eval_params

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1.0 / sourceFrequency  # (s)
angularFrequency = 2 * np.pi * sourceFrequency  # (rad / s)

# Material parameters
speedOfSound = 1500.0  # (m/s)
density = 1000  # (kg/m^3)
attenuationCoefficientdB = 1500  # (dB/m)
attenuationCoefficientNp = attenuationCoefficientdB / 20 * np.log(10)
diffusivityOfSound = compute_diffusivity_of_sound(
    angularFrequency, speedOfSound, attenuationCoefficientdB)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# RK parameter
rkOrder = 4

# Physical parameters
wavelength = speedOfSound / sourceFrequency  # wavelength (m)

# Mesh parameters
numElementPerWavelength = 8
numberOfElements = int(numElementPerWavelength * domainLength / wavelength) + 1
meshSize = domainLength / numberOfElements

# Generate mesh
mesh = create_interval(MPI.COMM_WORLD, numberOfElements, [0, domainLength])

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

# Define a DG function space for the medium properties
V_DG = FunctionSpace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speedOfSound

rho0 = Function(V_DG)
rho0.x.array[:] = density

delta0 = Function(V_DG)
delta0.x.array[:] = diffusivityOfSound

# Temporal parameters
CFL = 1/12
timeStepSize = CFL * meshSize / speedOfSound
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod  # adjust time step size
startTime = 0.0
finalTime = (domainLength / speedOfSound + 4 / sourceFrequency) / 4
numberOfStep = int(finalTime / timeStepSize + 1)

# Model
model = LossySpectralExplicit(
    mesh, mt, degreeOfBasis, c0, rho0, delta0, sourceFrequency,
    sourceAmplitude, speedOfSound, rkOrder, timeStepSize)

# Solve
model.init()
u_n, v_n, tf = model.rk(startTime, finalTime)


# Best approximation
class Analytical:
    """ Analytical solution """

    def __init__(self, c0, a0, f0, p0, t):
        self.c0 = c0
        self.a0 = a0
        self.p0 = p0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.exp(1j*(self.w0*self.t - self.w0/self.c0*x[0])) \
                * np.exp(-self.a0*x[0])

        return val.imag


V_ba = FunctionSpace(mesh, ("Lagrange", degreeOfBasis))
u_ba = Function(V_ba)
u_ba.interpolate(Analytical(speedOfSound, attenuationCoefficientNp,
                            sourceFrequency, sourceAmplitude, tf))

# Plot the solution
npts = 1024
x0 = np.linspace(0, domainLength, npts)
points = np.zeros((3, npts))
points[0] = x0
x, cells = compute_eval_params(mesh, points)

uh_eval = u_n.eval(x, cells).flatten()
ue_eval = u_ba.eval(x, cells).flatten()

plt.plot(x.T[0], uh_eval, "-C0", lw=2, label="FEniCSx")
plt.plot(x.T[0], ue_eval, "--C1", label="Analytical")
plt.xlim([0, 0.12])
plt.legend()
plt.savefig("sol.png")
plt.close()

print(f"{attenuationCoefficientdB},{diffusivityOfSound:5.5},{degreeOfBasis},",
      f"{meshSize:6.6},{rkOrder},{CFL:5.5},{timeStepSize:5.5}")
