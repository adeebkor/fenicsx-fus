#
# .. exp4:
#
# Experiment 4: 1D linear planewave
# =================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import (create_interval, locate_entities_boundary, meshtags)
from ufl import inner, dx

from fenicsxfus import LinearSpectralExplicit
from fenicsxfus.utils import compute_eval_params

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

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# RK parameter
rkOrder = 1

# Physical parameters
wavelength = speedOfSound / sourceFrequency  # wavelength (m)

# Mesh parameters
numElementPerWavelength = 4
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

# Temporal parameters
CFL = 1/3200
timeStepSize = CFL * meshSize / speedOfSound
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod  # adjust time step size
startTime = 0.0
finalTime = domainLength / speedOfSound + 16 / sourceFrequency
numberOfStep = int(finalTime / timeStepSize + 1)

# Model
model = LinearSpectralExplicit(
    mesh, mt, degreeOfBasis, c0, rho0, sourceFrequency, sourceAmplitude,
    speedOfSound, rkOrder, timeStepSize)

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


V_e = FunctionSpace(mesh, ("Lagrange", degreeOfBasis))
u_e = Function(V_e)
u_e.interpolate(Analytical(speedOfSound, sourceFrequency, sourceAmplitude, tf))

# L2 error
diff = uh - u_e
L2_diff = mesh.comm.allreduce(
    assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = mesh.comm.allreduce(
    assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM)

L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

print(
    f"{degreeOfBasis},{meshSize:6.6},{rkOrder},{CFL:5.5},",
    f"{timeStepSize:8.8},{L2_error:8.8}")

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
plt.xlim([0.0, 0.12])
plt.legend()
plt.savefig("sol.png")
plt.close()

L2_error_disc = np.linalg.norm(uh_eval - ue_eval)/np.linalg.norm(ue_eval)
print(f"L2 error (numpy): {L2_error_disc:5.5}")
