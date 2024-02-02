#
# .. _westervelt_planewave2d_1_exp:
#
# Westervelt solver for the 2D planewave problem
# - structured mesh
# - first-order Sommerfeld ABC
# - explicit Runge-Kutta
# ==============================================
# Copyright (C) 2023 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp, common

from hifusim import WesterveltSpectralExplicit
from hifusim.utils import compute_diffusivity_of_sound

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1.0 / sourceFrequency  # (s)
angularFrequency = 2 * np.pi * sourceFrequency  # (rad/s)

# Material parameters
speedOfSound = 1500.0  # (m/s)
density = 1000  # (kg/m^3)
nonlinearCoefficient = 300.0
attenuationCoefficientdB = 50  # (dB/m)
attenuationCoefficientNp = attenuationCoefficientdB / 20 * np.log(10)
diffusivityOfSound = compute_diffusivity_of_sound(
    angularFrequency, speedOfSound, attenuationCoefficientdB)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# RK parameter
rkOrder = 4

# Read mesh and mesh tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "planewave_2d_1"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")

# Mesh parameters
numCell = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh, tdim, range(numCell)).min()])
meshSize = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, meshSize, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(meshSize, root=0)

# Define a DG function space for the physical parameters of the domain
V_DG = FunctionSpace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speedOfSound

rho0 = Function(V_DG)
rho0.x.array[:] = density

beta0 = Function(V_DG)
beta0.x.array[:] = nonlinearCoefficient

delta0 = Function(V_DG)
delta0.x.array[:] = diffusivityOfSound

# Temporal parameters
CFL = 0.8
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis ** 2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSound + 4.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Model: Westervelt", flush=True)
    print("Problem type: Planewave 2D", flush=True)
    print("Runge-Kutta type: Explicit", flush=True)
    print(f"Speed of sound: {speedOfSound}", flush=True)
    print(f"Density: {density}", flush=True)
    print(f"Diffusivity of sound: {diffusivityOfSound}", flush=True)
    print(f"Source frequency: {sourceFrequency}", flush=True)
    print(f"Source amplitude: {sourceAmplitude}", flush=True)
    print(f"Domain length: {domainLength}", flush=True)
    print(f"Polynomial basis degree: {degreeOfBasis}", flush=True)
    print(f"Minimum mesh size: {meshSize[0]:4.4}", flush=True)
    print(f"CFL number: {CFL}", flush=True)
    print(f"Time step size: {timeStepSize:4.4}", flush=True)
    print(f"Number of step per period: {stepPerPeriod}", flush=True)
    print(f"Number of steps: {numberOfStep}", flush=True)

# Model
model = WesterveltSpectralExplicit(
    mesh, mt_facet, degreeOfBasis, c0, rho0, delta0, beta0,
    sourceFrequency, sourceAmplitude, speedOfSound,
    rkOrder, timeStepSize)

# Solve
model.init()

with common.Timer() as tsolve:
    u_n, v_n, tf = model.rk(startTime, finalTime)

if mpi_rank == 0:
    print("Solve time: ", tsolve.elapsed()[0])
    print("Time per step: ", tsolve.elapsed()[0]/numberOfStep)


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

with VTXWriter(mesh.comm, "output_nonlinear.bp", u_n) as f:
    f.write(0.0)

with VTXWriter(mesh.comm, "output_lossy.bp", u_ba) as f_ba:
    f_ba.write(0.0)
