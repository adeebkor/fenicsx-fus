#
# .. _westervelt_planar2d_2_exp:
#
# Westervelt solver for the 2D planar transducer problem
# - structured mesh
# - first-order Sommerfeld ABC
# - different attenuation between 2 medium (x < 0.06 m, x > 0.06 m)
# - explicit Runge-Kutta
# =================================================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp

from fenicsxfus import WesterveltSpectralExplicit
from fenicsxfus.utils import compute_diffusivity_of_sound

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1 / sourceFrequency  # (s)
angularFrequency = 2 * np.pi * sourceFrequency  # (rad / s)

# Material parameters
speedOfSound = 1500  # (m/s)
density = 1000  # (kg/m^3)
nonlinearCoefficient = 150.0
attenuationCoefficientdB = 100  # (dB/m)
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
    mesh_name = "planar_2d_2"
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
delta0.x.array[:] = 0.0
delta0.x.array[mt_cell.find(2)] = diffusivityOfSound

# Temporal parameters
CFL = 1.0
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis ** 2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSound + 8.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Model: Westervelt", flush=True)
    print("Problem type: Planar 2D", flush=True)
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
u_n, v_n, tf = model.rk(startTime, finalTime)

with VTXWriter(mesh.comm, "output_final.bp", u_n) as f:
    f.write(0.0)
