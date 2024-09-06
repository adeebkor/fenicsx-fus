#
# .. _westervelt_planar2d_3_exp:
#
# Westervelt solver for the 2D planar transducer problem
# - structured mesh
# - first-order Sommerfeld ABC
# - two different medium (x < 0.06 m (water with no attenuation),
#                         x > 0.06 m (bone with attenuation))
# - explicit Runge-Kutta
# ===============================================================
# Copyright (C) 2023 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import functionspace, Function
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
speedOfSoundWater = 1500  # (m/s)
speedOfSoundBone = 2800  # (m/s)
densityWater = 1000  # (kg/m^3)
densityBone = 1850  # (kg/m^3)
nonlinearCoefficientWater = 100
nonlinearCoefficientBone = 200
attenuationCoefficientdBWater = 0.0  # (dB/m)
attenuationCoefficientdBBone = 100.0  # (dB/m)
diffusivityOfSoundWater = compute_diffusivity_of_sound(
    angularFrequency, speedOfSoundWater, attenuationCoefficientdBWater
)
diffusivityOfSoundBone = compute_diffusivity_of_sound(
    angularFrequency, speedOfSoundBone, attenuationCoefficientdBBone
)

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
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")
    mt = [mt_cell, mt_facet]

# Mesh parameters
numCell = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh._cpp_object, tdim, np.arange(numCell)).min()])
meshSize = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, meshSize, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(meshSize, root=0)

# Define a DG function space for the physical parameters of the domain
V_DG = functionspace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speedOfSoundWater
c0.x.array[mt_cell.find(2)] = speedOfSoundBone

rho0 = Function(V_DG)
rho0.x.array[:] = densityWater
rho0.x.array[mt_cell.find(2)] = densityBone

beta0 = Function(V_DG)
beta0.x.array[:] = nonlinearCoefficientWater
beta0.x.array[mt_cell.find(2)] = nonlinearCoefficientBone

delta0 = Function(V_DG)
delta0.x.array[:] = diffusivityOfSoundWater
delta0.x.array[mt_cell.find(2)] = diffusivityOfSoundBone

# Temporal parameters
CFL = 1.0
timeStepSize = CFL * meshSize / (speedOfSoundBone * degreeOfBasis**2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSoundWater + 8.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Model: Westervelt", flush=True)
    print("Problem type: Planar 2D", flush=True)
    print("Runge-Kutta type: Explicit", flush=True)
    print(f"Speed of sound (Water): {speedOfSoundWater}", flush=True)
    print(f"Speed of sound (Bone): {speedOfSoundBone}", flush=True)
    print(f"Density (Water): {densityWater}", flush=True)
    print(f"Density (Bone): {densityBone}", flush=True)
    print(f"Diffusivity of sound (Water): {diffusivityOfSoundWater}", flush=True)
    print(f"Diffusivity of sound (Bone): {diffusivityOfSoundBone}", flush=True)
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
    mesh,
    mt_facet,
    degreeOfBasis,
    c0,
    rho0,
    delta0,
    beta0,
    sourceFrequency,
    sourceAmplitude,
    speedOfSoundWater,
    rkOrder,
    timeStepSize,
)

# Solve
model.init()
u_n, v_n, tf = model.rk(startTime, finalTime)

with VTXWriter(mesh.comm, "output_final.bp", u_n) as f:
    f.write(0.0)
