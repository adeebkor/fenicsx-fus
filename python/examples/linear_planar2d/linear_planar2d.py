#
# .. _linear_planar2d:
#
# Linear solver for the 2D planewave problem
# ==========================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx import cpp

from hifusim import LinearGLL

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Material parameters
speedOfSound = 1500  # (m/s)

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1 / sourceFrequency  # (s)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# Read mesh and mesh tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh = fmesh.read_mesh(name="planar2d")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name="planar2d_regions")
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name="planar2d_boundaries")

# Mesh parameters
numCell = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh, tdim, range(numCell)).min()])
meshSize = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, meshSize, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(meshSize, root=0)

# Temporal parameters
CFL = 0.62
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis**2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSound + 8.0 / sourceFrequency
numberOfStep = int(finalTime / timeStepSize + 1)

if mpi_rank == 0:
    print(f"Problem type: Planar 2D", flush=True)
    print(f"Speed of sound: {speedOfSound}", flush=True)
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
model = LinearGLL(mesh, mt_facet, degreeOfBasis, speedOfSound,
                  sourceFrequency, sourceAmplitude)

# Solve
model.init()
model.rk4(startTime, finalTime, timeStepSize)
