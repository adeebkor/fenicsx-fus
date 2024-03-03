#
# .. exp3:
#
# Experiment 3: 2D linear "planar" wave
# =====================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp

from fenicsxfus import LinearSpectralExplicit, LinearSpectralS2

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

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# RK parameter
rkOrder = 4

# -----------------------------------------------------------------------------
# Conforming source

# Read mesh and mesh tags
with XDMFFile(MPI.COMM_WORLD, "mesh_conform.xdmf", "r") as fmesh:
    meshc_name = "planar_2d_2"
    meshc = fmesh.read_mesh(name=f"{meshc_name}")
    tdim = meshc.topology.dim
    mt_cellc = fmesh.read_meshtags(meshc, name=f"{meshc_name}_cells")
    meshc.topology.create_connectivity(tdim - 1, tdim)
    mt_facetc = fmesh.read_meshtags(meshc, name=f"{meshc_name}_facets")

# Mesh parameters
numCellc = meshc.topology.index_map(tdim).size_local
hminc = np.array([cpp.mesh.h(meshc, tdim, range(numCellc)).min()])
meshSizec = np.zeros(1)
MPI.COMM_WORLD.Reduce(hminc, meshSizec, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(meshSizec, root=0)

# Define a DG function space for the physical parameters of the domain
V_DGc = FunctionSpace(meshc, ("DG", 0))
c0c = Function(V_DGc)
c0c.x.array[:] = speedOfSound

rho0c = Function(V_DGc)
rho0c.x.array[:] = density

# Temporal parameters
CFLc = 0.9
timeStepSizec = CFLc * meshSizec / (speedOfSound * degreeOfBasis**2)
stepPerPeriodc = int(period / timeStepSizec + 1)
timeStepSizec = period / stepPerPeriodc
startTimec = 0.0
finalTimec = domainLength / speedOfSound + 4.0 / sourceFrequency
numberOfStep = int((finalTimec - startTimec) / timeStepSizec + 1)

if mpi_rank == 0:
    print("Problem type: Conforming source", flush=True)

# Model
modelc = LinearSpectralExplicit(
    meshc,
    mt_facetc,
    degreeOfBasis,
    c0c,
    rho0c,
    sourceFrequency,
    sourceAmplitude,
    speedOfSound,
    rkOrder,
    timeStepSizec,
)

# Solve
modelc.init()
u_nc, v_nc, tfc = modelc.rk(startTimec, finalTimec)

# Output solution
with VTXWriter(meshc.comm, "output_conform.bp", u_nc) as f_nc:
    f_nc.write(0.0)


# -----------------------------------------------------------------------------
# Non-conforming source

# Read mesh and mesh tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "planewave_2d_6"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim - 1, tdim)
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

# Temporal parameters
CFL = 0.9
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis**2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSound + 4.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Problem type: Non-conforming source", flush=True)

# Model
model = LinearSpectralS2(
    mesh,
    mt_facet,
    degreeOfBasis,
    c0,
    rho0,
    sourceFrequency,
    sourceAmplitude,
    speedOfSound,
)

# Solve
model.init()
u_n, v_n, tf = model.rk4(startTime, finalTime, timeStepSize)

# Output solution
with VTXWriter(mesh.comm, "output_nonconform.bp", u_n) as f_n:
    f_n.write(0.0)
