#
# .. _linear_planewave2d_1:
#
# Linear solver for the 2D planewave problem
# - structured mesh
# - first-order Sommerfeld  ABC
# ==========================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import Function, FunctionSpace
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp

from hifusim import LinearGLL

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1 / sourceFrequency  # (s)

# Material parameters
speedOfSound = 1500  # (m/s)
density = 1000  # (kg/m^3)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

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

# Temporal parameters
CFL = 0.4
timeStepSize = CFL * meshSize / (speedOfSound * degreeOfBasis ** 2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSound + 4.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Problem type: Planewave 2D", flush=True)
    print(f"Speed of sound: {speedOfSound}", flush=True)
    print(f"Density: {density}", flush=True)
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
model = LinearGLL(mesh, mt_facet, degreeOfBasis, speedOfSound, density,
                  sourceFrequency, sourceAmplitude)

# Solve
model.init()
u_n, v_n, tf = model.rk4(startTime, finalTime, timeStepSize)


# Best approximation
class Analytical:
    """ Analytical solution """

    def __init__(self, c0, f0, p0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.exp(1j*(self.w0*self.t - self.w0/self.c0*x[0]))

        return val.imag


V_ba = FunctionSpace(mesh, ("Lagrange", degreeOfBasis))
u_ba = Function(V_ba)
u_ba.interpolate(Analytical(speedOfSound, sourceFrequency, sourceAmplitude,
                            tf))

with VTXWriter(mesh.comm, "output_final.bp", u_n) as f:
    f.write(0.0)

with VTXWriter(mesh.comm, "output_analytical.bp", u_ba) as f_ba:
    f_ba.write(0.0)
