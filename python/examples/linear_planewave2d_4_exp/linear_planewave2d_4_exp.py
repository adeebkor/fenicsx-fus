#
# .. _linear_planewave2d_4_exp:
#
# Linear solver for the 2D planewave problem
# - structured mesh
# - first-order Sommerfeld
# - two different medium (x < 0.06 m, x > 0.06 m)
# - explicit Runge-Kutta solver
# ===================================================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp

from hifusim import LinearGLLExplicit

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Pa)
period = 1 / sourceFrequency  # (s)

# Material parameters
speedOfSoundWater = 1500  # (m/s)
speedOfSoundBone = 2800  # (m/s)
densityWater = 1000  # (kg/m^3)
densityBone = 1850  # (kg/m^3)

# Domain parameters
domainLength = 0.12  # (m)

# FE parameters
degreeOfBasis = 4

# RK parameter
rkOrder = 4

# Read mesh and mesh tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "planewave_2d_4"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")
    mt = [mt_cell, mt_facet]

# Mesh parameters
numCell = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh, tdim, range(numCell)).min()])
meshSize = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, meshSize, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(meshSize, root=0)

# Define DG functions to specify different medium
V_DG = FunctionSpace(mesh, ("DG", 0))
c0 = Function(V_DG)
c0.x.array[:] = speedOfSoundWater
c0.x.array[mt_cell.find(2)] = speedOfSoundBone

rho0 = Function(V_DG)
rho0.x.array[:] = densityWater
rho0.x.array[mt_cell.find(2)] = densityBone

# Temporal parameters
CFL = 0.9
timeStepSize = CFL * meshSize / (speedOfSoundBone * degreeOfBasis ** 2)
stepPerPeriod = int(period / timeStepSize + 1)
timeStepSize = period / stepPerPeriod
startTime = 0.0
finalTime = domainLength / speedOfSoundWater + 4.0 / sourceFrequency
numberOfStep = int((finalTime - startTime) / timeStepSize + 1)

if mpi_rank == 0:
    print("Problem type: Planewave 2D (heterogenous)", flush=True)
    print(f"Speed of sound (Water): {speedOfSoundWater}", flush=True)
    print(f"Speed of sound (Bone): {speedOfSoundBone}", flush=True)
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
model = LinearGLLExplicit(mesh, mt_facet, degreeOfBasis, c0, rho0,
                          sourceFrequency, sourceAmplitude, speedOfSoundWater,
                          rkOrder, timeStepSize)

# Solve
model.init()
u_n, v_n, tf = model.rk(startTime, finalTime)

with VTXWriter(mesh.comm, "output_final.bp", u_n) as out:
    out.write(0.0)


# Best approximation
class Wave:
    """ Analytical solution """

    def __init__(self, c1, c2, rho1, rho2, f, p, t):
        self.r1 = c1*rho1
        self.r2 = c2*rho2

        self.ratio = self.r2 / self.r1

        self.R = (self.ratio - 1) / (self.ratio + 1)
        self.T = 2 * self.ratio / (self.ratio + 1)

        self.f = f
        self.w = 2 * np.pi * f
        self.p = p
        self.k1 = self.w / c1
        self.k2 = self.w / c2

        self.t = t

    def field(self, x):
        x0 = x[0] + 0.j  # need to plus 0.j because piecewise return same type
        val = np.piecewise(
            x0, [x0 < 0.06, x0 >= 0.06],
            [lambda x: self.R * self.p * np.exp(
                1j * (self.w * self.t - self.k1 * (x - 0.06))),
             lambda x: self.T * self.p * np.exp(
                1j * (self.w * self.t - self.k2 * (x - 0.06)))])

        return val.imag


V_ba = FunctionSpace(mesh, ("Lagrange", degreeOfBasis))
u_ba = Function(V_ba)
wave = Wave(speedOfSoundWater, speedOfSoundBone, densityWater, densityBone,
            sourceFrequency, sourceAmplitude, tf)
u_ba.interpolate(wave.field)

with VTXWriter(mesh.comm, "output_analytical.bp", u_ba) as f_ba:
    f_ba.write(0.0)
