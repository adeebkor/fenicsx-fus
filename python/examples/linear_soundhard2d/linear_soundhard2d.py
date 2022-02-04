#
# .. _linear_soundhard2d:
#
# Linear solver for the 2D sound hard problem
# ===========================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp
from dolfinx.common import Timer
from dolfinx.io import XDMFFile

from hifusim import LinearSoundHardGLL

# Material parameters
c0 = 1  # speed of sound (m/s)
rho0 = 1  # density of medium (kg / m^3)

# Source parameters
f0 = 10  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 4

# Read mesh and meshtags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="sound_hard")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    mt = xdmf.read_meshtags(mesh, "sound_hard_surface")

# Mesh parameters
tdim = mesh.topology.dim
num_cells = num_cells = mesh.topology.index_map(tdim).size_local
hmin = np.array([cpp.mesh.h(mesh, tdim, range(num_cells)).min()])
h = np.zeros(1)
MPI.COMM_WORLD.Reduce(hmin, h, op=MPI.MIN, root=0)
MPI.COMM_WORLD.Bcast(h, root=0)

# Model
eqn = LinearSoundHardGLL(mesh, mt, degree, c0, f0, p0)
PETSc.Sys.syncPrint("Degrees of freedom:", eqn.V.dofmap.index_map.size_global)

# Temporal parameters : allows wave to fully propagate across the domain
CFL = 0.225
tstart = 0.0  # start time (s)
dt = CFL * h[0] / (c0 * degree**2)  # time step size
tend = L / c0 + 6 / f0  # final time (s)

# Solve
eqn.init()
with Timer() as tsolve:
    u, v, tf, nstep = eqn.rk4(tstart, tend, dt)

print("Solve time per step:", tsolve.elapsed()[0] / nstep)
