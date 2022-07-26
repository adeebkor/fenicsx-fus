#
# .. _lossy_planewave2d:
#
# Lossy solver for the 2D planewave problem
# =========================================
# Copyright (C) 2021 Adeeb Arif Kor

import wave
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import VTKFile, XDMFFile
from dolfinx.mesh import (CellType, create_rectangle, locate_entities_boundary,
                          meshtags)

from hifusim import LossyGLL
from hifusim.utils import compute_diffusivity_of_sound


# MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
pressureAmplitude = 60000  # (Pa)
period = 1.0 / sourceFrequency  # (s)
angularFrequency = 2 * np.pi * sourceFrequency  # (rad / s)

# Material parameters
speedOfSoundWater = 1500.0  # (m/s)
attenuationCoefficientWaterdB = 0.0  # (dB/m)
diffusivityOfSoundWater = compute_diffusivity_of_sound(
    angularFrequency, speedOfSoundWater, attenuationCoefficientWaterdB)

# Wave parameters
waveLength = speedOfSoundWater / sourceFrequency

# Domain parameters
domainRadius = 0.035
domainLength = 0.12
sourceRadius = 0.01
pmlWidth = 15
PML_0 = domainRadius + pmlWidth*waveLength
PML_1 = domainLength + pmlWidth*waveLength
elementPerWavelength = 3
nx = int(elementPerWavelength * PML_1 / waveLength) + 1
ny = int(elementPerWavelength * 2 * PML_0 / waveLength) + 1

# FE parameters
degreeOfBasis = 4

# Create mesh
mesh = create_rectangle(
    MPI.COMM_WORLD,
    ((0., -PML_0), (PML_1, PML_0)),
    (nx, ny),
    cell_type=CellType.quadrilateral)

# Tag facets
source_facets = locate_entities_boundary(mesh, 1,
                                         lambda x: np.logical_and(
                                             np.isclose(x[0], 0.0),
                                             np.logical_and(
                                                 x[1] > -sourceRadius,
                                                 x[1] < sourceRadius)))
abc_facets0 = locate_entities_boundary(
    mesh, 1, lambda x: np.isclose(x[1], -PML_0))
abc_facets1 = locate_entities_boundary(
    mesh, 1, lambda x: np.isclose(x[1], PML_0))
abc_facets2 = locate_entities_boundary(
    mesh, 1, lambda x: np.isclose(x[0], PML_1))

marked_facets = np.hstack(
    [source_facets, abc_facets0, abc_facets1, abc_facets2])
marked_values = np.hstack(
    [np.full_like(source_facets, 1), np.full_like(abc_facets0, 2),
     np.full_like(abc_facets1, 2), np.full_like(abc_facets2, 2)])
sorted_facets = np.argsort(marked_facets)
mt = meshtags(
    mesh, 1, marked_facets[sorted_facets], marked_values[sorted_facets])

# Output mesh to XDMF
# with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as fmesh:
# fmesh.write_mesh(mesh)
# fmesh.write_meshtags(mt)

# Temporal parameters
meshSize = PML_1 / nx
CFL = 0.65
timeStepSize = CFL * meshSize / (speedOfSoundWater * degreeOfBasis**2)
startTime = 0.0
finalTime = domainLength / speedOfSoundWater + 8.0 / sourceFrequency
stepPerPeriod = period / timeStepSize + 1
timeStepSize = period / stepPerPeriod
numberOfStep = finalTime / timeStepSize + 1

# Model
eqn = LossyGLL()
