#
# .. _lossy_planewave2d:
#
# Lossy solver for the 2D planewave problem
# =========================================
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp
from dolfinx.io import VTKFile

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

# Domain parameters
domainLength = 0.12

# FE parameters
degreeOfBasis = 4

# Read mesh and mesh tags
