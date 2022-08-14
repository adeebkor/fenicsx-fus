#
# .. _lossy_planar2d:
#
# Lossy solver for the 2D planar transducer problem
# =================================================
# Copyright (C) 2022 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx import cpp

from hifusim import LossyGLL, compute_diffusivity_of_sound

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Source parameters
sourceFrequency = 0.5e6  # (Hz)
sourceAmplitude = 60000  # (Hz)
period = 1 / sourceFrequency  # (s)

# Material parameters
speedOfSound = 1500  # (m/s)
attenuationCoefficientdB = 100.0  # (dB/m)
diffusivityOfSound = compute_diffusivity_of_sound(
    sourceFrequency, speedOfSound, attenuationCoefficientdB)
