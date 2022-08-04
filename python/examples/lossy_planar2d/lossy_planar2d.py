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

from hifusim import LossyGLL
