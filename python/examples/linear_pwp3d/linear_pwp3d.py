#
# .. _linear_pwp3d:
#
# Linear solver for the 3D periodic wave propagation problem
# ==========================================================
# Section 4.1 
# Copyright (C) 2021 Adeeb Arif Kor

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.io import XDMFFile
from dolfinx.mesh import (MeshTags, CellType, create_box, 
                          locate_entities_boundary)
from ufl import FiniteElement, inner, dx

from hifusim import PeriodicWavePropagation, PeriodicWavePropagationExact


# Material parameters
c0 = 1500  # speed of sound (m / s)
rho0 = 1000  # density of medium (kg / m^3)

# Source parameters
f0 = 1e6  # source frequency
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1.0  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# Domain parameters
L = lmbda  # domain length (m)

# FE parameters
degree = 3

# Mesh parameters
epw = 10  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = np.sqrt(2 * (L/nx)**2)

# Temporal parameters
CFL = 0.8
dt = CFL * h / c0 / degree**2
t0 = 0.0
tf = 1 / f0 / 2.0

# Create box mesh
mesh = create_box(
	MPI.COMM_WORLD,
	((0.0, 0.0, 0.0), (L, L, L)),
	(nx, nx, nx),
	CellType.hexahedron)

# Tag boundaries
facets0 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 0.0))
facets1 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[1], 0.0))
facets2 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 0.0))
facets3 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], L))
facets4 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[1], L))
facets5 = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], L))
indices, pos = np.unique(np.hstack((
	facets0, facets1, facets2, facets3, facets3, facets5)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
					np.full(facets1.shape, 1, np.intc),
					np.full(facets2.shape, 1, np.intc),
					np.full(facets3.shape, 1, np.intc),
					np.full(facets4.shape, 1, np.intc),
					np.full(facets5.shape, 1, np.intc)))
mt = MeshTags(mesh, 2, indices, values[pos])

# Model
eqn = PeriodicWavePropagation(mesh, mt, degree, rho0, c0, f0, p0)
PETSc.Sys.syncPrint("Degrees of freedom:", eqn.V.dofmap.index_map.size_global)

# Set initial condition
# eqn.u_n.interpolate(lambda x: rho0*c0*(np.sin(x[0]*k)
# 										+ np.sin(x[1]*k)
# 										+ np.sin(x[2]*k)))
eqn.u_n.x.array[:] = 0.0
eqn.v_n.x.array[:] = 0.0

# Solve
eqn.rk4(t0, tf, dt)
eqn.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
						   mode=PETSc.ScatterMode.FORWARD)


# Calculate L2 error
class Analytical:
	def __init__(self, rho0, c0, f0, t):
		self.rho0 = rho0
		self.c0 = c0
		self.k = 2 * np.pi * f0 / c0
		self.f0 = f0
		self.w0 = 2 * np.pi * f0
		self.t = t

	def __call__(self, x):
		val = self.rho0 * self.c0 * (
			np.sin(self.k*x[0]) + np.sin(self.k*x[1]) + np.sin(self.k*x[2])) \
				* np.cos(self.w0*self.t)

		return val

FE_exact = FiniteElement("Lagrange", mesh.ufl_cell(), degree+3, variant="gll")
V_exact = FunctionSpace(mesh, FE_exact)
u_exact = Function(V_exact)
u_exact.interpolate(Analytical(rho0, c0, f0, tf))

# L2 error
diff = eqn.u_n - u_exact
L2_diff = mesh.comm.allreduce(
	assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = mesh.comm.allreduce(
	assemble_scalar(form(inner(u_exact, u_exact) * dx)), op=MPI.SUM)

L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

print("L2 error is:", L2_error)

with XDMFFile(MPI.COMM_WORLD, "u_fe.xdmf", "w") as f:
	f.write_mesh(mesh)
	f.write_function(eqn.u_n)
	# f.write_function(u_exact)

with XDMFFile(MPI.COMM_WORLD, "u_exact.xdmf", "w") as f:
	f.write_mesh(mesh)
	f.write_function(u_exact)
