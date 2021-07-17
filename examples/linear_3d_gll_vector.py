import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import BoxMesh, FunctionSpace, Function
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx

from models import LinearGLLv
from runge_kutta_methods import solve2

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000  # density of medium (kg / m^3)
beta = 3.5  # coefficient of nonlinearity

# Source parameters
f0 = 5E6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
u0 = 1  # velocity amplitude (m / s)
p0 = rho0*c0*u0  # pressure amplitude (Pa)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 0.9 * xsh  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 3  # degree of basis function

# Mesh parameters
epw = 4  # number of element per wavelength
nw = L / lmbda  # number of waves
n = int(epw * nw + 1)  # total number of elements
h = np.sqrt(3*(L / n)**2)

PETSc.Sys.syncPrint("Element size:", h)

# Generate mesh
mesh = BoxMesh(
    MPI.COMM_WORLD,
    [np.array([0., 0., 0.]), np.array([L, L, L])],
    [n, n, n],
    CellType.hexahedron
)

# Tag boundaries
tdim = mesh.topology.dim

facets0 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
facets1 = locate_entities_boundary(
    mesh, tdim-1, lambda x: x[0] > L - np.finfo(float).eps)

indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
values = np.hstack((np.full(facets0.shape, 1, np.intc),
                    np.full(facets1.shape, 2, np.intc)))
mt = MeshTags(mesh, tdim-1, indices, values[pos])

# Temporal parameters
tstart = 0.0  # simulation start time (s)
tend = L / c0 + 2 / f0  # simulation final time (s)
CFL = 0.7
dt = CFL * h / (c0 * (2 * degree + 1))

nstep = int(tend / dt)

PETSc.Sys.syncPrint("Final time:", tend)
PETSc.Sys.syncPrint("Number of steps:", nstep)

# Instantiate model
eqn = LinearGLLv(mesh, mt, degree, c0, f0, p0)
dofs = eqn.V.dofmap.index_map.size_global
PETSc.Sys.syncPrint("Degree of freedoms:", dofs)

# Solve
u, tf = solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4)
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)
PETSc.Sys.syncPrint("tf:", tf)


# Calculate L2 error
class Analytical:
    def __init__(self, c0, f0, p0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.t = t

    def __call__(self, x):
        val = self.p0 * np.sin(self.w0 * (self.t - x[0]/self.c0)) * \
              np.heaviside(self.t-x[0]/self.c0, 0)

        return val


u_ba = Function(eqn.V)
u_ba.interpolate(Analytical(c0, f0, p0, tf))

V_e = FunctionSpace(mesh, ("Lagrange", degree+3))
u_e = Function(V_e)
u_e.interpolate(Analytical(c0, f0, p0, tf))

# L2 error
diff_fe = u - u_e
L2_diff_fe = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(diff_fe, diff_fe) * dx), op=MPI.SUM)

diff_ba = u_ba - u_e
L2_diff_ba = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(diff_ba, diff_ba) * dx), op=MPI.SUM)

L2_exact = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(u_e, u_e) * dx), op=MPI.SUM)

L2_error_fe = abs(np.sqrt(L2_diff_fe) / np.sqrt(L2_exact))
PETSc.Sys.syncPrint("Relative L2 error of FEM solution:", L2_error_fe)

L2_error_ba = abs(np.sqrt(L2_diff_ba) / np.sqrt(L2_exact))
PETSc.Sys.syncPrint("Relative L2 error of BA solution:", L2_error_ba)

# Plot solution
filename = "solution/3d/linear_gll_p{}_epw{}.xdmf".format(degree, epw)
with XDMFFile(MPI.COMM_WORLD, filename, "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)
