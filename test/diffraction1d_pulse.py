import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import IntervalMesh, FunctionSpace, Function
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx

from utils import get_eval_params
from models import LinearPulse
from runge_kutta_methods import solve2

# Material parameters
c0 = 1500  # speed of sound (m/s)
rho0 = 1000 # density of medium (kg/m^3)
beta = 10  # coefficient of nonlinearity

# Source parameters
f0 = 1E5  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
p0 = 1E6  # pressure amplitude (Pa)
Td = 6/f0  # source envelope delay (s)
Tw = 3/f0  # source envelope width (s)
Tend = 2 * Td  # source end time (s)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = xsh + Tend*c0  # domain length (m)
print(L)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)
print(k)

# FE parameters
degree = 1  # degree of basis function

# Mesh parameters
epw = 256  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx
print(nw)
exit()

# Generate mesh
mesh = IntervalMesh(
    MPI.COMM_WORLD,
    nx,
    [0, L]
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
tend = L / c0  # simulation final time (s)

CFL = 0.9
dt = CFL * h / 2 / (c0 * (2 * degree + 1))
nstep = int(tend / dt)

print("Final time:", tend)
print("Number of steps:", nstep)

# Instantiate model
eqn = LinearPulse(mesh, mt, degree, c0, f0, p0)

# Solve
u, tf = solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4)
# tf += dt

# Calculate L2 and H1 errors of FEM solution and best approximation
class Analytical:
    def __init__(self, c0, f0, p0, t):
        self.p0 = p0
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.Td = 6/f0
        self.Tw = 3/f0
        self.Tend = 2 * self.Td
        self.t = t

    def __call__(self, x):
        val = - self.p0 * np.sin(self.w0 * (self.t-x[0]/self.c0-self.Td)) * \
              np.exp(-((self.t-x[0]/self.c0-self.Td)/(self.Tw/2))**2) * \
              (np.heaviside(self.t-x[0]/self.c0, 0) - 
               np.heaviside(self.t-x[0]/self.c0-self.Tend, 0))
        return val

V_e = FunctionSpace(mesh, ("Lagrange", degree+2))
u_e = Function(V_e)
u_e.interpolate(Analytical(c0, f0, p0, tf))

# L2 error
diff = u - u_e
L2_diff = mesh.mpi_comm().allreduce(assemble_scalar(inner(diff, diff) * dx), op=MPI.SUM)
L2_exact = mesh.mpi_comm().allreduce(assemble_scalar(inner(u_e, u_e) * dx), op=MPI.SUM)
print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))


# Plot solution
# npts = 5000
# x0 = np.linspace(0, L, npts)
# points = np.zeros((3, npts))
# points[0] = x0
# idx, x, cells = get_eval_params(mesh, points)

# u_eval = u.eval(x, cells).flatten()
# print((tf-tend) / dt)

# Analytical solution
# u_analytic = - p0 * np.sin(w0 * (tf-x0/c0-Td)) * \
#              np.exp(-((tf-x0/c0-Td)/(Tw/2))**2) * \
#              (np.heaviside(tf-x0/c0, 0)-np.heaviside(tf-x0/c0-Tend, 0))

# plt.plot(x0, u_eval)
# plt.plot(x0, u_analytic, '-.')
# plt.xlim([0.5, 0.7])

# plt.savefig("u_eval.png")
# plt.close()

# plt.plot(x0, u_eval-u_analytic)

# plt.savefig("error.png")
# plt.close()

# print(np.linalg.norm(u_eval-u_analytic)/np.linalg.norm(u_analytic))
