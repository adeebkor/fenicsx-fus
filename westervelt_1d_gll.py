import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import IntervalMesh, FunctionSpace, Function
from dolfinx.fem import assemble_scalar
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx

from utils import get_eval_params
from models import Westervelt1DGLL
from runge_kutta_methods import solve2

# Material parameters
c0 = 1500  # speed of sound (m / s)
rho0 = 1000  # density of medium (kg / m^3)
beta = 3.5  # coefficient of nonlinearity

# Source parameters
f0 = 5E6  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
p0 = 1.5E6  # pressure amplitude (Pa)
u0 = p0 / rho0 / c0  # velocity amplitude (m / s)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 0.9*xsh  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 4  # degree of basis function

# Mesh parameters
epw = 2  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx

print("Element size:", h)

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
tend = L / c0 + 2 / f0  # simulation final time (s)

CFL = 0.9
dt = CFL * h / (c0 * (2 * degree + 1))

nstep = int(2 * tend / dt)

print("Final time:", tend)
print("Number of steps:", nstep)

# Instantiate model
eqn = Westervelt1DGLL(mesh, mt, degree, c0, f0, p0, 0.0, beta, rho0)
dofs = eqn.V.dofmap.index_map.size_global
print("Degree of freedoms:", dofs)

# Solve
u, tf = solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4)
u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)
print("tf:", tf)


# Calculate L2 error
class Analytical:
    def __init__(self, c0, f0, p0, rho0, beta, t):
        self.c0 = c0
        self.f0 = f0
        self.w0 = 2 * np.pi * f0
        self.p0 = p0
        self.u0 = p0 / rho0 / c0
        self.rho0 = rho0
        self.beta = beta
        self.t = t

    def __call__(self, x):
        xsh = self.c0**2 / self.w0 / self.beta / self.u0
        sigma = (x[0]+0.0000001) / xsh

        val = np.zeros(sigma.shape[0])
        for term in range(1, 50):
            val += 2/term/sigma * jv(term, term*sigma) * \
                   np.sin(term*self.w0*(self.t - x[0]/self.c0))

        return self.p0 * val


u_ba = Function(eqn.V)
u_ba.interpolate(Analytical(c0, f0, p0, rho0, beta, tf))

V_e = FunctionSpace(mesh, ("Lagrange", degree+3))
u_e = Function(V_e)
u_e.interpolate(Analytical(c0, f0, p0, rho0, beta, tf))

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
print("Relative L2 error of FEM solution:", L2_error_fe)

L2_error_ba = abs(np.sqrt(L2_diff_ba) / np.sqrt(L2_exact))
print("Relative L2 error of BA solution:", L2_error_ba)

# Plot solution
npts = 3 * dofs
x0 = np.linspace(0.00001, L, npts)
points = np.zeros((3, npts))
points[0] = x0
idx, x, cells = get_eval_params(mesh, points)

u_eval_fe = u.eval(x, cells).flatten()
u_eval_ba = u_ba.eval(x, cells).flatten()
u_analytic = u_e.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval_fe, x.T[0], u_analytic, 'r--')
plt.xlim([0.0, 10*lmbda])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_1d_gll_p{}_epw{}_soln1.png".format(degree, epw))

plt.xlim([L/2-5*lmbda, L/2+5*lmbda])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_1d_gll_p{}_epw{}_soln2.png".format(degree, epw))

plt.xlim([L-10*lmbda, L])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_1d_gll_p{}_epw{}_soln3.png".format(degree, epw))
plt.close()

print("L2 error (FE using array):",
      np.linalg.norm(u_eval_fe-u_analytic)/np.linalg.norm(u_analytic))

print("L2 error (BA using array):",
      np.linalg.norm(u_eval_ba-u_analytic)/np.linalg.norm(u_analytic))
