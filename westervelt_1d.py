import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from mpi4py import MPI

from dolfinx import IntervalMesh, FunctionSpace, Function
from dolfinx.fem import assemble_scalar
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import inner, dx

from utils import get_eval_params
from models import Westervelt1D
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
epw = 16  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx

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
eqn = Westervelt1D(mesh, mt, degree, c0, f0, p0, 1E-10, beta, rho0)
print("Degree of freedoms:", eqn.V.dofmap.index_map.size_global)

# Solve
u, tf = solve2(eqn.f0, eqn.f1, *eqn.init(), dt, nstep, 4)
print("tf:", tf)


# Calculate L2
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


V_e = FunctionSpace(mesh, ("Lagrange", degree+2))
u_e = Function(V_e)
u_e.interpolate(Analytical(c0, f0, p0, rho0, beta, tf))

# L2 error
diff = u - u_e
L2_diff = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(diff, diff) * dx), op=MPI.SUM)
L2_exact = mesh.mpi_comm().allreduce(
    assemble_scalar(inner(u_e, u_e) * dx), op=MPI.SUM)
print("Relative L2 error of FEM solution:",
      abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))

# Plot solution
npts = 3 * degree * (nx+1)
x0 = np.linspace(0.00001, L, npts)
points = np.zeros((3, npts))
points[0] = x0
idx, x, cells = get_eval_params(mesh, points)

u_eval = u.eval(x, cells).flatten()
u_analytic = u_e.eval(x, cells).flatten()

plt.plot(x.T[0], u_eval, x.T[0], u_analytic, 'r--')
plt.xlim([0.0, 10*lmbda])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_soln1_n{}_p{}.png".format(nx, degree))

plt.xlim([L/2-5*lmbda, L/2+5*lmbda])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_soln2_n{}_p{}.png".format(nx, degree))

plt.xlim([L-10*lmbda, L])
plt.legend(["FEM", "Analytical"])
plt.savefig("plots/westervelt_soln3_n{}_p{}.png".format(nx, degree))
plt.close()

print("Relative L2 error of FEM solution (data points):",
      np.linalg.norm(u_eval-u_analytic)/np.linalg.norm(u_analytic))
