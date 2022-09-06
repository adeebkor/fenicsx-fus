import pytest
import numpy as np
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import create_interval, locate_entities_boundary, meshtags
from ufl import inner, dx

from hifusim import LinearGLLSciPy


@pytest.mark.parametrize("degree, epw", [(3, 16), (4, 8), (5, 4), (6, 4)])
def test_linear_scipy_L2(degree, epw):
    # Material parameters
    c0 = 1  # speed of sound (m/s)
    rho0 = 1  # density of medium (kg / m^3)

    # Source parameters
    f0 = 10  # source frequency (Hz)
    u0 = 1  # velocity amplitude (m / s)
    p0 = rho0*c0*u0  # pressure amplitude (Pa)

    # Domain parameters
    L = 1.0  # domain length (m)

    # Physical parameters
    lmbda = c0/f0  # wavelength (m)

    # Mesh parameters
    nw = L / lmbda  # number of waves
    nx = int(epw * nw + 1)  # total number of elements

    # Generate mesh
    mesh = create_interval(MPI.COMM_WORLD, nx, [0, L])

    # Tag boundaries
    tdim = mesh.topology.dim

    facets0 = locate_entities_boundary(
        mesh, tdim-1, lambda x: x[0] < np.finfo(float).eps)
    facets1 = locate_entities_boundary(
        mesh, tdim-1, lambda x: x[0] > L - np.finfo(float).eps)

    indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
    values = np.hstack((np.full(facets0.shape, 1, np.intc),
                        np.full(facets1.shape, 2, np.intc)))
    mt = meshtags(mesh, tdim-1, indices, values[pos])

    # Temporal parameters
    tstart = 0.0  # simulation start time (s)
    tend = L / c0 + 16 / f0  # simulation final time (s)

    # Instantiate model
    eqn = LinearGLLSciPy(mesh, mt, degree, c0, f0, p0)
    eqn.alpha = 4
    print("Degree of freedoms: ", eqn.V.dofmap.index_map.size_global)

    # Solve
    eqn.init()
    eqn.rk(tstart, tend)

    class Analytical:
        """ Analytical solution """

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

    V_e = FunctionSpace(mesh, ("Lagrange", degree+3))
    u_e = Function(V_e)
    u_e.interpolate(Analytical(c0, f0, p0, tend))

    # L2 error
    diff = eqn.u_n - u_e
    L2_diff = mesh.comm.allreduce(
        assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
    L2_exact = mesh.comm.allreduce(
        assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM)

    L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

    assert (L2_error < 1E-3)
