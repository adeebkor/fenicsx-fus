import pytest
import numpy as np
from scipy.special import jv
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, Function, assemble_scalar, form
from dolfinx.mesh import create_interval, locate_entities_boundary, meshtags
from ufl import inner, dx

from hifusim import Westervelt


@pytest.mark.parametrize("degree, epw", [(3, 16), (4, 8), (5, 4), (6, 2)])
def test_westervelt_L2(degree, epw):
    # Material parameters
    c0 = 1  # speed of sound (m / s)
    rho0 = 1  # density of medium (kg / m^3)
    beta = 0.01  # coefficient of nonlinearity

    # Source parameters
    f0 = 10  # source frequency (Hz)
    p0 = 1  # pressure amplitude (Pa)

    # Domain parameters
    L = 1.0  # domain length (m)

    # Physical parameters
    lmbda = c0/f0  # wavelength (m)

    # Mesh parameters
    nw = L / lmbda  # number of waves
    nx = int(epw * nw + 1)  # total number of elements
    h = L / nx

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
    tend = L / c0 + 8 / f0  # simulation final time (s)

    CFL = 0.9
    dt = CFL * h / (c0 * degree**2)

    print("Final time:", tend)

    # Instantiate model
    eqn = Westervelt(mesh, mt, degree, c0, f0, p0, 0.0, beta, rho0)
    eqn.alpha = 4
    print("Degree of freedoms:", eqn.V.dofmap.index_map.size_global)

    # Solve
    eqn.init()
    eqn.rk4(tstart, tend, dt)

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

    V_e = FunctionSpace(mesh, ("Lagrange", degree+3))
    u_e = Function(V_e)
    u_e.interpolate(Analytical(c0, f0, p0, rho0, beta, tend))

    # L2 error
    diff = eqn.u_n - u_e
    L2_diff = mesh.comm.allreduce(
        assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
    L2_exact = mesh.comm.allreduce(
        assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM)

    L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

    assert(L2_error < 1E-1)
