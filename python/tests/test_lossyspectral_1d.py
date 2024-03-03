import pytest
import numpy as np
from mpi4py import MPI

from dolfinx.fem import functionspace, Function, assemble_scalar, form
from dolfinx.mesh import create_interval, locate_entities_boundary, meshtags
from ufl import inner, dx

from fenicsxfus import LossySpectralExplicit, LossySpectralImplicit
from fenicsxfus.utils import compute_diffusivity_of_sound


@pytest.mark.parametrize("degree, epw", [(3, 8), (4, 4), (5, 2), (6, 2)])
def test_lossyspectral_explicit(degree, epw):
    # Source parameters
    f0 = 10  # source frequency (Hz)
    w0 = 2 * np.pi * f0  # angular frequency (rad/s)
    u0 = 1  # velocity amplitude (m/s)

    # Material parameters
    c0 = 1  # speed of sound (m/s)
    rho0 = 4  # density of medium (kg/m^3)
    alphadB = 5  # (dB/m)
    alphaNp = alphadB / 20 * np.log(10)  # (Np/m/MHz^2)
    delta0 = compute_diffusivity_of_sound(w0, c0, alphadB)

    # Domain parameters
    L = 1.0  # domain length (m)

    # Physical parameters
    p0 = rho0 * c0 * u0  # pressure amplitude (Pa)
    lmbda = c0 / f0  # wavelength (m)

    # Mesh parameters
    nw = L / lmbda  # number of waves
    nx = int(epw * nw + 1)  # total number of elements
    h = L / nx

    # Generate mesh
    mesh = create_interval(MPI.COMM_WORLD, nx, [0, L])

    # Tag boundaries
    tdim = mesh.topology.dim

    facets0 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: x[0] < np.finfo(float).eps
    )
    facets1 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: x[0] > L - np.finfo(float).eps
    )

    indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
    values = np.hstack(
        (np.full(facets0.shape, 1, np.intc), np.full(facets1.shape, 2, np.intc))
    )
    mt = meshtags(mesh, tdim - 1, indices, values[pos])

    # Define DG function for physical parameters
    V_DG = functionspace(mesh, ("DG", 0))
    c = Function(V_DG)
    c.x.array[:] = c0

    rho = Function(V_DG)
    rho.x.array[:] = rho0

    delta = Function(V_DG)
    delta.x.array[:] = delta0

    # Temporal parameters
    tstart = 0.0  # simulation start time (s)
    tend = L / c0 + 16 / f0  # simulation final time (s)

    CFL = 0.5
    dt = CFL * h / (c0 * degree**2)

    # Instantiate model
    eqn = LossySpectralExplicit(mesh, mt, degree, c, rho, delta, f0, p0, c0, 4, dt)

    # Solve
    eqn.init()
    u_n, _, tf = eqn.rk(tstart, tend)

    class Analytical:
        """Analytical solution"""

        def __init__(self, c0, a0, f0, p0, t):
            self.c0 = c0
            self.a0 = a0
            self.p0 = p0
            self.f0 = f0
            self.w0 = 2 * np.pi * f0
            self.t = t

        def __call__(self, x):
            val = (
                self.p0
                * np.exp(1j * (self.w0 * self.t - self.w0 / self.c0 * x[0]))
                * np.exp(-self.a0 * x[0])
            )

            return val.imag

    V_e = functionspace(mesh, ("Lagrange", degree + 3))
    u_e = Function(V_e)
    u_e.interpolate(Analytical(c0, alphaNp, f0, p0, tf))

    # L2 error
    diff = u_n - u_e
    L2_diff = mesh.comm.allreduce(
        assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM
    )
    L2_exact = mesh.comm.allreduce(
        assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM
    )

    L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

    assert L2_error < 1e-2


@pytest.mark.parametrize("degree, epw", [(3, 8), (4, 4), (5, 2), (6, 2)])
def test_lossyspectral_implicit(degree, epw):
    # Source parameters
    f0 = 10  # source frequency (Hz)
    w0 = 2 * np.pi * f0  # angular frequency (rad/s)
    u0 = 1  # velocity amplitude (m/s)

    # Material parameters
    c0 = 1  # speed of sound (m/s)
    rho0 = 4  # density of medium (kg/m^3)
    alphadB = 5  # (dB/m)
    alphaNp = alphadB / 20 * np.log(10)  # (Np/m/MHz^2)
    delta0 = compute_diffusivity_of_sound(w0, c0, alphadB)

    # Domain parameters
    L = 1.0  # domain length (m)

    # Physical parameters
    p0 = rho0 * c0 * u0  # pressure amplitude (Pa)
    lmbda = c0 / f0  # wavelength (m)

    # Mesh parameters
    nw = L / lmbda  # number of waves
    nx = int(epw * nw + 1)  # total number of elements
    h = L / nx

    # Generate mesh
    mesh = create_interval(MPI.COMM_WORLD, nx, [0, L])

    # Tag boundaries
    tdim = mesh.topology.dim

    facets0 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: x[0] < np.finfo(float).eps
    )
    facets1 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: x[0] > L - np.finfo(float).eps
    )

    indices, pos = np.unique(np.hstack((facets0, facets1)), return_index=True)
    values = np.hstack(
        (np.full(facets0.shape, 1, np.intc), np.full(facets1.shape, 2, np.intc))
    )
    mt = meshtags(mesh, tdim - 1, indices, values[pos])

    # Define DG function for physical parameters
    V_DG = functionspace(mesh, ("DG", 0))
    c = Function(V_DG)
    c.x.array[:] = c0

    rho = Function(V_DG)
    rho.x.array[:] = rho0

    delta = Function(V_DG)
    delta.x.array[:] = delta0

    # Temporal parameters
    tstart = 0.0  # simulation start time (s)
    tend = L / c0 + 16 / f0  # simulation final time (s)

    CFL = 0.5
    dt = CFL * h / (c0 * degree**2)

    # Instantiate model
    eqn = LossySpectralImplicit(mesh, mt, degree, c, rho, delta, f0, p0, c0, 4, dt)

    # Solve
    eqn.init()
    u_n, _, tf = eqn.dirk(tstart, tend)

    class Analytical:
        """Analytical solution"""

        def __init__(self, c0, a0, f0, p0, t):
            self.c0 = c0
            self.a0 = a0
            self.p0 = p0
            self.f0 = f0
            self.w0 = 2 * np.pi * f0
            self.t = t

        def __call__(self, x):
            val = (
                self.p0
                * np.exp(1j * (self.w0 * self.t - self.w0 / self.c0 * x[0]))
                * np.exp(-self.a0 * x[0])
            )

            return val.imag

    V_e = functionspace(mesh, ("Lagrange", degree + 3))
    u_e = Function(V_e)
    u_e.interpolate(Analytical(c0, alphaNp, f0, p0, tf))

    # L2 error
    diff = u_n - u_e
    L2_diff = mesh.comm.allreduce(
        assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM
    )
    L2_exact = mesh.comm.allreduce(
        assemble_scalar(form(inner(u_e, u_e) * dx)), op=MPI.SUM
    )

    L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))

    assert L2_error < 1e-2
