import numpy as np
from scipy.special import jv
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.generation import IntervalMesh
from dolfinx.fem import (assemble_scalar, assemble_vector, FunctionSpace,
                         Function)
from dolfinx.mesh import locate_entities_boundary, MeshTags
from ufl import FiniteElement, TestFunction, Measure, inner, grad, dx


class WesterveltGLL:
    """
    Solver for Westervelt equation

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta, beta, rho0):
        self.mesh = mesh

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = inner(self.u, self.v) * dx(metadata=md) \
            + self.delta / self.c0 * inner(self.u, self.v) \
            * ds(2, metadata=md) \
            - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n \
            * inner(self.u, self.v) * dx(metadata=md)

        self.L = self.c0**2 * (- inner(grad(self.u_n), grad(self.v))
                               * dx(metadata=md)
                               + inner(self.g, self.v)
                               * ds(1, metadata=md)
                               - 1 / self.c0*inner(self.v_n, self.v)
                               * ds(2, metadata=md)) \
            + self.delta * (- inner(grad(self.v_n), grad(self.v))
                            * dx(metadata=md)
                            + inner(self.dg, self.v)
                            * ds(1, metadata=md)) \
            + 2 * self.beta / self.rho0 / self.c0**2 \
            * inner(self.v_n*self.v_n, self.v) * dx(metadata=md)

    def init(self):
        """
        Set the inital values of u and v, i.e. u_0 and v_0
        """
        self.u_n.x.array[:] = 0.0
        self.v_n.x.array[:] = 0.0

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec):
        """
        Evaluate du/dt = f0(t, u, v)

        Parameters
        ----------
        t : Current time, i.e. tn
        u : Current u, i.e. un
        v : Current v, i.e. vn

        Return
        ------
        result : Result, i.e. dun/dtn
        """

        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec):
        """
        Evaluate dv/dt = f1(t, u, v)

        Parameters
        ----------
        t : Current time, i.e. tn
        u : Current u, i.e. un
        v : Current v, i.e. vn

        Return
        ------
        result : Result, i.e. dvn/dtn
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                np.sin(self.freq * np.pi * t / self.alpha)
        else:
            window = 1.0
            dwindow = 0.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        m = assemble_vector(self.a)
        m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(b, m)

        return result

    def rk4(self, t0: float, tf: float, dt: float):
        """
        Runge-Kutta 4th order solver

        Parameters
        ----------
        t0 : start time
        tf : final time
        dt : time step size

        Returns
        -------
        u : u at final time
        v : v at final time
        """

        # Placeholder vectors at time step n
        u_ = self.u_n.vector.copy()
        v_ = self.v_n.vector.copy()

        # Placeholder vectors at intermediate time step
        un = self.u_n.vector.copy()
        vn = self.v_n.vector.copy()

        # Placeholder vectors at start of time step
        u0 = self.u_n.vector.copy()
        v0 = self.v_n.vector.copy()

        # Placeholder at k intermediate time step
        ku = u0.copy()
        kv = v0.copy()

        # Runge-Kutta timestepping data
        n_RK = 4
        a_runge = np.array([0.0, 0.5, 0.5, 1.0])
        b_runge = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
        c_runge = np.array([0.0, 0.5, 0.5, 1.0])

        t = t0
        step = 0
        nstep = int((tf - t0) / dt) + 1
        while t < tf:
            dt = min(dt, tf-t)

            # Store solution at start of time step
            u_.copy(result=u0)
            v_.copy(result=v0)

            # Runge-Kutta step
            for i in range(n_RK):
                u0.copy(result=un)
                v0.copy(result=vn)

                un.axpy(a_runge[i]*dt, ku)
                vn.axpy(a_runge[i]*dt, kv)

                # RK time evaluation
                tn = t + c_runge[i] * dt

                # Compute slopes
                ku = self.f0(tn, un, vn, result=ku)
                kv = self.f1(tn, un, vn, result=kv)

                # Update solution
                u_.axpy(b_runge[i]*dt, ku)
                v_.axpy(b_runge[i]*dt, kv)

            # Update time
            t += dt
            step += 1

            if step % 100 == 0:
                PETSc.Sys.syncPrint("t: {},\t Steps: {}/{}".format(
                    t, step, nstep))

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.vector)
        v_.copy(result=self.v_n.vector)


# Material parameters
c0 = 1  # speed of sound (m / s)
rho0 = 1  # density of medium (kg / m^3)
beta = 0.01  # coefficient of nonlinearity

# Source parameters
f0 = 10  # source frequency (Hz)
w0 = 2 * np.pi * f0  # angular frequency (rad / s)
p0 = 1  # pressure amplitude (Pa)
u0 = p0 / rho0 / c0  # velocity amplitude (m / s)

# Domain parameters
xsh = rho0*c0**3/beta/p0/w0  # shock formation distance (m)
L = 1.0  # domain length (m)

# Physical parameters
lmbda = c0/f0  # wavelength (m)
k = 2 * np.pi / lmbda  # wavenumber (m^-1)

# FE parameters
degree = 4  # degree of basis function

# Mesh parameters
epw = 8  # number of element per wavelength
nw = L / lmbda  # number of waves
nx = int(epw * nw + 1)  # total number of elements
h = L / nx

# Generate mesh
mesh = IntervalMesh(MPI.COMM_WORLD, nx, [0, L])

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
tend = L / c0 + 16 / f0  # simulation final time (s)
tspan = [tstart, tend]

CFL = 0.9
dt = CFL * h / (c0 * degree**2)

print("Final time:", tend)

# Instantiate model
eqn = WesterveltGLL(mesh, mt, degree, c0, f0, p0, 0.0, beta, rho0)
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
    assemble_scalar(inner(diff, diff) * dx), op=MPI.SUM)
L2_exact = mesh.comm.allreduce(
    assemble_scalar(inner(u_e, u_e) * dx), op=MPI.SUM)

L2_error = abs(np.sqrt(L2_diff) / np.sqrt(L2_exact))


def test_L2_error():
    assert(L2_error < 1E-1)
