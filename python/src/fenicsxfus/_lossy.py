import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import TestFunction, TrialFunction, Measure, inner, grad, dx


class LossySpectralExplicit:
    """
    Solver for the viscoelastic wave equation.

    - GLL lattice and GLL quadrature -> diagonal mass matrix.
    - Explicit Runge-Kutta solver.

    """

    def __init__(
        self, mesh, meshtags, k, c0, rho0, delta0, freq0, p0, s0, rk_order, dt
    ):
        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.delta0 = delta0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.s0 = s0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Runge-Kutta timestepping data
        self.dt = dt

        # Forward Euler
        if rk_order == 1:
            self.n_RK = 1
            self.a_runge = np.array([0])
            self.b_runge = np.array([1])
            self.c_runge = np.array([0])

        # Ralston's 2nd order
        if rk_order == 2:
            self.n_RK = 2
            self.a_runge = np.array([0, 2 / 3])
            self.b_runge = np.array([1 / 4, 3 / 4])
            self.c_runge = np.array([0, 2 / 3])

        # Ralston's 3rd order
        if rk_order == 3:
            self.n_RK = 3
            self.a_runge = np.array([0, 1 / 2, 3 / 4])
            self.b_runge = np.array([2 / 9, 1 / 3, 4 / 9])
            self.c_runge = np.array([0, 1 / 2, 3 / 4])

        # Classical 4th order
        if rk_order == 4:
            self.n_RK = 4
            self.a_runge = np.array([0.0, 0.5, 0.5, 1.0])
            self.b_runge = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
            self.c_runge = np.array([0.0, 0.5, 0.5, 1.0])

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure("ds", subdomain_data=meshtags, domain=mesh)

        # Define cell, finite element and function space
        cell_type = mesh.ufl_cell().cellname()
        FE = basix.ufl.element(
            basix.ElementFamily.P, cell_type, k, basix.LagrangeVariant.gll_warped
        )
        V = functionspace(mesh, FE)

        # Define functions
        self.v = TestFunction(V)
        self.u = Function(V)
        self.g = Function(V)
        self.dg = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Quadrature parameters
        qd = {
            "2": 3,
            "3": 4,
            "4": 6,
            "5": 8,
            "6": 10,
            "7": 12,
            "8": 14,
            "9": 16,
            "10": 18,
        }
        md = {"quadrature_rule": "GLL", "quadrature_degree": qd[str(k)]}

        # Define forms
        self.u.x.array[:] = 1.0
        self.a = form(
            inner(self.u / self.rho0 / self.c0 / self.c0, self.v) * dx(metadata=md)
            + inner(
                self.delta0 / self.rho0 / self.c0 / self.c0 / self.c0 * self.u, self.v
            )
            * ds(2, metadata=md)
        )
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            -inner(1 / self.rho0 * grad(self.u_n), grad(self.v)) * dx(metadata=md)
            + inner(1 / self.rho0 * self.g, self.v) * ds(1, metadata=md)
            - inner(1 / self.rho0 / self.c0 * self.v_n, self.v) * ds(2, metadata=md)
            - inner(
                self.delta0 / self.rho0 / self.c0 / self.c0 * grad(self.v_n),
                grad(self.v),
            )
            * dx(metadata=md)
            + inner(self.delta0 / self.rho0 / self.c0 / self.c0 * self.dg, self.v)
            * ds(1, metadata=md)
        )
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

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
        result : Result, i.e. k^{u}
        """

        v.copy(result=result)

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
        result : Result, i.e. k^{v}
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            dwindow = (
                0.5
                * np.pi
                * self.freq
                / self.alpha
                * np.sin(self.freq * np.pi * t / self.alpha)
            )
        else:
            window = 1.0
            dwindow = 0.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.s0 * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.s0 * np.cos(
            self.w0 * t
        ) - window * self.p0 * self.w0**2 / self.s0 * np.sin(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.x.petsc_vec)
        self.u_n.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        v.copy(result=self.v_n.x.petsc_vec)
        self.v_n.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(self.b, self.m)

    def rk(self, t0: float, tf: float):
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
        t : final time
        step : number of RK steps
        """

        # Runge-Kutta data
        n_RK = self.n_RK
        a_runge = self.a_runge
        b_runge = self.b_runge
        c_runge = self.c_runge

        # Placeholder vectors at time step n
        u_ = self.u_n.x.petsc_vec.copy()
        v_ = self.v_n.x.petsc_vec.copy()

        # Placeholder vectors at intermediate time step
        un = self.u_n.x.petsc_vec.copy()
        vn = self.v_n.x.petsc_vec.copy()

        # Placeholder vectors at start of time step
        u0 = self.u_n.x.petsc_vec.copy()
        v0 = self.v_n.x.petsc_vec.copy()

        # Placeholder at k intermediate time step
        ku = u0.copy()
        kv = v0.copy()

        # Temporal data
        dt = self.dt
        t = t0
        step = 0
        nstep = int((tf - t0) / dt) + 1

        while t < tf:
            dt = min(dt, tf - t)

            # Store solution at start of time step
            u_.copy(result=u0)
            v_.copy(result=v0)

            # Runge-Kutta step
            for i in range(n_RK):
                u0.copy(result=un)
                v0.copy(result=vn)

                un.axpy(a_runge[i] * dt, ku)
                vn.axpy(a_runge[i] * dt, kv)

                # RK time evaluation
                tn = t + c_runge[i] * dt

                # Compute slopes
                self.f1(tn, un, vn, result=kv)
                self.f0(tn, un, vn, result=ku)

                # Update solution
                u_.axpy(b_runge[i] * dt, ku)
                v_.axpy(b_runge[i] * dt, kv)

            # Update time
            t += dt
            step += 1

            if step % 100 == 0:
                PETSc.Sys.syncPrint(f"t: {t:5.5},\t Steps: {step}/{nstep}")

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.x.petsc_vec)
        v_.copy(result=self.v_n.x.petsc_vec)

        return self.u_n, self.v_n, t


class LossySpectralImplicit:
    """
    Solver for the viscoelastic wave equation.

    - GLL lattice and GLL quadrature.
    - Singly diagonal implicit Runge-Kutta solver.

    """

    def __init__(
        self, mesh, meshtags, k, c0, rho0, delta0, freq0, p0, s0, rk_order, dt
    ):
        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.delta0 = delta0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.s0 = s0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Runge-Kutta timestepping data
        self.dt = dt

        # Backward Euler
        if rk_order == 1:
            self.n_RK = 1
            self.a_runge = np.array([[1]])
            self.b_runge = np.array([1])
            self.c_runge = np.array([1])

        # Crouzeix 2 stages
        if rk_order == 2:
            self.n_RK = 2
            self.a_runge = np.array([[1 / 4, 0], [1 / 2, 1 / 4]])
            self.b_runge = np.array([1 / 2, 1 / 2])
            self.c_runge = np.array([1 / 4, 3 / 4])

        # Crouzeix 3 stages
        if rk_order == 3:
            q = 2 * np.cos(np.pi / 18) / np.sqrt(3)
            self.n_RK = 3
            self.a_runge = np.array(
                [
                    [(1 + q) / 2, 0, 0],
                    [-q / 2, (1 + q) / 2, 0],
                    [1 + q, -(1 + 2 * q), (1 + q) / 2],
                ]
            )
            self.b_runge = np.array(
                [1 / (6 * q**2), 1 - 1 / (3 * q**2), 1 / (6 * q**2)]
            )
            self.c_runge = np.array([(1 + q) / 2, 1 / 2, (1 - q) / 2])

        # 4 stages
        if rk_order == 4:
            self.n_RK = 4
            self.a_runge = np.array(
                [
                    [1 / 2, 0, 0, 0],
                    [1 / 6, 1 / 2, 0, 0],
                    [-1 / 2, 1 / 2, 1 / 2, 0],
                    [3 / 2, -3 / 2, 1 / 2, 1 / 2],
                ]
            )
            self.b_runge = np.array([3 / 2, -3 / 2, 1 / 2, 1 / 2])
            self.c_runge = np.array([1 / 2, 2 / 3, 1 / 2, 1])

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure("ds", subdomain_data=meshtags, domain=mesh)

        # Define cell, finite element and function space
        cell_type = mesh.ufl_cell().cellname()
        FE = basix.ufl.element(
            basix.ElementFamily.P, cell_type, k, basix.LagrangeVariant.gll_warped
        )
        V = functionspace(mesh, FE)

        # Define functions
        self.v = TestFunction(V)
        self.u = TrialFunction(V)
        self.g = Function(V)
        self.dg = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)
        self.tau = self.dt * self.a_runge[0, 0]

        # Quadrature parameters
        qd = {
            "2": 3,
            "3": 4,
            "4": 6,
            "5": 8,
            "6": 10,
            "7": 12,
            "8": 14,
            "9": 16,
            "10": 18,
        }
        md = {"quadrature_rule": "GLL", "quadrature_degree": qd[str(k)]}

        # Define forms
        self.a = form(
            inner(self.u / self.rho0 / self.c0 / self.c0, self.v) * dx(metadata=md)
            + inner(
                self.delta0 / self.rho0 / self.c0 / self.c0 / self.c0 * self.u, self.v
            )
            * ds(2, metadata=md)
            + inner(self.tau * self.tau / self.rho0 * grad(self.u), grad(self.v))
            * dx(metadata=md)
            + inner(
                self.tau * self.delta0 / self.rho0 / self.c0 / self.c0 * grad(self.u),
                grad(self.v),
            )
            * dx(metadata=md)
            + inner(self.tau / self.rho0 / self.c0 * self.u, self.v)
            * ds(2, metadata=md)
        )
        self.A = assemble_matrix(self.a)
        self.A.assemble()

        self.L = form(
            -inner(1 / self.rho0 * grad(self.u_n), grad(self.v)) * dx(metadata=md)
            + inner(1 / self.rho0 * self.g, self.v) * ds(1, metadata=md)
            - inner(1 / self.rho0 / self.c0 * self.v_n, self.v) * ds(2, metadata=md)
            - inner(
                self.delta0 / self.rho0 / self.c0 / self.c0 * grad(self.v_n),
                grad(self.v),
            )
            * dx(metadata=md)
            + inner(self.delta0 / self.rho0 / self.c0 / self.c0 * self.dg, self.v)
            * ds(1, metadata=md)
            - inner(self.tau / self.rho0 * grad(self.v_n), grad(self.v))
            * dx(metadata=md)
        )
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Linear solver
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setType(PETSc.KSP.Type.CG)
        self.solver.getPC().setType(PETSc.PC.Type.JACOBI)
        self.solver.setOperators(self.A)

    def init(self):
        """
        Set the initial values of u and v, i.e. u_0 and v_0
        """

        self.u_n.x.array[:] = 0.0
        self.v_n.x.array[:] = 0.0

    def f0(
        self,
        t: float,
        u: PETSc.Vec,
        v: PETSc.Vec,
        ku: PETSc.Vec,
        kv: PETSc.Vec,
        result: PETSc.Vec,
    ):
        """
        Evaluate du/dt = f0(t, u, v)

        Parameters
        ----------
        t : Current time, i.e. tn
        u : Current u, i.e. un
        v : Current v, i.e. vn

        Return
        ------
        result : Result, i.e. k^{u}
        """

        result.waxpy(self.tau, kv, v)

    def f1(
        self,
        t: float,
        u: PETSc.Vec,
        v: PETSc.Vec,
        ku: PETSc.Vec,
        kv: PETSc.Vec,
        result: PETSc.Vec,
    ):
        """
        Evaluate dv/dt = f1(t, u, v)

        Parameters
        ----------
        t : Current time, i.e. tn
        u : Current u, i.e. un
        v : Current v, i.e. vn

        Return
        ------
        result : Result, i.e. k^{v}
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            dwindow = (
                0.5
                * np.pi
                * self.freq
                / self.alpha
                * np.sin(self.freq * np.pi * t / self.alpha)
            )
        else:
            window = 1.0
            dwindow = 0.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.s0 * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.s0 * np.cos(
            self.w0 * t
        ) - window * self.p0 * self.w0**2 / self.s0 * np.sin(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.x.petsc_vec)
        self.u_n.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        v.copy(result=self.v_n.x.petsc_vec)
        self.v_n.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(self.b, result)

    def dirk(self, t0: float, tf: float):
        """
        Diagonally implicit Runge-Kutta solver

        Parameters
        ----------
        t0: start time
        tf: final time
        dt: time step size

        Returns
        -------
        u : u at final time
        v : v at final time
        t : final time
        """

        # Runge-Kutta data
        n_RK = self.n_RK
        a_runge = self.a_runge
        b_runge = self.b_runge
        c_runge = self.c_runge

        # Placeholder vectors at time step n
        u_ = self.u_n.x.petsc_vec.copy()
        v_ = self.v_n.x.petsc_vec.copy()

        # Placeholder vectors at intermediate time step
        un = self.u_n.x.petsc_vec.copy()
        vn = self.v_n.x.petsc_vec.copy()

        # Placeholder vectors at start of time step
        u0 = self.u_n.x.petsc_vec.copy()
        v0 = self.v_n.x.petsc_vec.copy()

        # Placeholder at k intermediate time step
        ku = n_RK * [u0.copy()]
        kv = n_RK * [v0.copy()]

        # Temporal data
        dt = self.dt
        t = t0
        step = 0
        nstep = int((tf - t0) / dt) + 1

        while t < tf:
            dt = min(dt, tf - t)

            # Store solution at start of time step
            u_.copy(result=u0)
            v_.copy(result=v0)

            # Runge-Kutta step
            for i in range(n_RK):
                u0.copy(result=un)
                v0.copy(result=vn)

                for j in range(i):
                    un.axpy(a_runge[i, j] * dt, ku[j])
                    vn.axpy(a_runge[i, j] * dt, kv[j])

                # RK time evaluation
                tn = t + c_runge[i] * dt

                # Solve for slopes
                self.f1(tn, un, vn, ku[i], kv[i], result=kv[i])
                self.f0(tn, un, vn, ku[i], kv[i], result=ku[i])

                # Update solution
                u_.axpy(b_runge[i] * dt, ku[i])
                v_.axpy(b_runge[i] * dt, kv[i])

            # Update time
            t += dt
            step += 1

            if step % 100 == 0:
                PETSc.Sys.syncPrint(f"t: {t:5.5},\t Steps: {step}/{nstep}")

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.x.petsc_vec)
        v_.copy(result=self.v_n.x.petsc_vec)

        return self.u_n, self.v_n, t
