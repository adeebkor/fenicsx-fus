import numpy as np
from scipy.integrate import RK45
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (FunctionSpace, Function, assemble_vector, form,
                         assemble_matrix)
from ufl import (FiniteElement, TestFunction, TrialFunction, Measure, inner,
                 grad, dx)


class Westervelt:
    """
    FE solver for the Westervelt equation.

    This solver uses GLL points and Gauss quadrature.

    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta, beta, rho0):
        self.mesh = mesh

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = TrialFunction(self.V)
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

        # Define variational form
        self.a = form(inner(self.u, self.v) * dx
                      + self.delta / self.c0 * inner(self.u, self.v) * ds(2)
                      - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n
                      * inner(self.u, self.v) * dx)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = form(self.c0**2 * (- inner(grad(self.u_n), grad(self.v)) * dx
                                    + inner(self.g, self.v) * ds(1)
                                    - 1 / self.c0*inner(self.v_n, self.v)
                                    * ds(2))
                      + self.delta * (- inner(grad(self.v_n), grad(self.v))
                                      * dx
                                      + inner(self.dg, self.v)
                                      * ds(1))
                      + 2 * self.beta / self.rho0 / self.c0**2
                      * inner(self.v_n*self.v_n, self.v) * dx)
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Linear solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.M)

    def init(self):
        """
        Set the initial values of u and v, i.e. u_0 and v_0
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
        self.M.zeroEntries()
        assemble_matrix(self.M, self.a)
        self.M.assemble()

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(self.b, result)

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
        t : final time
        step : number of RK steps
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

        return self.u_n, self.v_n, t, step


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
        self.a = form(inner(self.u, self.v) * dx(metadata=md)
                      + self.delta / self.c0 * inner(self.u, self.v)
                      * ds(2, metadata=md)
                      - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n
                      * inner(self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(self.c0**2 * (- inner(grad(self.u_n), grad(self.v))
                                    * dx(metadata=md)
                                    + inner(self.g, self.v)
                                    * ds(1, metadata=md)
                                    - 1 / self.c0*inner(self.v_n, self.v)
                                    * ds(2, metadata=md))
                      + self.delta * (- inner(grad(self.v_n), grad(self.v))
                                      * dx(metadata=md)
                                      + inner(self.dg, self.v)
                                      * ds(1, metadata=md))
                      + 2 * self.beta / self.rho0 / self.c0**2
                      * inner(self.v_n*self.v_n, self.v) * dx(metadata=md))
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

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
        with self.m.localForm() as m_local:
            m_local.set(0.0)
        assemble_vector(self.m, self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(self.b, self.m)

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

        return self.u_n, self.v_n, t, step


class WesterveltGLLSciPy:
    """
    Solver for Westervelt equation

    This solver uses GLL lattice and GLL quadrature. It is design to use
    SciPy Runge-Kutta solver. Currently, the solver only works in serial.
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

        # Get degrees of freedom
        self.ndof = self.V.dofmap.index_map.size_global

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
        self.a = form(inner(self.u, self.v) * dx(metadata=md)
                      + self.delta / self.c0 * inner(self.u, self.v)
                      * ds(2, metadata=md)
                      - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n
                      * inner(self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(self.c0**2 * (- inner(grad(self.u_n), grad(self.v))
                                    * dx(metadata=md)
                                    + inner(self.g, self.v)
                                    * ds(1, metadata=md)
                                    - 1 / self.c0*inner(self.v_n, self.v)
                                    * ds(2, metadata=md))
                      + self.delta * (- inner(grad(self.v_n), grad(self.v))
                                      * dx(metadata=md)
                                      + inner(self.dg, self.v)
                                      * ds(1, metadata=md))
                      + 2 * self.beta / self.rho0 / self.c0**2
                      * inner(self.v_n*self.v_n, self.v) * dx(metadata=md))
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

    def init(self):
        """
        Set the inital values of u and v, i.e. u_0 and v_0
        """
        self.u_n.x.array[:] = 0.0
        self.v_n.x.array[:] = 0.0

    def f(self, t: float, y: np.ndarray):
        """
        The right-hand side function of the system of ODEs.

        Parameters:
        -----------
        t : current time
        y : current solution vector

        Return:
        -------
        ystep : solution vector at next time-step

        """

        # Compute window
        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                np.sin(self.freq * np.pi * t / self.alpha)
        else:
            window = 1.0
            dwindow = 0.0

        # Update source
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields
        self.u_n.x.array[:] = y[:self.ndof]
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        self.v_n.x.array[:] = y[self.ndof:]
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        with self.m.localForm() as m_local:
            m_local.set(0.0)
        assemble_vector(self.m, self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result = self.b.array[:] / self.m.array[:]

        # Full solution
        ynext = np.concatenate([y[self.ndof:], result])

        return ynext

    def rk(self, t0: float, tf: float):
        """
        RK solver, uses SciPy RK45 adaptive time-step solver.

        Parameters:
        -----------
        t0 : start time
        tf : final time

        Returns
        -------
        u : u at final time
        v : v at final time
        t : final time
        step : number of RK steps
        """

        # Create a placeholder vector for initial condition
        u0 = self.u_n.vector.copy()
        v0 = self.v_n.vector.copy()
        y0 = np.concatenate([u0, v0])

        # Instantiate RK45 solver
        model = RK45(self.f, t0, y0, tf, atol=1e-9, rtol=1e-9)

        # RK solve
        step = 0
        while model.t < tf:
            model.step()
            step += 1
            if step % 100 == 0:
                PETSc.Sys.syncPrint("t: {}".format(model.t))

        # Solution at final time
        self.u_n.x.array[:] = model.y[:self.ndof]
        self.v_n.x.array[:] = model.y[self.ndof:]

        return self.u_n, self.v_n, model.t, step
