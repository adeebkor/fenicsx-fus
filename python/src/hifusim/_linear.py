import numpy as np
from scipy.integrate import RK45
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import FunctionSpace, Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import locate_entities
from ufl import (FiniteElement, TestFunction, TrialFunction, Measure, inner,
                 grad, dx)


class LinearGLL:
    """
    Solver for linear second order wave equation.

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0):
        self.mesh = mesh

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # JIT compilation parameters
        jit_params = {"cffi_extra_compile_args": ["-Ofast", "-march=native"]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u, self.v) * dx(metadata=md),
                      jit_params=jit_params)
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(self.c0**2*(- inner(grad(self.u_n), grad(self.v))
                                  * dx(metadata=md)
                                  + inner(self.g, self.v)
                                  * ds(1, metadata=md)
                                  - 1/self.c0*inner(self.v_n, self.v)
                                  * ds(2, metadata=md)),
                      jit_params=jit_params)
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
        result : Result, i.e. dvn/dtn
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
        else:
            window = 1.0

        # Update source
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(self.b, self.m)

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
                self.f0(tn, un, vn, result=ku)
                self.f1(tn, un, vn, result=kv)

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


class LinearGLLSciPy:
    """
    Solver for linear second order wave equation.

    This solver uses GLL lattice and GLL quadrature. It is design to use
    SciPy Runge-Kutta solver. Currently, the solver only works in serial.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0):
        self.mesh = mesh

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
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
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(self.c0**2*(- inner(grad(self.u_n), grad(self.v))
                                  * dx(metadata=md)
                                  + inner(self.g, self.v)
                                  * ds(1, metadata=md)
                                  - 1/self.c0 * inner(self.v_n, self.v)
                                  * ds(2, metadata=md)))
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
        else:
            window = 1.0

        # Update source
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields
        self.u_n.x.array[:] = y[:self.ndof]
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        self.v_n.x.array[:] = y[self.ndof:]
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

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


class Linear:
    """
    FE solver for the second order linear wave equation.

    This solver uses GLL lattice and Gauss quadrature.

    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0):
        self.mesh = mesh

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = TrialFunction(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Define variational form
        self.a = form(inner(self.u, self.v) * dx)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = form(self.c0**2*(- inner(grad(self.u_n), grad(self.v)) * dx
                                  + inner(self.g, self.v) * ds(1)
                      - 1/self.c0*inner(self.v_n, self.v) * ds(2)))
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
        result : Result, i.e. dvn/dtn
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
        else:
            window = 1.0

        # Update source
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(self.b, result)

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
                self.f0(tn, un, vn, result=ku)
                self.f1(tn, un, vn, result=kv)

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


class LinearInhomogenousGLL:
    """
    Solver for linear second order wave equation in a homogenous domain.

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0, ref_idx):
        self.mesh = mesh
        tdim = mesh.topology.dim

        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant="gll")
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Refractive index function
        V_DG = FunctionSpace(mesh, ("DG", 0))
        self.coeff = Function(V_DG)
        length = 1
        cells_0 = locate_entities(
            mesh, tdim, lambda x: np.logical_and(x[0] >= length / 3,
                                                 x[0] <= 2*length / 3))

        self.coeff.x.array[:] = c0
        self.coeff.x.array[cells_0] = np.full(len(cells_0), ref_idx*c0)

        self.L = form(
            - inner(self.coeff * self.coeff * grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.coeff * self.coeff * self.g, self.v)
            * ds(1, metadata=md)
            - inner(self.coeff * self.v_n, self.v)
            * ds(2, metadata=md))
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
        result : Result, i.e. dvn/dtn
        """

        if t < self.T * self.alpha:
            window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
        else:
            window = 1.0

        # Update source
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(self.b, self.m)

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
                self.f0(tn, un, vn, result=ku)
                self.f1(tn, un, vn, result=kv)

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
