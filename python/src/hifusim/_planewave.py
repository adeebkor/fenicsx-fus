import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl_wrapper
from dolfinx.fem import FunctionSpace, Function, form
from dolfinx.fem.petsc import assemble_vector
from ufl import FacetNormal, TestFunction, Measure, inner, grad, dx


class PlanewaveGLL:
    """
    Solver for linear second order wave equation for planewave problem.

    - The solver solves for the scattered field.
    - This solver uses GLL lattice and GLL quadrature.

    Note:
        - mainly use for test purposes.

    """

    def __init__(self, mesh, meshtags, k, c0, rho0, freq0, p0):

        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        self.mesh = mesh
        self.surface_normal = FacetNormal(mesh)

        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)
        self.v = TestFunction(V)
        self.u = Function(V)
        self.u_i = Function(V)
        self.d2u_i = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # JIT compilation parameters
        jit_params = {"cffi_extra_compile_args": ["-Ofast", "-march=native"]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(1 / self.rho0 / self.c0 / self.c0
                      * inner(self.u, self.v) * dx(metadata=md),
                      jit_params=jit_params)
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(- 1 / self.rho0 * inner(grad(self.u_n), grad(self.v))
                      * dx(metadata=md)
                      + 1 / self.rho0 / self.c0 * inner(self.v_n, self.v)
                      * ds(1, metadata=md)
                      - 1 / self.rho0 / self.c0 * inner(self.v_n, self.v)
                      * ds(2, metadata=md)
                      - 1 / self.rho0 / self.c0 / self.c0
                      * inner(self.d2u_i, self.v)
                      * dx(metadata=md)
                      - 1 / self.rho0 * inner(grad(self.u_i), grad(self.v))
                      * dx(metadata=md)
                      + 1 / self.rho0 * inner(inner(
                            grad(self.u_i), self.surface_normal), self.v)
                      * ds(metadata=md),
                      jit_params=jit_params)
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

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

        # Update incident field
        self.u_i.interpolate(lambda x: self.p0 * np.sin(
                                self.w0 / self.c0 * (x[0] - self.c0 * t)))
        self.d2u_i.interpolate(
            lambda x: - self.p0 * self.w0 * self.w0 * np.sin(
                self.w0 / self.c0 * (x[0] - self.c0 * t)))

        # Update solution field
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

        return self.u_n, self.v_n, t


class PlanewaveHeterogenousGLL:
    """
    Solver for the linear second order wave equation for planewave problem.

    - The solver solves for the scattered field.
    - This solver uses GLL lattice and GLL quadrature.

    Note:
        - mainly use for test purposes.

    """

    def __init__(self, mesh, meshtags, k, c0, c1, rho0, rho1, freq0, p0):

        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        self.mesh = mesh
        self.surface_normal = FacetNormal(mesh)

        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)

        V = FunctionSpace(mesh, FE)
        self.v = TestFunction(V)
        self.u = Function(V)
        self.u_i = Function(V)
        self.d2u_i = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags[1], domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.c1 = c1
        self.rho0 = rho0
        self.rho1 = rho1
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period

        # Define function for different mediums
        V_DG = FunctionSpace(mesh, ("DG", 0))
        self.c = Function(V_DG)
        self.c.x.array[:] = c0
        self.c.x.array[meshtags[0].find(2)] = c1

        self.rho = Function(V_DG)
        self.rho.x.array[:] = rho0
        self.rho.x.array[meshtags[0].find(2)] = rho1

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u / self.rho / self.c / self.c, self.v)
                      * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(- inner(1 / self.rho * grad(self.u_n), grad(self.v))
                      * dx(metadata=md)
                      - inner(1 / self.rho / self.c * self.v_n, self.v)
                      * ds(1, metadata=md)
                      - inner(1 / self.rho / self.c * self.v_n, self.v)
                      * ds(2, metadata=md)
                      - 1 / self.rho0 / self.c0 / self.c0
                      * inner(self.d2u_i, self.v)
                      * dx(metadata=md)
                      - 1 / self.rho0 * inner(grad(self.u_i), grad(self.v))
                      * dx(metadata=md)
                      + 1 / self.rho0 * inner(inner(
                            grad(self.u_i), self.surface_normal), self.v)
                      * ds(metadata=md))
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

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

        # Update incident field
        self.u_i.interpolate(lambda x: - self.p0 * np.sin(
                                self.w0 / self.c0 * (self.c0 * t - x[0])))
        self.d2u_i.interpolate(lambda x: self.p0 * self.w0 * self.w0 * np.sin(
                                self.w0 / self.c0 * (self.c0 * t - x[0])))

        # Update solution field
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

        return self.u_n, self.v_n, t
