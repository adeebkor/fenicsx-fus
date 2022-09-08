import numpy as np
from scipy.integrate import RK45
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl_wrapper
from dolfinx.fem import FunctionSpace, Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import VTXWriter, VTKFile
from ufl import TestFunction, TrialFunction, Measure, inner, grad, dx


class Linear:
    """
    Solver for the second order linear wave equation for homogenous media.

    This solver uses GLL lattice and Gauss quadrature.

    """

    def __init__(self, mesh, meshtags, k, c0, rho0, freq0, p0):

        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)
        
        # Define functions
        self.v = TestFunction(V)
        self.u = TrialFunction(V)
        self.g = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Define forms
        self.a = form(inner(1/self.rho0*self.u, self.v) * dx)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = form(
            - inner(self.c0*self.c0/self.rho0*grad(self.u_n), grad(self.v)) 
            * dx
            + inner(self.c0*self.c0/self.rho0*self.g, self.v) 
            * ds(1)
            - inner(self.c0/self.rho0*self.v_n, self.v) * ds(2))
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

        return self.u_n, self.v_n, t


class LinearGLL:
    """
    Solver for linear second order wave equation for homogenous media.

    This solver uses GLL lattice and GLL quadrature such that it produces
    a diagonal mass matrix.

    """

    def __init__(self, mesh, meshtags, k, c0, rho0, freq0, p0):

        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)

        # Define functions
        self.v = TestFunction(V)
        self.u = Function(V)
        self.g = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # JIT compilation parameters
        jit_params = {"cffi_extra_compile_args": ["-Ofast", "-march=native"]}

        # Define forms
        self.u.x.array[:] = 1.0
        self.a = form(inner(1/self.rho0*self.u, self.v) * dx(metadata=md),
                      jit_params=jit_params)
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            - inner(self.c0*self.c0/self.rho0*grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.c0*self.c0/self.rho0*self.g, self.v)
            * ds(1, metadata=md)
            - inner(self.c0/self.rho0*self.v_n, self.v)
            * ds(2, metadata=md),
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

        return self.u_n, self.v_n, t


class LinearGLLSciPy:
    """
    Solver for linear second order wave equation for homogenous media.

    This solver uses GLL lattice and GLL quadrature such that it produces
    a diagonal mass matrix. It is design to use SciPy Runge-Kutta solver
    and as such only works in serial.

    Note:
        - Experimental
    
    """

    def __init__(self, mesh, meshtags, k, c0, rho0, freq0, p0):

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)

        # Define functions
        self.v = TestFunction(V)
        self.u = Function(V)
        self.g = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Get degrees of freedom
        self.ndof = V.dofmap.index_map.size_global

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define forms
        self.u.x.array[:] = 1.0
        self.a = form(inner(1/self.rho0*self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            - inner(self.c0*self.c0/self.rho0*grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.c0*self.c0/self.rho0*self.g, self.v)
            * ds(1, metadata=md)
            - inner(self.c0/self.rho0*self.v_n, self.v)
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


class LinearHeterogenous:
    """
    Solver for the second order linear wave equation in 2 different media.

    This solver uses GLL lattice and Gauss quadrature.

    """

    def __init__(self, mesh, meshtags, k, c0, c1, rho0, rho1, freq0, p0):

        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.c1 = c1
        self.rho0 = rho0
        self.rho1 = rho1
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags[1], domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)
        
        # Define functions
        self.v = TestFunction(V)
        self.u = TrialFunction(V)
        self.g = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Define functions for different medium
        V_DG = FunctionSpace(mesh, ("DG", 0))
        self.c = Function(V_DG)
        self.c.x.array[:] = c0
        self.c.x.array[meshtags[0].find(2)] = c1

        self.rho = Function(V_DG)
        self.rho.x.array[:] = rho0
        self.rho.x.array[meshtags[0].find(2)] = rho1

        # Define forms
        self.a = form(inner(self.u/self.rho/self.c/self.c, self.v) * dx)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = form(
            - inner(1/self.rho*grad(self.u_n), grad(self.v)) * dx
            + inner(1/self.rho*self.g, self.v) * ds(1)
            - inner(1/self.rho/self.c*self.v_n, self.v)*ds(2))
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

        return self.u_n, self.v_n, t


class LinearHeterogenousGLL:
    """
    Solver for the second order linear wave equation in 2 different media.

    This solver uses GLL lattice and GLL quadrature such that it produces
    a diagonal mass matrix.

    """

    def __init__(self, mesh, meshtags, k, c0, c1, rho0, rho1, freq0, p0):

        # MPI
        self.mpi_size = MPI.COMM_WORLD.size
        self.mpi_rank = MPI.COMM_WORLD.rank

        # Physical parameters
        self.c0 = c0
        self.c1 = c1
        self.rho0 = rho0
        self.rho1 = rho1
        self.freq = freq0
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags[1], domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)
        
        # Define functions
        self.v = TestFunction(V)
        self.u = Function(V)
        self.g = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # Define functions for different medium
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

        # Define forms
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u/self.rho/self.c/self.c, self.v) 
                      * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            - inner(1/self.rho*grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(1/self.rho*self.g, self.v)
            * ds(1, metadata=md)
            - inner(1/self.rho/self.c*self.v_n, self.v)
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

        return self.u_n, self.v_n, t


class LinearGLLSponge:
    """
    Solver for linear second order wave equation for homogenous media.

    This solver uses GLL lattice and GLL quadrature such that it produces a
    diagonal mass matrix. It also uses a sponge layer to absorb outgoing
    waves.

    Note:
        - Experimental
    
    """

    def __init__(self, mesh, meshtags, k, c0, rho0, delta0, freq0, p0):

        # MPI
        self.mpi_rank = MPI.COMM_WORLD.rank
        self.mpi_size = MPI.COMM_WORLD.size

        # Physical parameters
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq0
        self.lmbda = self.c0 / self.freq
        self.w0 = 2 * np.pi * freq0
        self.p0 = p0
        self.T = 1 / freq0  # period
        self.alpha = 4  # window length

        # Initialise mesh
        self.mesh = mesh

        # Boundary facets
        ds = Measure('ds', subdomain_data=meshtags[1], domain=mesh)

        # Define cell, finite element and function space
        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        V = FunctionSpace(mesh, FE)

        # Define functions
        self.v = TestFunction(V)
        self.u = Function(V)
        self.g = Function(V)
        self.dg = Function(V)
        self.u_n = Function(V)
        self.v_n = Function(V)

        # --------------------------------------------------------------------
        # Sponge layer
        self.delta = Function(V)
        
        # Linear function
        self.delta.interpolate(lambda x:
                               np.piecewise(x[0], [x[0] < 0.12, x[0] >= 0.12],
                                            [0.0,
                                             lambda x: delta0 / 5 /
                                             self.lmbda * x - 0.12 * delta0
                                             / 5 / self.lmbda]))

        # Quadratic function
        # self.delta.interpolate(lambda x:
        #     np.piecewise(x[0], [x[0] < 0.12, x[0] >= 0.12],
        #                  [0.0,
        #                  lambda x: 25*self.lmbda**2/delta0 * (x -
        #                                                       0.12)**2]))
        # --------------------------------------------------------------------

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
              "9": 16, "10": 18}
        md = {"quadrature_rule": "GLL",
              "quadrature_degree": qd[str(k)]}

        # Define forms
        self.u.x.array[:] = 1.0
        self.a = form(
            inner(1/self.rho0*self.u, self.v) * dx(metadata=md)
            + inner(self.delta/self.rho0/self.c0*self.u, self.v) 
            * ds(2, metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            - inner(self.c0*self.c0/self.rho0*grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.c0*self.c0/self.rho0*self.g, self.v)
            * ds(1, metadata=md)
            - inner(self.c0/self.rho0*self.v_n, self.v)
            * ds(2, metadata=md)
            - inner(self.delta/self.rho0*grad(self.v_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.delta/self.rho0*self.dg, self.v)
            * ds(1, metadata=md))
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
