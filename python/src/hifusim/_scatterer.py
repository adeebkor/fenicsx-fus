import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl_wrapper
from dolfinx.fem import (FunctionSpace, Function, dirichletbc, form,
                         locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from ufl import (FiniteElement, TestFunction, Measure, inner, grad, dx)


class LinearSoundSoftGLL:
    """
    Solver for the 2D sound soft scattering problem.

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0):
        self.mesh = mesh

        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)
        facets = meshtags.indices[meshtags.values == 4]
        tdim = mesh.topology.dim

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

        # Set boundary condition
        # Since the LHS is a vector, we need to set the bc to ones.
        self.m_bc = dirichletbc(PETSc.ScalarType(
            1), locate_dofs_topological(self.V, tdim-1, facets), self.V)
        self.b_bc = dirichletbc(PETSc.ScalarType(
            0), locate_dofs_topological(self.V, tdim-1, facets), self.V)

        # Define variational form
        self.u.x.array[:] = 1.0
        self.a = form(inner(self.u, self.v) * dx(metadata=md))
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.m, [self.m_bc])

        self.L = form(self.c0**2*(- inner(grad(self.u_n), grad(self.v))
                                  * dx(metadata=md)
                                  + inner(self.g, self.v)
                                  * ds(1, metadata=md)
                                  - 1/self.c0*inner(self.v_n, self.v)
                                  * ds(2, metadata=md)))
        self.b = assemble_vector(self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, [self.b_bc])

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
        # apply_lifting(self.b, [self.a], bcs=[[self.b_bc]])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, [self.b_bc])

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
        file = XDMFFile(MPI.COMM_WORLD, "u_t.xdmf", "w")
        file.write_mesh(self.mesh)
        file.write_function(self.u_n, t=t0)

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
                file.write_function(self.u_n, t=t)
                PETSc.Sys.syncPrint("t: {},\t Steps: {}/{}".format(
                    t, step, nstep))

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.vector)
        v_.copy(result=self.v_n.vector)
        file.close()

        return self.u_n, self.v_n, t, step


class LinearSoundHardGLL:
    """
    Solver for the 2D sound soft scattering problem.

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0):
        self.mesh = mesh

        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
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
                      jit_parameters=jit_params)
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(self.c0**2*(- inner(grad(self.u_n), grad(self.v))
                                  * dx(metadata=md)
                                  + inner(self.g, self.v)
                                  * ds(1, metadata=md)
                                  - 1/self.c0*inner(self.v_n, self.v)
                                  * ds(2, metadata=md)),
                      jit_parameters=jit_params)
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
        file = XDMFFile(MPI.COMM_WORLD, "u_t.xdmf", "w")
        file.write_mesh(self.mesh)
        file.write_function(self.u_n, t=t0)

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
                file.write_function(self.u_n, t=t)
                PETSc.Sys.syncPrint("t: {:10.10f},\t Steps: {}/{}".format(
                    t, step, nstep))

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.vector)
        v_.copy(result=self.v_n.vector)
        file.close()

        return self.u_n, self.v_n, t, step


class LinearPenetrableGLL:
    """
    Solver for the 2D penetrable scattering problem.

    This solver uses GLL lattice and GLL quadrature.
    """

    def __init__(self, mesh, meshtags, k, c0, freq0, p0, r0, ref_index):
        self.mesh = mesh
        self.scatterer_radius = r0

        cell_type = basix.cell.string_to_type(mesh.ufl_cell().cellname())
        element = basix.create_element(
            basix.ElementFamily.P, cell_type, k,
            basix.LagrangeVariant.gll_warped)
        FE = basix.ufl_wrapper.BasixElement(element)
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.u = Function(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtags, domain=mesh)
        tdim = mesh.topology.dim

        # Refractive index function
        V_DG = FunctionSpace(mesh, ("DG", 0))
        self.coeff = Function(V_DG)
        cells_0 = locate_entities(
            mesh, tdim, lambda x: np.sqrt((x[0] - 0.5)**2 +
                                          (x[1] - 0.5)**2 +
                                          (x[2])**2) <= r0)
        self.coeff.x.array[:] = c0
        self.coeff.x.array[cells_0] = np.full(len(cells_0), ref_index*c0)

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
                      jit_parameters=jit_params)
        self.m = assemble_vector(self.a)
        self.m.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

        self.L = form(
            - inner(self.coeff * self.coeff * grad(self.u_n), grad(self.v))
            * dx(metadata=md)
            + inner(self.coeff * self.coeff * self.g, self.v)
            * ds(1, metadata=md)
            - inner(self.coeff * self.v_n, self.v)
            * ds(2, metadata=md),
            jit_parameters=jit_params)
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
        file = XDMFFile(MPI.COMM_WORLD, "u_t.xdmf", "w")
        file.write_mesh(self.mesh)
        file.write_function(self.u_n, t=t0)

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
                file.write_function(self.u_n, t=t)
                PETSc.Sys.syncPrint("t: {},\t Steps: {}/{}".format(
                    t, step, nstep))

        u_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        v_.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        u_.copy(result=self.u_n.vector)
        v_.copy(result=self.v_n.vector)
        file.close()

        return self.u_n, self.v_n, t, step
