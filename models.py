import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import FunctionSpace, Function
from dolfinx.fem import assemble_matrix, assemble_vector
from ufl import (FiniteElement, TrialFunction, TestFunction, Measure, inner,
                 grad, dx)


class Wave:
    """
    Base class for wave models.
    """

    def __init__(self, mesh, meshtags, fe, k, c0, freq0, p0):
        FE = FiniteElement("Lagrange", mesh.ufl_cell(), k, variant=fe)
        self.V = FunctionSpace(mesh, FE)
        self.v = TestFunction(self.V)
        self.g = Function(self.V)
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        # Tag boundary facets
        self.ds = Measure('ds', subdomain_data=meshtags, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        u.x.array[:] = 0.0
        v.x.array[:] = 0.0

        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""

        return v.copy(result=result)


class LinearEquispaced(Wave):
    """
    Solver for linear second order wave equation.

    The model uses an equispaced lattice and Gauss quadrature to compute
    the mass matrix. A direct solver is use to solve Ax=b.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, windowing=True):
        super().__init__(mesh, meshtags, "equispaced", k, c0, freq0, p0)
        self.u = TrialFunction(self.V)
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Define variational formulation
        self.a = inner(self.u, self.v) * dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2*(- inner(grad(self.u_n), grad(self.v)) * dx
                             + inner(self.g, self.v) * self.ds(1)
                             - 1/self.c0*inner(self.v_n, self.v) * self.ds(2))

        # Linear solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.M)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            else:
                window = 1.0
        else:
            window = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(b, result)

        return result


class LinearGLL(Wave):
    """
    Solver for linear second order wave equation.

    The model uses an GLL lattice and GLL quadrature to compute
    the mass matrix. This results in a diagonal mass matrix.
    The linear solver is then just a 1 / diagonal.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, windowing=True,
                 vectorised=False):
        super().__init__(mesh, meshtags, "gll", k, c0, freq0, p0)
        self.vectorised = vectorised
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8}
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": qd[str(k)]}

        # Define variational formulation
        if vectorised:
            self.u = Function(self.V)
            self.u.x.array[:] = 1.0
            self.a = inner(self.u, self.v) * dx(metadata=quad_params)
            self.m = assemble_vector(self.a)
            self.m.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
        else:
            self.u = TrialFunction(self.V)
            self.a = inner(self.u, self.v) * dx(metadata=quad_params)
            self.M = assemble_matrix(self.a)
            self.M.assemble()

            # Linear solver
            self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.JACOBI)
            self.solver.setOperators(self.M)

        self.L = self.c0**2*(- inner(grad(self.u_n), grad(self.v))
                             * dx(metadata=quad_params)
                             + inner(self.g, self.v)
                             * self.ds(1, metadata=quad_params)
                             - 1/self.c0*inner(self.v_n, self.v)
                             * self.ds(2, metadata=quad_params))

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
            else:
                window = 1.0
        else:
            window = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        if self.vectorised:
            result.pointwiseDivide(b, self.m)
        else:
            self.solver.solve(b, result)

        return result


class LossyEquispaced(Wave):
    """
    Solver for the linear second order wave equation with absorption term.

    The model uses an equispaced lattice and Gauss quadrature to compute the
    mass matrix. A direct solver is use to solve Ax=b.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta,
                 windowing=True):
        super().__init__(mesh, meshtags, "equispaced", k, c0, freq0, p0)
        self.u = TrialFunction(self.V)
        self.dg = Function(self.V)
        self.delta = delta
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Define variational formulation
        self.a = inner(self.u, self.v) * dx \
            + self.delta / self.c0 * inner(self.u, self.v) * self.ds(2)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2 * (- inner(grad(self.u_n), grad(self.v)) * dx
                               + inner(self.g, self.v) * self.ds(1)
                               - 1 / self.c0 * inner(self.v_n, self.v)
                               * self.ds(2))  \
            + self.delta*(- inner(grad(self.v_n), grad(self.v)) * dx
                          + inner(self.dg, self.v) * self.ds(1))

        # Linear solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.M)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
                dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                    np.sin(self.freq * np.pi * t / self.alpha)
            else:
                window = 1.0
                dwindow = 0.0
        else:
            window = 1.0
            dwindow = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        self.solver.solve(b, result)

        return result


class LossyGLL(Wave):
    """
    Solver for the linear second order wave equation with absorption term.

    The model uses an GLL lattice and GLL quadrature to compute
    the mass matrix. This results in a diagonal mass matrix.
    The linear solver is then just a 1 / diagonal.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta,
                 windowing=True, vectorised=False):
        super().__init__(mesh, meshtags, "gll", k, c0, freq0, p0)
        self.vectorised = vectorised
        self.dg = Function(self.V)
        self.delta = delta
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8}
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": qd[str(k)]}

        # Define variational formulation
        if vectorised:
            self.u = Function(self.V)
            self.u.x.array[:] = 1.0
            self.a = inner(self.u, self.v) * dx(metadata=quad_params) \
                + self.delta / self.c0 * inner(self.u, self.v) \
                * self.ds(2, metadata=quad_params)
            self.m = assemble_vector(self.a)
            self.m.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
        else:
            self.u = TrialFunction(self.V)
            self.a = inner(self.u, self.v) * dx(metadata=quad_params) \
                + self.delta / self.c0 * inner(self.u, self.v) \
                * self.ds(2, metadata=quad_params)
            self.M = assemble_matrix(self.a)
            self.M.assemble()

            # Linear solver
            self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.JACOBI)
            self.solver.setOperators(self.M)

        self.L = self.c0**2 * (- inner(grad(self.u_n), grad(self.v))
                               * dx(metadata=quad_params)
                               + inner(self.g, self.v)
                               * self.ds(1, metadata=quad_params)
                               - 1 / self.c0 * inner(self.v_n, self.v)
                               * self.ds(2, metadata=quad_params)) \
            + self.delta*(- inner(grad(self.v_n), grad(self.v))
                          * dx(metadata=quad_params)
                          + inner(self.dg, self.v)
                          * self.ds(1, metadata=quad_params))

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
                dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                    np.sin(self.freq * np.pi * t / self.alpha)
            else:
                window = 1.0
                dwindow = 0.0
        else:
            window = 1.0
            dwindow = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        if self.vectorised:
            result.pointwiseDivide(b, self.m)
        else:
            self.solver.solve(b, result)

        return result


class WesterveltEquispaced(Wave):
    """
    Solver for the Westervelt equation.

    The model uses an equispaced lattice and Gauss quadrature to compute the
    mass matrix. A direct solver is use to solve Ax=b.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta, beta, rho0,
                 windowing=True):
        super().__init__(mesh, meshtags, "equispaced", k, c0, freq0, p0)
        self.u = TrialFunction(self.V)
        self.dg = Function(self.V)
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Define variational formulation
        self.a = inner(self.u, self.v) * dx \
            + self.delta / self.c0 * inner(self.u, self.v) * self.ds(2) \
            - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n \
            * inner(self.u, self.v) * dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2 * (- inner(grad(self.u_n), grad(self.v)) * dx
                               + inner(self.g, self.v) * self.ds(1)
                               - 1 / self.c0*inner(self.v_n, self.v)
                               * self.ds(2)) \
            + self.delta * (- inner(grad(self.v_n), grad(self.v)) * dx
                            + inner(self.dg, self.v) * self.ds(1)) \
            + 2 * self.beta / self.rho0 / self.c0**2 \
            * inner(self.v_n*self.v_n, self.v) * dx

        # Linear solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.getPC().setFactorSolverType("mumps")
        self.solver.setOperators(self.M)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
                dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                    np.sin(self.freq * np.pi * t / self.alpha)
            else:
                window = 1.0
                dwindow = 0.0
        else:
            window = 1.0
            dwindow = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Assemble LHS
        self.M = assemble_matrix(self.a)
        self.M.assemble()
        self.solver.setOperators(self.M)

        # Solve
        self.solver.solve(b, result)

        return result


class WesterveltGLL(Wave):
    """
    Solver for the linear second order wave equation with absorption term.

    The model uses an GLL lattice and GLL quadrature to compute
    the mass matrix. This results in a diagonal mass matrix.
    The linear solver is then just a 1 / diagonal.
    """
    def __init__(self, mesh, meshtags, k, c0, freq0, p0, delta, beta, rho0,
                 windowing=True, vectorised=False):
        super().__init__(mesh, meshtags, "gll", k, c0, freq0, p0)
        self.vectorised = vectorised
        self.dg = Function(self.V)
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0
        self.windowing = True
        self.T = 1 / self.freq  # period
        self.alpha = 4  # window length

        # Quadrature parameters
        qd = {"2": 3, "3": 4, "4": 6, "5": 8}
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": qd[str(k)]}

        # Define variational formulation
        if vectorised:
            self.u = Function(self.V)
            self.u.x.array[:] = 1.0
            self.a = inner(self.u, self.v) * dx(metadata=quad_params) \
                + self.delta / self.c0 * inner(self.u, self.v) \
                * self.ds(2, metadata=quad_params) \
                - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n \
                * inner(self.u, self.v) * dx(metadata=quad_params)
            self.m = assemble_vector(self.a)
            self.m.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
        else:
            self.u = TrialFunction(self.V)
            self.a = inner(self.u, self.v) * dx(metadata=quad_params) \
                + self.delta / self.c0 * inner(self.u, self.v) \
                * self.ds(2, metadata=quad_params) \
                - 2 * self.beta / self.rho0 / self.c0**2 * self.u_n \
                * inner(self.u, self.v) * dx(metadata=quad_params)
            self.M = assemble_matrix(self.a)
            self.M.assemble()

            # Linear solver
            self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.JACOBI)
            self.solver.setOperators(self.M)

        self.L = self.c0**2 * (- inner(grad(self.u_n), grad(self.v))
                               * dx(metadata=quad_params)
                               + inner(self.g, self.v)
                               * self.ds(1, metadata=quad_params)
                               - 1 / self.c0*inner(self.v_n, self.v)
                               * self.ds(2, metadata=quad_params)) \
            + self.delta * (- inner(grad(self.v_n), grad(self.v))
                            * dx(metadata=quad_params)
                            + inner(self.dg, self.v)
                            * self.ds(1, metadata=quad_params)) \
            + 2 * self.beta / self.rho0 / self.c0**2 \
            * inner(self.v_n*self.v_n, self.v) * dx(metadata=quad_params)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        if self.windowing:
            if t < self.T * self.alpha:
                window = 0.5 * (1 - np.cos(self.freq * np.pi * t / self.alpha))
                dwindow = 0.5 * np.pi * self.freq / self.alpha * \
                    np.sin(self.freq * np.pi * t / self.alpha)
            else:
                window = 1.0
                dwindow = 0.0
        else:
            window = 1.0
            dwindow = 1.0

        # Update boundary condition
        self.g.x.array[:] = window * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t)
        self.dg.x.array[:] = dwindow * self.p0 * self.w0 / self.c0 \
            * np.cos(self.w0 * t) - window * self.p0 * self.w0**2 / self.c0 \
            * np.sin(self.w0 * t)

        # Update fields that depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        if self.vectorised:
            self.m = assemble_vector(self.a)
            self.m.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
            result.pointwiseDivide(b, self.m)
        else:
            self.M = assemble_matrix(self.a)
            self.M.assemble()
            self.solver.setOperators(self.M)
            self.solver.solve(b, result)

        return result
