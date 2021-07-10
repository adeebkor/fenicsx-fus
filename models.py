import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import FunctionSpace, Function
from dolfinx.fem import assemble_matrix, assemble_vector
from ufl import TrialFunction, TestFunction, Measure, inner, grad, dx


class Linear:
    """
    Model that consider only diffraction
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2))

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        # self.solver.setType("cg")
        # self.solver.getPC().setType("jacobi")
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

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


class LinearGLL:
    """
    Model that consider only diffraction (GLL)
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Quadrature parameters
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": k+1}

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx(metadata=quad_params)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        # Get diagonal
        # self.m = PETSc.Vec().create(MPI.COMM_WORLD)
        # lsize = self.M.getLocalSize()[0]
        # gsize = self.M.getSize()[0]
        # self.m.setSizes((lsize, gsize))
        # self.m.setUp()
        # self.M.getDiagonal(self.m)

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(
            - inner(grad(self.u_n), grad(self.v))*dx(metadata=quad_params)
            + inner(self.g, self.v)*ds(1, metadata=quad_params)
            - 1/self.c0*inner(self.v_n, self.v)*ds(2, metadata=quad_params))

        # Setup solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType("preonly")
        self.solver.getPC().setType("jacobi")
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

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
        # result.pointwiseDivide(b, self.m)
        self.solver.solve(b, result)

        return result


class Linear1D:
    """
    Model that consider only diffraction (1D)
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2))

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

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


class Linear1DGLL:
    """
    Model that consider only diffraction (1D)
    Use GLL points and quadratures
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Quadrature parameters
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": k+1}

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx(metadata=quad_params)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        # Get diagonal
        self.m = PETSc.Vec().create(MPI.COMM_WORLD)
        lsize = self.M.getLocalSize()[0]
        gsize = self.M.getSize()[0]
        self.m.setSizes((lsize, gsize))
        self.m.setUp()
        self.M.getDiagonal(self.m)

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(
            - inner(grad(self.u_n), grad(self.v))*dx(metadata=quad_params)
            + inner(self.g, self.v)*ds(1, metadata=quad_params)
            - 1/self.c0*inner(self.v_n, self.v)*ds(2, metadata=quad_params))

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

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
        result.pointwiseDivide(b, self.m)

        return result


class Lossy:
    """
    Model that consider diffraction + absorption
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx
                          + inner(self.dg, self.v)*ds(1))

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(-self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(self.p0*self.w0**2/self.c0 * np.sin(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
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


class Westervelt:
    """
    Model for the Westervelt equation, i.e. diffraction + absorption +
    nonlinearity.
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta, beta, rho0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0

        # Define variational formulation
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        self.a = inner(self.u, self.v)*dx \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2) \
            - 2*self.beta/self.rho0/self.c0**2*self.u_n \
            * inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx
                          + inner(self.dg, self.v)*ds(1)) \
            + 2*self.beta/self.rho0/self.c0**2 \
            * inner(self.v_n*self.v_n, self.v)*dx

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(-self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(self.p0*self.w0**2/self.c0 * np.sin(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        self.solver.setOperators(self.M)

        # Solve
        self.solver.solve(b, result)

        return result


class Westervelt1D:
    """
    Model for the 1D Westervelt equation, i.e. diffraction + absorption +
    nonlinearity.
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta, beta, rho0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0

        # Define variational formulation
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        self.a = inner(self.u, self.v)*dx \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2) \
            - 2*self.beta/self.rho0/self.c0**2*self.u_n \
            * inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx
                          + inner(self.dg, self.v)*ds(1)) \
            + 2*self.beta/self.rho0/self.c0**2 \
            * inner(self.v_n*self.v_n, self.v)*dx

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(-self.p0*self.w0**2/self.c0 * np.sin(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        self.solver.setOperators(self.M)

        # Solve
        self.solver.solve(b, result)

        return result


class Westervelt1DGLL:
    """
    Model for the 1D Westervelt equation, i.e. diffraction + absorption +
    nonlinearity.
    Use GLL points and quadratures.
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0, delta, beta, rho0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)
        self.dg = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0
        self.delta = delta
        self.beta = beta
        self.rho0 = rho0

        # Quadrature parameters
        quad_params = {"quadrature_rule": "GLL",
                       "quadrature_degree": k+1}

        # Define variational formulation
        self.u_n = Function(self.V)
        self.v_n = Function(self.V)

        self.a = inner(self.u, self.v)*dx(metadata=quad_params) \
            + self.delta/self.c0*inner(self.u, self.v)*ds(2, metadata=quad_params) \
            - 2*self.beta/self.rho0/self.c0**2*self.u_n \
            * inner(self.u, self.v)*dx(metadata=quad_params)
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.m = PETSc.Vec().create(MPI.COMM_WORLD)
        lsize = self.M.getLocalSize()[0]
        gsize = self.M.getSize()[0]
        self.m.setSizes((lsize, gsize))
        self.m.setUp()

        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx(metadata=quad_params)
                             + inner(self.g, self.v)*ds(1, metadata=quad_params)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2, metadata=quad_params)) \
            + self.delta*(-inner(grad(self.v_n), grad(self.v))*dx(metadata=quad_params)
                          + inner(self.dg, self.v)*ds(1, metadata=quad_params)) \
            + 2*self.beta/self.rho0/self.c0**2 \
            * inner(self.v_n*self.v_n, self.v)*dx(metadata=quad_params)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(self.p0*self.w0/self.c0 * np.cos(self.w0 * t))

        with self.dg.vector.localForm() as dg_local:
            dg_local.set(-self.p0*self.w0**2/self.c0 * np.sin(self.w0 * t))
            # dg_local.set(0.0)

        # Update fields that f depends on
        u.copy(result=self.u_n.vector)
        self.u_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        v.copy(result=self.v_n.vector)
        self.v_n.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)

        # Assemble LHS
        self.M = assemble_matrix(self.a)
        self.M.assemble()
        self.M.getDiagonal(self.m)

        # Assemble RHS
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Solve
        result.pointwiseDivide(b, self.m)

        return result


class LinearPulse:
    """
    Model for the linear wave equation
    """

    def __init__(self, mesh, meshtag, k, c0, freq0, p0):
        self.V = FunctionSpace(mesh, ("Lagrange", k))
        self.u, self.v = TrialFunction(self.V), TestFunction(self.V)
        self.g = Function(self.V)

        # Tag boundary facets
        ds = Measure('ds', subdomain_data=meshtag, domain=mesh)

        # Physical parameters
        self.c0 = c0
        self.freq = freq0
        self.w0 = 2 * np.pi * self.freq
        self.p0 = p0

        # Temporal parameters
        self.Td = 6/self.freq
        self.Tw = 3/self.freq
        self.Tend = 2*self.Td

        # Define variational formulation
        self.a = inner(self.u, self.v)*dx
        self.M = assemble_matrix(self.a)
        self.M.assemble()

        self.u_n = Function(self.V)
        self.v_n = Function(self.V)
        self.L = self.c0**2*(-inner(grad(self.u_n), grad(self.v))*dx
                             + inner(self.g, self.v)*ds(1)
                             - 1/self.c0*inner(self.v_n, self.v)*ds(2))

        # Build solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver.setOperators(self.M)

    def init(self):
        """Return vectors with the initial condition"""
        u, v = Function(self.V), Function(self.V)
        with u.vector.localForm() as _u, v.vector.localForm() as _v:
            _u.set(0.0)
            _v.set(0.0)
        return u, v

    def f0(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For du/dt = f0(t, u, v), return f0"""
        return v.copy(result=result)

    def f1(self, t: float, u: PETSc.Vec, v: PETSc.Vec, result: PETSc.Vec) \
            -> PETSc.Vec:
        """For dv/dt = f1(t, u, v), return f1"""

        # Update boundary condition
        with self.g.vector.localForm() as g_local:
            g_local.set(-self.p0*self.w0/self.c0 * \
                        np.cos(self.w0 * (t-self.Td)) * \
                        np.exp(-((t-self.Td)/(self.Tw/2))**2) * \
                        (np.heaviside(t, 0)-np.heaviside(t-self.Tend, 0))
                        +4*self.p0/self.c0/self.Tw * \
                        np.sin(self.w0 * (t-self.Td)) * \
                        np.exp(-((t-self.Td)/(self.Tw/2))**2) * \
                        (np.heaviside(t, 0)-np.heaviside(t-self.Tend, 0)))

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
    
