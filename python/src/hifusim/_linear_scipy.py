import numpy as np
from scipy.integrate import RK45
from petsc4py import PETSc

from dolfinx.fem import FunctionSpace, Function, assemble_vector, form
from ufl import FiniteElement, TestFunction, Measure, inner, grad, dx


class LinearSciPy:
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
        RK solver, uses SciPy RK adaptive time-step solver.

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
