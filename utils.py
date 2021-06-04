from dolfinx import FunctionSpace
from dolfinx.fem import LinearProblem
from ufl import Circumradius, TrialFunction, TestFunction, dx

from mpi4py import MPI


def get_hmin(mesh):
    cr = Circumradius(mesh)
    V = FunctionSpace(mesh, ("DG", 0))
    u, v = TrialFunction(V), TestFunction(V)
    a = u*v*dx
    L = cr*v*dx
    lp = LinearProblem(a, L)
    r = lp.solve()
    min_distance = MPI.COMM_WORLD.allreduce(min(r.vector.array), op=MPI.MIN)
    return min_distance
