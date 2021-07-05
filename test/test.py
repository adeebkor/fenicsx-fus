import numpy as np
from mpi4py import MPI

from dolfinx import IntervalMesh, FunctionSpace, Function
from dolfinx.fem import LinearProblem
from ufl import Circumradius, TrialFunction, TestFunction, dx


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


n = 20
L = 1

mesh = IntervalMesh(
	MPI.COMM_WORLD,
	n,
	[0, L]
)

V = FunctionSpace(mesh, ("Lagrange", 1))

print("Degree of freedoms:", V.dofmap.index_map.size_global)

h = get_hmin(mesh)

print("1. dx:", h)

hc = L / n

print("2. dx:", hc)

def test_meshsize():
    assert np.isclose(2*h, hc)
