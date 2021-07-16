import numpy as np

from dolfinx import FunctionSpace
from dolfinx.fem import LinearProblem
from dolfinx.geometry import (BoundingBoxTree, compute_collisions_point,
                              select_colliding_cells)
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


def get_eval_params(mesh, points):
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    idx_on_proc = []
    for i, point in enumerate(points.T):
        # Find cells that are close to the point
        cell_candidates = compute_collisions_point(tree, point)
        # Choose one of the cells that contains the point
        cell = select_colliding_cells(mesh, cell_candidates, point, 1)
        # Only use evaluate for points on current processor
        if len(cell) == 1:
            points_on_proc.append(point)
            cells.append(cell[0])
            idx_on_proc.append(i)

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    return idx_on_proc, points_on_proc, cells
