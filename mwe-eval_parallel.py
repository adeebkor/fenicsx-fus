from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from dolfinx.geometry import BoundingBoxTree, compute_collisions_point, select_colliding_cells
from dolfinx import FunctionSpace, Function, RectangleMesh
from mpi4py import MPI
from petsc4py import PETSc

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0., 0., 0.]), np.array([1., 1., 0.])],
    [16, 16]
)

V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
u.interpolate(lambda x: np.sin(2*np.pi*x[0]))   

npts = 100
u_eval_global = np.full(npts, -np.inf)
x0 = np.linspace(0, 1, npts)
points = np.zeros((3, npts))
points[0] = x0
points[1] = 0.5

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

if len(points_on_proc) != 0:
    u_eval_global[idx_on_proc] = u.eval(points_on_proc, cells).flatten()

u_eval = np.zeros((npts), dtype=None)
MPI.COMM_WORLD.Reduce(u_eval_global, u_eval, op=MPI.MAX, root=0)

if MPI.COMM_WORLD.rank == 0:
    plt.plot(x0, u_eval)
    plt.savefig("global.png")
