import numpy as np
from mpi4py import MPI

import dolfinx.cpp
import dolfinx.fem
import dolfinx.mesh
import ufl


def generate_mesh(dimension):
    """
    Generate mesh for testing purposes.
    """

    if dimension == 1:
        # Interval mesh
        mesh = dolfinx.mesh.create_interval(
            MPI.COMM_WORLD,
            1,
            [-1, 1]
        )
        cell_type = ufl.interval
    elif dimension == 2:
        # Quad mesh
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([-1., -1.]), np.array([1., 1.])],
            [1, 1],
            cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
        )
        cell_type = ufl.quadrilateral
    elif dimension == 3:
        # Hex mesh
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([-1., -1., -1.]), np.array([1., 1., 1.])],
            [1, 1, 1],
            cell_type=dolfinx.cpp.mesh.CellType.hexahedron
        )
        cell_type = ufl.hexahedron
    else:
        raise Exception("Dimension {} is not a valid \
                        dimension!".format(dimension))

    return mesh, cell_type


# Choose mesh
mesh, cell_type = generate_mesh(3)

# Create function space
p = 5
FE = ufl.FiniteElement("Lagrange", cell_type, p, variant="gll")
V = dolfinx.fem.FunctionSpace(mesh, FE)
ndof = V.dofmap.index_map.size_global

# Set quadrature degree
qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
      "9": 16, "10": 18}
md = {"quadrature_rule": "GLL",
      "quadrature_degree": qd[str(p)]}

# Define variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = u*v*ufl.dx(metadata=md)

# Build element mass matrix
A = dolfinx.fem.assemble_matrix(a)
A.assemble()

# Get nonzero indices
idx = np.nonzero(A[:, :])


def test_diagonal():
    assert(np.allclose(idx[0], np.arange(ndof)) and
           np.allclose(idx[1], np.arange(ndof)))
