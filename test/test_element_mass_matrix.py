import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.cpp
import dolfinx.fem
import ufl


def generate_mesh(dimension):
    """
    Generate mesh for testing purposes.
    """

    if dimension == 1:
        # Interval mesh
        mesh = dolfinx.IntervalMesh(
            MPI.COMM_WORLD,
            1,
            [-1, 1]
        )
    elif dimension == 2:
        # Quad mesh
        mesh = dolfinx.RectangleMesh(
            MPI.COMM_WORLD,
            [np.array([-1., -1., 0.]), np.array([1., 1., 0.])],
            [1, 1],
            cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
        )
    elif dimension == 3:
        # Hex mesh
        mesh = dolfinx.BoxMesh(
            MPI.COMM_WORLD,
            [np.array([-1., -1., -1.]), np.array([1., 1., 1.])],
            [1, 1, 1],
            cell_type=dolfinx.cpp.mesh.CellType.hexahedron
        )
    else:
        raise Exception("Dimension {} is not a valid \
                        dimension!".format(dimension))

    return mesh


# Choose mesh
mesh = generate_mesh(2)

# Create function space
p = 3
V = dolfinx.FunctionSpace(mesh, ("Lagrange", p))

# Define variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = u*v*ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": p+1})

# Build element mass matrix
A = dolfinx.fem.assemble_matrix(a)
A.assemble()

# Print element mass matrix
print("Me = ")
print(A[:, :])
