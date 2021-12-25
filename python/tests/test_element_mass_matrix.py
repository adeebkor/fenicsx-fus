import pytest
import numpy as np
from mpi4py import MPI

import dolfinx.cpp
import dolfinx.fem
import dolfinx.mesh
import ufl


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("p", [2, 3, 4, 5, 6, 7])
def test_diagonal(dimension, p):
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

    # Create function space
    FE = ufl.FiniteElement("Lagrange", cell_type, p, variant="gll")
    V = dolfinx.fem.FunctionSpace(mesh, FE)
    ndof = V.dofmap.index_map.size_global

    # Define variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Set quadrature degree
    qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14,
          "9": 16, "10": 18}
    md = {"quadrature_rule": "GLL",
          "quadrature_degree": qd[str(p)]}

    a = u*v*ufl.dx(metadata=md)

    # Build element mass matrix
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()

    # Get nonzero indices
    idx = np.nonzero(A[:, :])

    assert(np.allclose(idx[0], np.arange(ndof)) and
           np.allclose(idx[1], np.arange(ndof)))
