import numpy as np

from dolfinx.geometry import (BoundingBoxTree, compute_collisions,
                              compute_colliding_cells)


def compute_eval_params(mesh, points):
    '''
    Compute the parameters required for dolfinx.Function eval

    Parameters
    ----------

    mesh :   dolfinx.mesh

    points : numpy.ndarray
             The evaluation points of shape (3 by n) where each row corresponds
             to x, y, and z coordinates.

    Returns
    -------

    points_on_proc : numpy.ndarray
                     The evaluation points owned by the process.

    cells          : list
                     A list containing the cell index of the evaluation point.
    '''

    tree = BoundingBoxTree(mesh, mesh.topology.dim, padding=1e-12)
    cells = []
    points_on_proc = []
    cell_candidates = compute_collisions(tree, points.T)
    cell_collisions = compute_colliding_cells(mesh, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        # Only use evaluate for points on current processor
        if len(cell_collisions.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(cell_collisions.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    return points_on_proc, cells


def compute_diffusivity_of_sound(frequency: float, speed: float, attenuationdB: float) -> float:
    attenuationNp = attenuationdB / 20 * np.log(10)  # (Np/m/MHz^2)
    diffusivity = 2 * attenuationNp * speed * speed * speed / frequency / frequency
    return diffusivity
