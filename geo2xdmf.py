import numpy as np
import gmsh

from dolfinx import cpp
from dolfinx.cpp.io import extract_local_entities, perm_gmsh
from dolfinx.io import (extract_gmsh_geometry, ufl_mesh_from_gmsh,
                        extract_gmsh_topology_and_markers, XDMFFile)
from dolfinx.mesh import create_mesh, create_meshtags
from mpi4py import MPI

# Initialization
gmsh.initialize()
gmsh.open("hifu_mesh_3d.geo")

if MPI.COMM_WORLD.rank == 0:
    gmsh.model.mesh.generate(3)

    x = extract_gmsh_geometry(gmsh.model)
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        gmsh.model.mesh.getElementType("hexahedron", 1), root=0
    )
    topologies = extract_gmsh_topology_and_markers(gmsh.model)
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]

    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = gmsh.model.mesh.getElementType("quadrangle", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)

else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets = np.empty((0, 3), dtype=np.int64)
    facet_values = np.empty((0, ), dtype=np.int32)

domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
gmsh_hexahedron8 = perm_gmsh(cpp.mesh.CellType.hexahedron, 8)
cells = cells[:, gmsh_hexahedron8]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hifu"

gmsh_quadrangle4 = perm_gmsh(cpp.mesh.CellType.quadrilateral, 4)
marked_facets = marked_facets[:, gmsh_quadrangle4]

local_entities, local_values = extract_local_entities(
    mesh, 2, marked_facets, facet_values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, 
                     cpp.graph.AdjacencyList_int32(local_entities),
    np.int32(local_values))
mt.name = "hifu_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(
        mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hifu']/Geometry")
