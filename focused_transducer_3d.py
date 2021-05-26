import numpy as np
from dolfinx import cpp
from dolfinx.cpp.io import extract_local_entities
from dolfinx.io import (XDMFFile, extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh, create_meshtags
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")
import gmsh

# Initialization
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()

if MPI.COMM_WORLD.rank == 0:
    # Setup model
    model.add("Piston2D")
    model.setCurrent("Piston2D")
    L, r, rmax = 0.07, 0.05, 0.033
    rect = model.occ.addRectangle(-0.04, -rmax, 0.0, L, 2*rmax)
    rect2 = model.occ.addRectangle(-r, -rmax, 0.0, L+0.01, 2*rmax)
    disk = model.occ.addDisk(0, 0, 0, r, r)
    union = model.occ.fuse([(2, rect)], [(2, disk)])[0]
    isect = model.occ.intersect(union, [(2, rect2)])

    # Tag physical entities
    model.occ.synchronize()
    surfaces = model.occ.getEntities(dim=2)
    surface_marker = 1
    surface_entities = [surface[1] for surface in surfaces]
    model.addPhysicalGroup(2, surface_entities, surface_marker)
    model.setPhysicalName(2, surface_marker, "Domain surface")
    edges = model.occ.getEntities(dim=1)
    transducer_marker, wall_marker = 1, 2
    transducer_idx = [1, 2]
    walls_idx = [0, 3, 4, 5, 6, 7, 8]
    transducer = [edges[idx][1] for idx in transducer_idx]
    walls = [edges[idx][1] for idx in walls_idx]
    model.addPhysicalGroup(1, walls, wall_marker)
    model.setPhysicalName(1, wall_marker, "Walls")
    model.addPhysicalGroup(1, transducer, transducer_marker)
    model.setPhysicalName(1, transducer_marker, "Transducer")

    # Meshing
    model.mesh.generate(2)
    model.mesh.refine()
    model.mesh.refine()

    gmsh.write("mesh2D.msh")

    # Create dolfinx mesh on process 0
    x = extract_gmsh_geometry(model, model_name="Piston2D")
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        model.mesh.getElementType("triangle", 1), root=0)
    topologies = extract_gmsh_topology_and_markers(model, "Piston2D")
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("line", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
else:
    # Create dolfinx mesh on other procsurfacesesses
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), \
                                  np.empty((0,), dtype=np.int32)

mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                   ufl_mesh_from_gmsh(gmsh_cell_id, 2))
mesh.name = "piston2d"
local_entities, local_values = extract_local_entities(mesh, 1,
                                                      marked_facets,
                                                      facet_values)

mesh.topology.create_connectivity(1, 0)
mt = create_meshtags(mesh, 1,
                     cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "edges"

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "piston2d.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='piston2d']/Geometry")

# ------------------------------------------------------------ #

if MPI.COMM_WORLD.rank == 0:
    # Setup model
    model.add("Piston2D_new")
    model.setCurrent("Piston2D_new")
    L, r, rmax = 0.07, 0.05, 0.033
    rect = model.occ.addRectangle(-0.03, -0.03, 0.0, L, 2*0.03)
    rect2 = model.occ.addRectangle(-0.03, -0.03, 0.0, L+0.01, 2*0.03)
    disk = model.occ.addDisk(0, 0, 0, r, r)
    union = model.occ.fuse([(2, rect)], [(2, disk)])[0]
    isect = model.occ.intersect(union, [(2, rect2)])

    # Tag physical entities
    model.occ.synchronize()
    surfaces = model.occ.getEntities(dim=2)
    surface_marker = 1
    surface_entities = [surface[1] for surface in surfaces]
    model.addPhysicalGroup(2, surface_entities, surface_marker)
    model.setPhysicalName(2, surface_marker, "Domain surface")
    edges = model.occ.getEntities(dim=1)
    transducer_marker, wall_marker = 1, 2
    transducer_idx = [0, 1]
    walls_idx = [3, 4, 5]
    transducer = [edges[idx][1] for idx in transducer_idx]
    walls = [edges[idx][1] for idx in walls_idx]
    model.addPhysicalGroup(1, walls, wall_marker)
    model.setPhysicalName(1, wall_marker, "Walls")
    model.addPhysicalGroup(1, transducer, transducer_marker)
    model.setPhysicalName(1, transducer_marker, "Transducer")

    # Meshing
    model.mesh.generate(2)
    model.mesh.refine()
    model.mesh.refine()

    gmsh.write("mesh2D_new.msh")

    # Create dolfinx mesh on process 0
    x = extract_gmsh_geometry(model, model_name="Piston2D_new")
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        model.mesh.getElementType("triangle", 1), root=0)
    topologies = extract_gmsh_topology_and_markers(model, "Piston2D_new")
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("line", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
else:
    # Create dolfinx mesh on other processes
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), \
                                  np.empty((0,), dtype=np.int32)

mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :2],
                   ufl_mesh_from_gmsh(gmsh_cell_id, 2))
mesh.name = "piston2d_new"
local_entities, local_values = extract_local_entities(mesh, 1,
                                                      marked_facets,
                                                      facet_values)

mesh.topology.create_connectivity(1, 0)
mt = create_meshtags(mesh, 1,
                     cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "edges"

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "piston2d_new.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='piston2d_new']/Geometry")


# --------------------------------------------------------------

if MPI.COMM_WORLD.rank == 0:
    # Setup model
    model.add("Rectangle")
    model.setCurrent("Rectangle")
    L, r, rmax = 0.07, 0.05, 0.033
    rect = model.occ.addRectangle(0, 0, 0.0, 1, 1)
    # rect2 = model.occ.addRectangle(-0.03, -0.03, 0.0, L+0.01, 2*0.03)
    # disk = model.occ.addDisk(0, 0, 0, r, r)
    # union = model.occ.fuse([(2, rect)], [(2, disk)])[0]
    # isect = model.occ.intersect(union, [(2, rect2)])

    # Tag physical entities
    model.occ.synchronize()
    surfaces = model.occ.getEntities(dim=2)
    surface_marker = 1
    surface_entities = [surface[1] for surface in surfaces]
    model.addPhysicalGroup(2, surface_entities, surface_marker)
    model.setPhysicalName(2, surface_marker, "Domain surface")
    edges = model.occ.getEntities(dim=1)
    transducer_marker, wall_marker = 1, 2
    transducer_idx = [3]
    walls_idx = [1]
    transducer = [edges[idx][1] for idx in transducer_idx]
    walls = [edges[idx][1] for idx in walls_idx]
    model.addPhysicalGroup(1, walls, wall_marker)
    model.setPhysicalName(1, wall_marker, "Walls")
    model.addPhysicalGroup(1, transducer, transducer_marker)
    model.setPhysicalName(1, transducer_marker, "Transducer")

    # Meshing
    model.mesh.generate(2)
    model.mesh.refine()
    model.mesh.refine()

    gmsh.write("rectangle.msh")

    # Create dolfinx mesh on process 0
    x = extract_gmsh_geometry(model, model_name="Rectangle")
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        model.mesh.getElementType("triangle", 1), root=0)
    topologies = extract_gmsh_topology_and_markers(model, "Rectangle")
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("line", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
else:
    # Create dolfinx mesh on other processes
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), \
                                  np.empty((0,), dtype=np.int32)

mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :2],
                   ufl_mesh_from_gmsh(gmsh_cell_id, 2))
mesh.name = "rectangle"
local_entities, local_values = extract_local_entities(mesh, 1,
                                                      marked_facets,
                                                      facet_values)

mesh.topology.create_connectivity(1, 0)
mt = create_meshtags(mesh, 1,
                     cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "edges"

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "rectangle.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='rectangle']/Geometry")