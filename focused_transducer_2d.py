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
# gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.1)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 1e-5)
# gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
model = gmsh.model()

if MPI.COMM_WORLD.rank == 0:
    # Setup model
    model.add("Piston2D")
    model.setCurrent("Piston2D")
    L, r, rmax = 0.07, 0.05, 0.033  # cm

    # Create object
    rect1 = model.occ.addRectangle(-0.04, -rmax, 0.0, L, 2*rmax)
    disk1 = model.occ.addDisk(0, 0, 0, r, r)
    union = model.occ.fuse([(2, rect1)], [(2, disk1)])[0]
    rect2 = model.occ.addRectangle(-r, -rmax, 0.0, L+0.01, 2*rmax)
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

    # Define mesh size
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "FacesList", walls)

    resolution = r / 1000
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "IField", 1)
    model.mesh.field.setNumber(2, "LcMin", resolution)
    model.mesh.field.setNumber(2, "LcMax", 20*resolution)
    model.mesh.field.setNumber(2, "DistMin", 0.5*r)
    model.mesh.field.setNumber(2, "DistMax", r)

    model.mesh.field.add("Distance", 3)
    model.mesh.field.setNumbers(3, "FacesList", transducer)
    model.mesh.field.add("Threshold", 4)
    model.mesh.field.setNumber(4, "IField", 3)
    model.mesh.field.setNumber(4, "LcMin", 5*resolution)
    model.mesh.field.setNumber(4, "LcMax", 10*resolution)
    model.mesh.field.setNumber(4, "DistMin", 0.1)
    model.mesh.field.setNumber(4, "DistMax", 0.5)

    model.mesh.field.add("Min", 5)
    model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    model.mesh.field.setAsBackgroundMesh(5)

    # Meshing
    model.occ.synchronize()
    model.mesh.generate(2)
    # model.mesh.refine()
    # model.mesh.refine()

    gmsh.write("mesh/gmsh/piston2D.msh")

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
    # Create dolfinx mesh on other processes
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
mt.name = "facets"

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/piston2d.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='piston2d']/Geometry")


if MPI.COMM_WORLD.rank == 0:
    # Setup model
    model.add("Piston2D-1")
    model.setCurrent("Piston2D-1")
    L, r, rmax, rinner = 0.07, 0.05, 0.033, 0.01

    # Setup object 1
    rect1 = model.occ.addRectangle(-0.04, -rmax, 0.0, L, 2*rmax)
    disk1 = model.occ.addDisk(0, 0, 0, r, r)
    union1 = model.occ.fuse([(2, rect1)], [(2, disk1)])[0]
    rect2 = model.occ.addRectangle(-r, -rmax, 0.0, L+0.01, 2*rmax)
    isect1 = model.occ.intersect(union1, [(2, rect2)])[0]
    rect3 = model.occ.addRectangle(-r, -0.01, 0.0, L+0.01, 0.02)
    obj1 = model.occ.intersect(isect1, [(2, rect3)], removeObject=True)[0]

    # Setup object 2
    rect1 = model.occ.addRectangle(-0.04, -rmax, 0.0, L, 2*rmax)
    disk1 = model.occ.addDisk(0, 0, 0, r, r)
    union1 = model.occ.fuse([(2, rect1)], [(2, disk1)])[0]
    rect2 = model.occ.addRectangle(-r, -rmax, 0.0, L+0.01, 2*rmax)
    obj2 = model.occ.intersect(union1, [(2, rect2)])[0]

    # Fuse object 1 and 2
    obj = model.occ.fragment(obj1, obj2)

    # Tag physical entities
    model.occ.synchronize()
    surfaces = model.occ.getEntities(dim=2)
    surface_marker = 1
    surface_entities = [surface[1] for surface in surfaces]
    model.addPhysicalGroup(2, surface_entities, surface_marker)
    model.setPhysicalName(2, surface_marker, "Domain surface")
    edges = model.occ.getEntities(dim=1)
    transducer_marker, wall_marker = 1, 2
    transducer_idx = [5, 6]
    walls_idx = [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14]
    transducer = [edges[idx][1] for idx in transducer_idx]
    walls = [edges[idx][1] for idx in walls_idx]
    model.addPhysicalGroup(1, walls, wall_marker)
    model.setPhysicalName(1, wall_marker, "Walls")
    model.addPhysicalGroup(1, transducer, transducer_marker)
    model.setPhysicalName(1, transducer_marker, "Transducer")

    # Meshing
    model.mesh.generate(2)
    # model.mesh.refine()
    # model.mesh.refine()

    gmsh.write("mesh/gmsh/piston2D-1.msh")

    # Create dolfinx mesh on process 0
    x = extract_gmsh_geometry(model, model_name="Piston2D-1")
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        model.mesh.getElementType("triangle", 1), root=0)
    topologies = extract_gmsh_topology_and_markers(model, "Piston2D-1")
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

mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                   ufl_mesh_from_gmsh(gmsh_cell_id, 2))
mesh.name = "piston2d-1"
local_entities, local_values = extract_local_entities(mesh, 1,
                                                      marked_facets,
                                                      facet_values)

mesh.topology.create_connectivity(1, 0)
mt = create_meshtags(mesh, 1,
                     cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "facets"

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "mesh/xdmf/piston2D-1.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    file.write_meshtags(
        mt,
        geometry_xpath="/Xdmf/Domain/Grid[@Name='piston2d-1']/Geometry")