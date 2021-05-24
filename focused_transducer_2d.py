import numpy as np
import dolfinx
from dolfinx import cpp
from dolfinx.cpp.io import extract_local_entities, perm_gmsh
from dolfinx.io import (XDMFFile, extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh, create_meshtags
from mpi4py import MPI
from petsc4py import PETSc

import warnings
warnings.filterwarnings("ignore")
import gmsh

# Initialization
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()

if MPI.COMM_WORLD.rank == 0:
    # Setup entities
    model.add("Piston")
    model.setCurrent("Piston")
    L, r, rmax = 7, 3, 3.3
    cylinder = model.occ.addCylinder(0, 0, 0, 0, 0, L, rmax)
    disk = model.occ.addDisk(0, 0, 0, r, r)
    fragment = model.occ.fragment([(3, cylinder)], [(2, disk)])

    # Tag physical entities
    model.occ.synchronize()
    volumes = model.occ.getEntities(dim=3)
    volume_marker = 1
    volume_entities = [model[1] for model in volumes]
    model.addPhysicalGroup(3, volume_entities, volume_marker)
    model.setPhysicalName(3, volume_marker, "Domain volume")
    surfaces = model.occ.getEntities(dim=2)
    transducer_marker, wall_marker = 2, 3
    transducer_idx = [1, 3]
    walls_idx = [0, 2]
    transducer = [surfaces[idx][1] for idx in transducer_idx]
    walls = [surfaces[idx][1] for idx in walls_idx]
    model.addPhysicalGroup(2, walls, wall_marker)
    model.setPhysicalName(2, wall_marker, "Walls")
    model.addPhysicalGroup(2, transducer, transducer_marker)
    model.setPhysicalName(2, transducer_marker, "Transducer")

    # Meshing
    model.mesh.generate(3)
    # model.mesh.refine()

    gmsh.write("mesh3D.mesh")

    # Create dolfinx mesh on process 0
    x = extract_gmsh_geometry(model, model_name="Piston")
    gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("tetrahedron", 1), root=0)
    topologies = extract_gmsh_topology_and_markers(model, "Piston")
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("triangle", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)

else:
    # Create dolfinx mesh on other processes
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 3))
mesh.name = "piston"
local_entities, local_values = extract_local_entities(mesh, 2, marked_facets, facet_values)

mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "surfaces"

# Check tags using Paraview
Q = dolfinx.FunctionSpace(mesh, ('DG', 0))
kappa = dolfinx.Function(Q)

with kappa.vector.localForm() as loc:
    transducer_facets = mt.indices[mt.values==transducer_marker]
    loc.setValues(transducer_facets, np.full(len(transducer_facets), 10.0))
    wall_facets = mt.indices[mt.values==wall_marker]
    loc.setValues(wall_facets, np.full(len(wall_facets), 0.0))

# Write mesh in XDMF
with XDMFFile(MPI.COMM_WORLD, "piston.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='piston']/Geometry")
    file.write_function(kappa)
