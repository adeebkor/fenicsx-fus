import numpy as np
import dolfinx
import dolfinx.fem
from dolfinx.io import XDMFFile
from mpi4py import MPI

# Read mesh
with XDMFFile(MPI.COMM_WORLD, "piston.xdmf", "r") as file:
    mesh = file.read_mesh(name='piston')
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim-3)
    mt = file.read_meshtags(mesh, name='surfaces')

# Create function space
Q = dolfinx.FunctionSpace(mesh, ('CG', 1))
kappa = dolfinx.Function(Q)
kappa.vector.set(0.0)

wall_facets = mt.indices[mt.values==1]
wall_dofs = dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim-1, wall_facets)
with kappa.vector.localForm() as loc:
    loc.setValues(wall_dofs, np.full(len(wall_dofs), 1.))

transducer_facets = mt.indices[mt.values==2]
transducer_dofs = dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim-1, transducer_facets)
with kappa.vector.localForm() as loc:
    loc.setValues(transducer_dofs, np.full(len(transducer_dofs), 2.))


with XDMFFile(MPI.COMM_WORLD, "kappa.xdmf", "w") as fout:
    fout.write_mesh(mesh)
    fout.write_function(kappa)
