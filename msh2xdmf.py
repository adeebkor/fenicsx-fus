import meshio


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = msh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=msh.points, cells={
                           cell_type : cells},
                           cell_data={"name_to_read": [cell_data]})

    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


msh = meshio.read("hifu_mesh_3d.msh")

hex_mesh = create_mesh(msh, "hexahedron")
quad_mesh = create_mesh(msh, "quad")

meshio.write("mesh.xdmf", hex_mesh)
meshio.write("mf.xdmf", quad_mesh)
