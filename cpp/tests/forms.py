import basix.ufl_wrapper
from ufl import (FunctionSpace, VectorElement, Mesh, Coefficient, Constant,
                 TestFunction, hexahedron, inner, grad, dx)

element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, 4,
    basix.LagrangeVariant.gll_warped)
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

c0 = Constant(mesh)

u = Coefficient(V)
v = TestFunction(V)

md = {"quadrature_rule": "GLL",
      "quadrature_degree": 6}

M = inner(u, v) * dx(metadata=md)
L = - c0**2 * inner(grad(u), grad(v)) * dx(metadata=md)

forms = [M, L]
