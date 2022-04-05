from ufl import (FunctionSpace, FiniteElement, VectorElement, Mesh, 
                 Coefficient, TestFunction, hexahedron, inner, dx)

element = FiniteElement("Lagrange", hexahedron, 4, variant="gll")
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

u = Coefficient(V)
v = TestFunction(V)

md = {"quadrature_rule": "GLL",
      "quadrature_degree": 6}

a = inner(u, v) * dx(metadata=md)

forms = [a]