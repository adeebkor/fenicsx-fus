from ufl import (FiniteElement, VectorElement, Mesh, FunctionSpace, Constant,
                 Coefficient, TestFunction, hexahedron, inner, grad, dx)

element = FiniteElement("Lagrange", hexahedron, 4, variant="gll")
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

c0 = Constant(mesh)
u = Coefficient(V)
v = TestFunction(V)

md = {"quadrature_rule": "GLL", "quadrature_degree": 6}

a = - c0**2 * inner(grad(u), grad(v)) * dx(metadata=md)

forms = [a]