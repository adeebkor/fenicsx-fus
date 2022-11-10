from ufl import (FiniteElement, VectorElement, Mesh, FunctionSpace, Constant,
                 Coefficient, TestFunction, hexahedron, inner, grad, ds, dx)

element = FiniteElement("Lagrange", hexahedron, 4, variant="gll")
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

c0 = Constant(mesh)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
v = TestFunction(V)

md = {"quadrature_rule": "GLL", "quadrature_degree": 6}

a0 = inner(u, v) * dx(metadata=md)
L0 = - c0**2 * inner(grad(u_n), grad(v)) * dx(metadata=md)
L1 = c0**2 * inner(g, v) * ds(1, metadata=md)
L2 = - c0 * inner(v_n, v) * ds(2, metadata=md)

forms = [a0, L0, L1, L2]