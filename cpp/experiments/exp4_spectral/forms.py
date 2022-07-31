import basix.ufl_wrapper
from ufl import (VectorElement, Mesh, FunctionSpace, Constant, Coefficient,
                 TestFunction, hexahedron, inner, grad, ds, dx)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points

element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, P,
    basix.LagrangeVariant.gll_warped)
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

c0 = Constant(mesh)
delta = Constant(mesh)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
dg = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

a = (inner(u, v) * dx(metadata=md) 
     + delta / c0 * inner(u, v) * ds(2, metadata=md))

L = (c0**2 * (inner(g, v) * ds(1, metadata=md)
              - 1/c0*inner(v_n, v) * ds(2, metadata=md))
     + delta * (inner(dg, v) * ds(1, metadata=md)))

forms = [a, L]
