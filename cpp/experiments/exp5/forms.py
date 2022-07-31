import basix.ufl_wrapper
from ufl import (VectorElement, Mesh, FunctionSpace, Constant, Coefficient,
                 TestFunction, hexahedron, inner, grad, ds, dx)

P = 4  # Degree of polynomial
Q = 5  # Number of quadrature points

element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, P,
    basix.LagrangeVariant.gll_warped)
element_DG = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, 0,
    basix.LagrangeVariant.gll_warped,
    basix.DPCVariant.unset, True)
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)
V_DG = FunctionSpace(mesh, element_DG)

c0 = Coefficient(V_DG)
delta = Coefficient(V_DG)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
dg = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Forms
a = (inner(u, v) * dx(metadata=md)
     + delta / c0 * inner(u, v) * ds(2, metadata=md))

L = (- c0 * c0 * inner(grad(u_n), grad(v)) * dx(metadata=md)
     + c0 * c0 * inner(g, v) * ds(1, metadata=md)
     - c0 * inner(v_n, v) * ds(2, metadata=md)
     - delta * inner(grad(v_n), grad(v)) * dx(metadata=md)
     + delta * inner(dg, v) * ds(1, metadata=md))

forms = [a, L]