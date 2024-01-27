import basix.ufl_wrapper
from ufl import (Coefficient, FunctionSpace, Mesh, TestFunction, VectorElement,
                 ds, dx, grad, hexahedron, inner)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points

# Define mesh and finite element
coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)
element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, P,
    basix.LagrangeVariant.gll_warped)
element_DG = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.hexahedron, 0,
    basix.LagrangeVariant.gll_warped, basix.DPCVariant.unset, True)

# Define function spaces
V = FunctionSpace(mesh, element)
V_DG = FunctionSpace(mesh, element_DG)

c0 = Coefficient(V_DG)
rho0 = Coefficient(V_DG)
delta0 = Coefficient(V_DG)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
dg = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define forms
a = inner(u/rho0/c0/c0, v) * dx(metadata=md) \
    + inner(delta0/rho0/c0/c0/c0*u, v) * ds(2, metadata=md)

L = - inner(1/rho0*grad(u_n), grad(v)) * dx(metadata=md) \
    + inner(1/rho0*g, v)*ds(1, metadata=md) \
    - inner(1/rho0/c0*v_n, v)*ds(2, metadata=md) \
    - inner(delta0/rho0/c0/c0*grad(v_n), grad(v)) * dx(metadata=md) \
    + inner(delta0/rho0/c0/c0*dg, v) * ds(1, metadata=md)

forms = [a, L]