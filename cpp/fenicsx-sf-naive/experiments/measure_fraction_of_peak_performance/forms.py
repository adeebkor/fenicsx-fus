import basix.ufl_wrapper
from ufl import (VectorElement, Mesh, FunctionSpace, Coefficient, TestFunction,
                 hexahedron, inner, grad, ds, dx)

P = 6  # Degree of polynomial basis
Q = P + 1  # Number of quadrature points

print(f"Polynomial degree: {P}")

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

u = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define forms
a = inner(u/rho0/c0/c0, v) * dx(metadata=md)

forms = [a]