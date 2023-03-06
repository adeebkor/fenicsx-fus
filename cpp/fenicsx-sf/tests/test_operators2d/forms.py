import basix.ufl_wrapper
from ufl import (VectorElement, Mesh, FunctionSpace, Coefficient, TestFunction, 
                 quadrilateral, inner, grad, dx)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points
G = 2  # Mesh geometry order

# Define mesh and finite element
coord_element = VectorElement("Lagrange", quadrilateral, G)
mesh = Mesh(coord_element)
element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.quadrilateral, P,
    basix.LagrangeVariant.gll_warped)
element_DG = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.quadrilateral, 0,
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

# Define operators
m = inner(u/rho0/c0/c0, v) * dx(metadata=md)
s = - inner(1/rho0*grad(u), grad(v)) * dx(metadata=md)

f0 = Coefficient(V)
f1 = Coefficient(V)

E = inner(f1 - f0, f1 - f0) * dx

forms = [m, s, E]
