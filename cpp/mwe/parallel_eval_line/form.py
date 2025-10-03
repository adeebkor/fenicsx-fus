import basix
from basix.ufl import element
from ufl import (Mesh, FunctionSpace, Constant, Coefficient,
                 TestFunction, inner, grad, ds, dx)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points
G = 1  # Order of mesh geometry

# Define mesh and finite element
coord_element = element("Lagrange", "quadrilateral", G, shape=(2, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.quadrilateral, P,
    basix.LagrangeVariant.gll_warped)

# Define function space
V = FunctionSpace(mesh, e)

# Define functions
c0 = Constant(mesh)
u = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

a = - c0**2 * inner(grad(u), grad(v)) * dx(metadata=md)

forms = [a]
