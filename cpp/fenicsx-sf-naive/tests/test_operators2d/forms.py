import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TestFunction,
                 grad, dx, inner)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points
G = 1  # Mesh geometry order

# Define mesh and finite element
coord_element = element("Lagrange", "quadrilateral", G, shape=(2, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.quadrilateral, P,
    basix.LagrangeVariant.gll_warped)
e_DG = element(basix.ElementFamily.P, basix.CellType.quadrilateral, 0,
    basix.LagrangeVariant.gll_warped, basix.DPCVariant.unset, True)

# Define function spaces
V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

c0 = Coefficient(V_DG)
rho0 = Coefficient(V_DG)
delta0 = Coefficient(V_DG)
beta0 = Coefficient(V_DG)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define operators
m1 = inner(u/rho0/c0/c0, v) * dx(metadata=md)
m2 = - inner(2.0*beta0/rho0/rho0/c0/c0/c0/c0*u_n*u, v) * dx(metadata=md)
m3 = inner(2.0*beta0/rho0/rho0/c0/c0/c0/c0*u_n*u_n, v) * dx(metadata=md)
b1 = - inner(1.0/rho0*grad(v_n), grad(v)) * dx(metadata=md)
b2 = - inner(delta0/rho0/c0/c0*grad(v_n), grad(v)) * dx(metadata=md)

f0 = Coefficient(V)
f1 = Coefficient(V)

E = inner(f1 - f0, f1 - f0) * dx

forms = [m1, m2, m3, b1, b2, E]
