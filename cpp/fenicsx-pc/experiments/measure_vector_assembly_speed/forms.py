import basix
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Mesh, TestFunction,
                 grad, ds, dx, inner)

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points

# Define mesh and finite element
coord_element = element("Lagrange", "hexahedron", 1, shape=(3, ))
mesh = Mesh(coord_element)
e = element(basix.ElementFamily.P, basix.CellType.hexahedron, P,
    basix.LagrangeVariant.gll_warped)
e_DG = element(basix.ElementFamily.P, basix.CellType.hexahedron, 0,
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
g = Coefficient(V)
dg = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define forms
a0 = inner(u/rho0/c0/c0, v) * dx(metadata=md)
a1 = inner(delta0/rho0/c0/c0/c0*u, v) * ds(2, metadata=md)
a2 = - inner(2*beta0/rho0/rho0/c0/c0/c0/c0*u_n*u, v) * dx(metadata=md)

L0 = - inner(1/rho0*grad(u_n), grad(v)) * dx(metadata=md)
L1 = inner(1/rho0*g, v) * ds(1, metadata=md)
L2 = - inner(1/rho0/c0*v_n, v) * ds(2, metadata=md)
L3 = - inner(delta0/rho0/c0/c0*grad(v_n), grad(v)) * dx(metadata=md)
L4 = inner(delta0/rho0/c0/c0*dg, v) * ds(1, metadata=md)
L5 = inner(2*beta0/rho0/rho0/c0/c0/c0/c0*v_n*v_n, v) * dx(metadata=md)

forms = [a0, a1, a2, L0, L1, L2, L3, L4, L5]
