from ufl import (FiniteElement, hexahedron, VectorElement, Mesh, FunctionSpace,
                 Constant, Coefficient, TestFunction, inner, dx, ds, grad)

element = FiniteElement("Lagrange", hexahedron, 4, variant="gll")
element_interp = FiniteElement("Lagrange", hexahedron, 2, variant="gll")

coord_element = VectorElement("Lagrange", hexahedron, 1)
mesh = Mesh(coord_element)
V = FunctionSpace(mesh, element)
V_interp = FunctionSpace(mesh, element_interp)

c0 = Constant(mesh)

u = Coefficient(V)
u_n = Coefficient(V)
v_n = Coefficient(V)
g = Coefficient(V)
v = TestFunction(V)

u_interp = Coefficient(V_interp)

md = {"quadrature_rule": "GLL", "quadrature_degree": 6}

a = inner(u, v) * dx(metadata=md)

a_interp = inner(u_interp, u_interp) * dx(metadata=md)

L = c0**2 * (- inner(grad(u_n), grad(v))
             * dx(metadata=md)
             + inner(g, v)
             * ds(1, metadata=md)
             - 1/c0*inner(v_n, v)
             * ds(2, metadata=md))

forms = [a, a_interp, L]
