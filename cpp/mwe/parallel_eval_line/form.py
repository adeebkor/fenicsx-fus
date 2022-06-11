import basix.ufl_wrapper
from ufl import (VectorElement, Mesh, FunctionSpace, Constant, Coefficient,
                 TestFunction, quadrilateral, inner, grad, ds, dx)

element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P, basix.CellType.quadrilateral, 4,
    basix.LagrangeVariant.gll_warped)
coord_element = VectorElement("Lagrange", quadrilateral, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

c0 = Constant(mesh)
u = Coefficient(V)
v = TestFunction(V)

md = {"quadrature_rule": "GLL", "quadrature_degree": 6}

a = - c0**2 * inner(grad(u), grad(v)) * dx(metadata=md)

forms = [a]
