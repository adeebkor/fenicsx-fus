import time
import json

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import UnitSquareMesh, FunctionSpace, Function, Form
from dolfinx.cpp.common import ScatterMode
from dolfinx.fem import assemble_matrix, assemble_vector, assemble_scalar
from dolfinx.cpp.mesh import CellType
from ufl import inner, TrialFunction, TestFunction, dx, action

results = {"Dimension": [], "Degree": [], "Number of cells": [],
           "Degrees of freedom": [], "Number of quadrature points": [],
           "Options": [], 
           "Time (PETSc)": [], "Time (ufl.action)": [],
           "L2 error (PETSc)": [], "L2 error (ufl.action)": []}

optimization_options = ["-O1", "-O2", "-O3", "-Ofast"]
N = [4, 8, 16, 32, 64, 128, 256]
p = [2, 3, 4, 5]
qp = [3, 4, 6, 8]

for opt in optimization_options:
    jit_parameters = {"cffi_extra_compile_args": ["-march=native", opt], 
                      "cffi_libraries": ["m"]}
    for degree, qdegree in zip(p, qp):
        for n in N:
            mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n,
                                  cell_type=CellType.quadrilateral)
            tdim = mesh.topology.dim
            K = mesh.topology.index_map(tdim).size_global
            V = FunctionSpace(mesh, ("Lagrange", degree))
            dof = V.dofmap.index_map.size_global

            u = TrialFunction(V)
            v = TestFunction(V)

            a_ = inner(u, v)*dx(metadata={"quadrature_rule": "GLL",
                                          "quadrature_degree": qdegree})
            a = Form(a_, jit_parameters=jit_parameters)

            f = Function(V)
            f.interpolate(lambda x: 2 + np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

            L_ = inner(f, v)*dx(metadata={"quadrature_rule": "GLL",
                                          "quadrature_degree": qdegree})
            L = Form(L_, jit_parameters=jit_parameters)

            b = Function(V)
            b.x.array[:] = 0
            assemble_vector(b.vector, L)
            b.x.scatter_reverse(ScatterMode.add)
            b.x.scatter_forward()

            # PETSc
            uh = Function(V)

            ksp = PETSc.KSP().create(MPI.COMM_WORLD)
            ksp.setType("preonly")
            ksp.getPC().setType("jacobi")

            # Assemble + Solve time
            ts_petsc = time.time()
            A = assemble_matrix(a)
            A.assemble()

            ksp.setOperators(A)
            ksp.solve(b.vector, uh.vector)

            uh.x.scatter_forward()
            te_petsc = time.time() - ts_petsc

            diff_petsc = uh - f
            L2_error_petsc = mesh.mpi_comm().allreduce(
                assemble_scalar(inner(diff_petsc, diff_petsc) * dx),
                op=MPI.SUM)

            # ufl.action 
            un = Function(V)

            b_act = Function(V)
            b_act.x.array[:] = 0

            uh_act = Function(V)

            # 'Assemble' + Solve time
            ts_ufl = time.time()

            # 'Assemble'
            un.x.array[:] = 1 / b.x.array
            a_act_ = action(a_, un)
            a_act = Form(a_act_, jit_parameters=jit_parameters)
            assemble_vector(b_act.vector, a_act)
            b_act.x.scatter_reverse(ScatterMode.add)
            b_act.x.scatter_forward()

            # 'Solve'
            uh_act.x.array[:] = 1 / b_act.x.array[:]

            te_ufl = time.time() - ts_ufl

            diff_act = uh_act - f
            L2_error_act = mesh.mpi_comm().allreduce(
                assemble_scalar(inner(diff_act, diff_act) * dx), op=MPI.SUM)

            print(
                "Option: {}\t\t N: {}\t\t Degree: {}\t\t DOF: {}\t\t L2 error: {}".format(
                opt, n, degree, dof, L2_error_act
            ))
            
            # Data
            results["Dimension"].append(2)
            results["Degree"].append(degree)
            results["Number of cells"].append(K)
            results["Degrees of freedom"].append(dof)
            results["Number of quadrature points"].append((degree+1))
            results["Options"].append(opt)
            results["Time (PETSc)"].append(te_petsc)
            results["Time (ufl.action)"].append(te_ufl)
            results["L2 error (PETSc)"].append(L2_error_petsc)
            results["L2 error (ufl.action)"].append(L2_error_act)

with open("examples/data_2d.json", "w") as data:
    json.dump(results, data)
