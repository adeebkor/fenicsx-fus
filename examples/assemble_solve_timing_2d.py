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
           "Time (matrix)": [], "Time (vector)": [], "Time (ufl.action)": [],
           "L2 error (matrix)": [], "L2 error (vector)": [], "L2 error (ufl.action)": []}

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

            md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree}

            # Create RHS
            v = TestFunction(V)

            f = Function(V)
            f.interpolate(lambda x: 2 + np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]))

            L_ = inner(f, v)*dx(metadata=md)
            L = Form(L_, jit_parameters=jit_parameters)

            # PETSc matrix
            um = TrialFunction(V)

            am_ = inner(um, v)*dx(metadata=md)
            am = Form(am_, jit_parameters=jit_parameters)

            bm = Function(V)

            ksp = PETSc.KSP().create(MPI.COMM_WORLD)
            ksp.setType("preonly")
            ksp.getPC().setType("jacobi")

            uhm = Function(V)

            ts_petscm = time.time()
            Am = assemble_matrix(am)
            Am.assemble()

            bm = assemble_vector(L)
            bm.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

            ksp.setOperators(Am)
            ksp.solve(bm, uhm.vector)

            uhm.x.scatter_forward()
            te_petscm = time.time()-ts_petscm

            diff_petscm = uhm - f
            L2_error_petscm = mesh.mpi_comm().allreduce(
                assemble_scalar(inner(diff_petscm, diff_petscm) * dx),
                op=MPI.SUM)

            # PETSc vector
            uv = Function(V)
            uv.x.array[:] = 1

            av_ = inner(uv, v)*dx(metadata=md)
            av = Form(av_, jit_parameters=jit_parameters)

            uhv = Function(V)

            ts_petscv = time.time()
            Av = assemble_vector(av)
            Av.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

            bv = assemble_vector(L)
            bv.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)

            uhv.vector.pointwiseDivide(bv, Av)
            uhv.x.scatter_forward()
            te_petscv = time.time()-ts_petscv

            diff_petscv = uhv - f
            L2_error_petscv = mesh.mpi_comm().allreduce(
                assemble_scalar(inner(diff_petscv, diff_petscv) * dx),
                op=MPI.SUM)

            # ufl.action 
            un = Function(V)
            un.x.array[:] = 1

            uh_act = Function(V)

            ts_ufl = time.time()
            b_ufl = assemble_vector(L)
            b_ufl.ghostUpdate(addv=PETSc.InsertMode.ADD,
                              mode=PETSc.ScatterMode.REVERSE)

            a_act_ = action(am_, un)
            a_act = Form(a_act_, jit_parameters=jit_parameters)
            A_act = assemble_vector(a_act)
            A_act.ghostUpdate(addv=PETSc.InsertMode.ADD,
                              mode=PETSc.ScatterMode.REVERSE)
            
            uh_act.vector.pointwiseDivide(b_ufl, A_act)
            te_ufl = time.time() - ts_ufl

            diff_act = uh_act - f
            L2_error_act = mesh.mpi_comm().allreduce(
                assemble_scalar(inner(diff_act, diff_act) * dx), op=MPI.SUM)

            PETSc.Sys.syncPrint(
                "Option: {} N: {}\t Degree: {} DOF: {}\t L2 error: {}, {}, {}".format(
                opt, n, degree, dof, L2_error_petscm, L2_error_petscv, L2_error_act
            ))
            
            # Data
            results["Dimension"].append(2)
            results["Degree"].append(degree)
            results["Number of cells"].append(K)
            results["Degrees of freedom"].append(dof)
            results["Number of quadrature points"].append((degree+1))
            results["Options"].append(opt)
            results["Time (matrix)"].append(te_petscm)
            results["Time (vector)"].append(te_petscv)
            results["Time (ufl.action)"].append(te_ufl)
            results["L2 error (matrix)"].append(L2_error_petscm)
            results["L2 error (vector)"].append(L2_error_petscv)
            results["L2 error (ufl.action)"].append(L2_error_act)

with open("examples/data_2d.json", "w") as data:
    json.dump(results, data)
