import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.io import XDMFFile


def solve_ibvp(f0, f1, u, v, dt, tspan, rk_type, filename=None):
	"""
	Solve 2nd order time dependent PDE using the Runge-Kutta method.
	"""

	t0, tf = tspan

	# Create solution vectors at RK intermediate stages
    un, vn = u.vector.copy(), v.vector.copy()

    # Solution at start of time step
    u0, v0 = u.vector.copy(), v.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(rk_order)

    # Create lists to hold intermediate vectors
    ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

    if filename is not None:
        file = dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, "{}.xdmf".format(filename), "w")
        file.write_mesh(u.function_space.mesh)
        file.write_function(u, t=t0)

	t = t0
	step = 0
	nstep = np.rint((tf - t0) / dt) + 1
	while t < tf:
		dt = min(dt, tf-dt)

		# Store solution at start of time step
        u.vector.copy(result=u0)
        v.vector.copy(result=v0)

        # Runge-Kutta step
        for i in range(n_RK):
            u0.copy(result=un)
            v0.copy(result=vn)

			for j in range(i):
                a = dt * a_runge[i, j]
                un.axpy(a, ku[j])
                vn.axpy(a, kv[j])

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f0(tn, un, vn, result=ku[i])
            kv[i] = f1(tn, un, vn, result=kv[i])

            # Update solution
            u.vector.axpy(dt * b_runge[i], ku[i])
            v.vector.axpy(dt * b_runge[i], kv[i])

		# Update time
		t += dt
		step += 1

		if step % 100 == 0:
			PETSc.Sys.syncPrint("Steps:{}/{}".format(step, nstep))
			if filename is not None:
				file.write_function(u, t=t)
	
	if filename is not None:
		file.close()

	return u, t, step





def butcher(order):
    """Butcher table data"""
    if order == 2:
        # Explicit trapezium method
        n_RK = 2
        b_runge = [1/2, 1/2]
        c_runge = [0, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1
    elif order == 3:
        # Third-order RK3
        n_RK = 3
        b_runge = [1/6, 4/6, 1/6]
        c_runge = [0, 1/2, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1/2
        a_runge[2, 0] = -1
        a_runge[2, 1] = 2
	elif order == "Heun3":
		# Heun's third-order method
		n_RK = 3
		b_runge = [1/4, 0, 3/4]
		c_runge = [0, 1/3, 2/3]
		a_runge = np.zeros((n_RK, n_RK), dtype=float)
		a_runge[1, 0] = 1/3
		a_runge[2, 1] = 2/3
	elif order == "Ralston3":
		# Ralston's third-order method
		n_RK = 3
		b_runge = [2/9, 1/3, 4/9]
		c_runge = [0, 1/2, 3/4]
		a_runge = np.zeros((n_RK, n_RK), dtype=float)
		a_runge[1, 0] = 1/2
		a_runge[2, 1] = 3/4
    elif order == 4:
        # "Classical" 4th-order Runge-Kutta method
        n_RK = 4
        b_runge = [1/6, 1/3, 1/3, 1/6]
        c_runge = np.array([0, 1/2, 1/2, 1])
        a_runge = np.zeros((4, 4), dtype=float)
        a_runge[1, 0] = 1/2
        a_runge[2, 1] = 1/2
        a_runge[3, 2] = 1

	return n_RK, a_runge, b_runge, c_runge