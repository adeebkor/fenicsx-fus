import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.io import XDMFFile


def RKF(f0, f1, u, v, start_time, final_time, fname=None):
	"""
	Solve 2nd order in time PDE problem using adaptive step size.
	"""

	# Create solution vectors at RK intermediate stages
	un, vn = u.vector.copy(), v.vector.copy()

	# Solution at start of time step
	u0, v0 = u.vector.copy(), v.vector.copy()

	# Get timestepping data
	n_RK, a_runge, b_runge, c_runge = butcher("RKF")
	CT = [1/360, 0, -128/4275, -2187/75240, 1/50, 2/55]

	# Create lists to hold intermediate k values
	ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

	# Create file to store solutions
	if fname is not None:
		file = XDMFFile(
			MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
		file.write_mesh(u.function_space.mesh)
		file.write_function(u, t=start_time)

	eps = 1E-3
	dt = 1E-2
	t = start_time
	step = 1
	TE = u0.copy()
	
	while t < final_time:
		TE.set(0.0)
		dt = min(dt, final_time-t)

		# Store solution at start of time step as u0
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

			# RK evaluation in time
			tn = t + c_runge[i] * dt

			# Compute RHS vector
			ku[i] = f0(tn, un, vn, result=ku[i])
			kv[i] = f1(tn, un, vn, result=kv[i])

			# Compute truncation error
			TE.axpy(dt * CT[i], ku[i])

		R = TE.norm()
		delta = 0.84 * (eps / R) ** (1/4)

		if R < eps:
			t += dt
			step += 1

			PETSc.Sys.syncPrint(
				"Time:{},\t Final time: {}".format(t, final_time))

			# Update u and v
			for i in range(n_RK):
				u.vector.axpy(dt * b_runge[i], ku[i])
				v.vector.axpy(dt * b_runge[i], kv[i])

			if step % 100 == 0 and fname is not None:
				file.write_function(u, t=t)

		dt = delta * dt

	if fname is not None:
		file.close()

	return u, t


def DP5(f0, f1, u, v, start_time, final_time, fname=None):
	"""
	Solve 2nd order in time PDE problem using adaptive step size.
	"""

	# Create solution vectors at RK intermediate stages
	un, vn = u.vector.copy(), v.vector.copy()

	# Solution at start of time step
	u0, v0 = u.vector.copy(), v.vector.copy()

	# Get timestepping data
	n_RK, a_runge, b_runge, c_runge = butcher("DP5")
	CT = [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]

	# Create lists to hold intermediate k values
	ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

	# Create file to store solutions
	if fname is not None:
		file = XDMFFile(
			MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
		file.write_mesh(u.function_space.mesh)
		file.write_function(u, t=start_time)

	eps = 1E-4
	dt = 1E-2
	t = start_time
	step = 1
	TE = u0.copy()
	
	while t < final_time:
		TE.set(0.0)
		dt = min(dt, final_time-t)

		# Store solution at start of time step as u0
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

			# RK evaluation in time
			tn = t + c_runge[i] * dt

			# Compute RHS vector
			ku[i] = f0(tn, un, vn, result=ku[i])
			kv[i] = f1(tn, un, vn, result=kv[i])

			# Compute truncation error
			TE.axpy(dt * CT[i], ku[i])

		R = TE.norm()
		delta = 0.9 * (eps / R) ** (1/6)

		if R < eps:
			t += dt
			step += 1

			PETSc.Sys.syncPrint(
				"Step: {},\t Time:{},\t Final time: {}".format(
					str(step).zfill(5), t, final_time))

			# Update u and v
			for i in range(n_RK):
				u.vector.axpy(dt * b_runge[i], ku[i])
				v.vector.axpy(dt * b_runge[i], kv[i])

			if step % 100 == 0 and fname is not None:
				file.write_function(u, t=t)

		dt = delta * dt

	if fname is not None:
		file.close()

	return u, t


def Tsit5(f0, f1, u, v, start_time, final_time, eps, fname=None):
	"""
	Solve 2nd order in time PDE problem using adaptive step size.
	"""

	# Create solution vectors at RK intermediate stages
	un, vn = u.vector.copy(), v.vector.copy()

	# Solution at start of time step
	u0, v0 = u.vector.copy(), v.vector.copy()

	# Get timestepping data
	n_RK, a_runge, b_runge, c_runge = butcher("Tsit5")
	CT = [-0.001780011052226, -0.000816434459657, 0.007880878010262, -0.144711007173263, 0.582357165452555, -0.458082105929187, 1/66]

	# Create lists to hold intermediate k values
	ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

	# Create file to store solutions
	if fname is not None:
		file = XDMFFile(
			MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
		file.write_mesh(u.function_space.mesh)
		file.write_function(u, t=start_time)

	# eps = 1E-3
	dt = 1E-2
	t = start_time
	step = 1
	TE = u0.copy()
	
	while t < final_time:
		TE.set(0.0)
		dt = min(dt, final_time-t)

		# Store solution at start of time step as u0
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

			# RK evaluation in time
			tn = t + c_runge[i] * dt

			# Compute RHS vector
			ku[i] = f0(tn, un, vn, result=ku[i])
			kv[i] = f1(tn, un, vn, result=kv[i])

			# Compute truncation error
			TE.axpy(dt * CT[i], ku[i])

		R = TE.norm()
		delta = 0.9 * (eps / R) ** (1/6)

		if R < eps:
			t += dt
			step += 1

			if step % 1000 == 0:
			    PETSc.Sys.syncPrint(
			    	"Step: {},\t Time:{},\t Final time: {}".format(
			    		str(step).zfill(5), t, final_time))

			# Update u and v
			for i in range(n_RK):
				u.vector.axpy(dt * b_runge[i], ku[i])
				v.vector.axpy(dt * b_runge[i], kv[i])

			if step % 100 == 0 and fname is not None:
				file.write_function(u, t=t)

		dt = delta * dt

	if fname is not None:
		file.close()

	return u, t, step


def butcher(order):
    """
    Butcher table data
    """
    if order == "Trapezium":
        # Explicit trapezium method
        n_RK = 2
        b_runge = [1 / 2, 1 / 2]
        c_runge = [0, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1
    elif order == "RK3":
        # Third-order RK3
        n_RK = 3
        b_runge = [1 / 6, 4 / 6, 1 / 6]
        c_runge = [0, 1 / 2, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 0] = -1
        a_runge[2, 1] = 2
    elif order == "RK4":
        # "Classical" 4th-order Runge-Kutta method
        n_RK = 4
        b_runge = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        c_runge = np.array([0, 1 / 2, 1 / 2, 1])
        a_runge = np.zeros((4, 4), dtype=float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 1] = 1 / 2
        a_runge[3, 2] = 1
    elif order == "BS23":
        # Bogacki-Shampine method
        n_RK = 4
        b_runge = [2/9, 1/3, 4/9, 0]
        c_runge = [0, 1/2, 3/4, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1/2
        a_runge[2, 0:2] = [0, 3/4]
        a_runge[3, 0:3] = [2/9, 1/3, 4/9]
    elif order == "RKF":
        # Fehlberg 5th order
        n_RK = 6
        b_runge = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]
        c_runge = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1 / 4
        a_runge[2, 0:2] = [3 / 32, 9 / 32]
        a_runge[3, 0:3] = [1932 / 2197, -7200 / 2197, 7296 / 2197]
        a_runge[4, 0:4] = [439 / 216, -8, 3680 / 513, -845 / 4104]
        a_runge[5, 0:5] = [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
    elif order == "DP5":
        # Dormand-Prince method
        n_RK = 7
        b_runge = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        c_runge = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[1, 0] = 1 / 5
        a_runge[2, 0:2] = [3/40, 9/40]
        a_runge[3, 0:3] = [44/45, -56/15, 32/9]
        a_runge[4, 0:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
        a_runge[5, 0:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        a_runge[6, 0:6] = b_runge[:6]
    elif order == "Tsit5":
        # Tsit5
        n_RK = 7
        c_runge = np.array([0, 0.161, 0.327, 0.9, 0.9800255409045097, 1, 1])
        b_runge = np.array([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0])
        a_runge = np.zeros((n_RK, n_RK), dtype=float)
        a_runge[2:5, 1] = [0.3354806554923570, -6.359448489975075, -11.74888356406283]
        a_runge[3:5, 2] = [4.362295432869581, 7.495539342889836]
        a_runge[4, 3] = -0.09249506636175525
        a_runge[5, 1:5] = [-12.92096931784711, 8.159367898576159, -0.07158497328140100, -0.02826905039406838]
        for i in range(1, 6):
            a_runge[i, 0] = c_runge[i] - a_runge[i, :].sum()
        a_runge[6, :] = b_runge
    
    return n_RK, a_runge, b_runge, c_runge
