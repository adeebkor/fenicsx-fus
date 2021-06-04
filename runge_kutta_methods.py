import numpy as np
from mpi4py import MPI

import dolfinx.io


def solve1(f, u, dt, num_steps, rk_order):
    """Solve 1st order time dependent PDE using the Runge Kutta method"""

    # Create solution vectors at RK intermediate stages
    un = u.vector.copy()

    # Solution at start of time step
    u0 = u.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(rk_order)

    # Create lists to hold intermediate vectors
    ku = n_RK * [u0.copy()]

    file = dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, "diffusion_rk{}.xdmf".format(rk_order), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=0.0)

    t = 0.0
    for step in range(num_steps):
        print("Time step:", step, t, dt)

        # Store solution at start of time step as u0
        u.vector.copy(result=u0)

        # Runge-Kutta step
        for i in range(n_RK):
            u0.copy(result=un)

            for j in range(i):
                a = dt * a_runge[i, j]
                un.axpy(a, ku[j])

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f(tn, un, result=ku[i])

            # Update solution
            u.vector.axpy(dt * b_runge[i], ku[i])

        # Update time
        t += dt

        file.write_function(u, t=t)

    file.close()

    return u


def solve2(f0, f1, u, v, dt, num_steps, rk_order, filename=""):
    """Solve 2nd order time dependent PDE using the Runge Kutta method"""

    # Create solution vectors at RK intermediate stages
    un, vn = u.vector.copy(), v.vector.copy()

    # Solution at start of time step
    u0, v0 = u.vector.copy(), v.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(rk_order)

    # Create lists to hold intermediate vectors
    ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

    file = dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, "{}.xdmf".format(filename), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=0.0)

    t = 0.0
    for step in range(num_steps):
        # print("Time step:", step, t, dt)

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

        print("Steps:", step)
        
        # Update time
        t += dt

        if step%50 == 0:
            file.write_function(u, t=t)

    file.close()

    return u


def ode231(f, u, start_time, final_time, fname):
    """Solve 1st order in time PDE problem using adaptive step size."""

    # Create solution vectors at RK intermediate stages
    un = u.vector.copy()

    # Solution at start of time step
    u0 = u.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(23)
    CT = [-5/72, 1/12, 1/9, -1/8]

    # Create lists to hold intermediate k values
    ku = n_RK * [u0.copy()]

    file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=start_time)

    eps = 1e-5
    t = start_time
    dt = 1e-1
    step = 0
    while t < final_time:
        TE = 0.0
        dt = min(dt, final_time-t)

        # Store solution at start of time step as u0
        u.vector.copy(result=u0)

        # Runge-Kutta step
        for i in range(n_RK):
            u0.copy(result=un)

            for j in range(i):
                a = dt * a_runge[i, j]
                un.axpy(a, ku[j])

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f(tn, un, result=ku[i])

            # Compute truncation error
            TE += dt*CT[i]*ku[i]

        TE = TE.norm()
        dt = 0.9*dt*(eps/TE)**(1/5)

        if TE <= eps:
            # Update time
            t += dt
            step += 1
            print("Time step:", step, t, dt)

            for i in range(n_RK):
                u.vector.axpy(dt * b_runge[i], ku[i])

            file.write_function(u, t=t)

    file.close()

    return u


def ode232(f0, f1, u, v, start_time, final_time, fname):
    """Solve 2nd order in time PDE problem using adaptive step size."""

    # Create solution vectors at RK intermediate stages
    un, vn = u.vector.copy(), v.vector.copy()

    # Solution at start of time step
    u0, v0 = u.vector.copy(), v.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(23)
    CT = [-5/72, 1/12, 1/9, -1/8]    

    # Create lists to hold intermediate k values
    ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

    file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=start_time)

    eps = 1e-5
    t = start_time
    dt = 1e-1
    step = 0
    while t < final_time:
        TE = 0.0
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

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f0(tn, un, vn, result=ku[i])
            kv[i] = f1(tn, un, vn, result=kv[i])

            # Compute truncation error
            TE += dt*CT[i]*ku[i]

        TE = TE.norm()
        dt = 0.9*dt*(eps/TE)**(1/3)

        if TE <= eps:
            # Update time
            t += dt
            step += 1
            print("Time step:", step, t, dt)

            for i in range(n_RK):
                u.vector.axpy(dt * b_runge[i], ku[i])
                v.vector.axpy(dt * b_runge[i], kv[i])

            file.write_function(u, t=t)

    file.close()

    return u


def ode451(f, u, start_time, final_time, fname):
    """Solve 1st order in time PDE problem using adaptive step size."""

    # Create solution vectors at RK intermediate stages
    un = u.vector.copy()

    # Solution at start of time step
    u0 = u.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(5)
    CT = [1/360, 0.0, -128/4275, -2187/75240, 1/50, 2/55]

    # Create lists to hold intermediate k values
    ku = n_RK * [u0.copy()]

    file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=start_time)

    eps = 1e-5
    t = start_time
    dt = 1e-1
    step = 0
    while t < final_time:
        TE = 0.0
        dt = min(dt, final_time-t)

        # Store solution at start of time step as u0
        u.vector.copy(result=u0)

        # Runge-Kutta step
        for i in range(n_RK):
            u0.copy(result=un)

            for j in range(i):
                a = dt * a_runge[i, j]
                un.axpy(a, ku[j])

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f(tn, un, result=ku[i])

            # Compute truncation error
            TE += dt*CT[i]*ku[i]

        TE = TE.norm()
        dt = 0.9*dt*(eps/TE)**(1/5)

        if TE <= eps:
            # Update time
            t += dt
            step += 1
            print("Time step:", step, t, dt)

            for i in range(n_RK):
                u.vector.axpy(dt * b_runge[i], ku[i])

            file.write_function(u, t=t)

    file.close()

    return u


def ode452(f0, f1, u, v, start_time, final_time, fname):
    """Solve 2nd order in time PDE problem using adaptive step size."""

    # Create solution vectors at RK intermediate stages
    un, vn = u.vector.copy(), v.vector.copy()

    # Solution at start of time step
    u0, v0 = u.vector.copy(), v.vector.copy()

    # Get Runge-Kutta timestepping data
    n_RK, a_runge, b_runge, c_runge = butcher(5)
    CT = [1/360, 0.0, -128/4275, -2187/75240, 1/50, 2/55]

    # Create lists to hold intermediate k values
    ku, kv = n_RK * [u0.copy()], n_RK * [v0.copy()]

    file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "{}.xdmf".format(fname), "w")
    file.write_mesh(u.function_space.mesh)
    file.write_function(u, t=start_time)

    eps = 1e-3
    t = start_time
    dt = 1e-1
    step = 0
    while t < final_time:
        TE = 0.0
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

            # RK evaluation time
            tn = t + c_runge[i]*dt

            # Compute RHS vector
            ku[i] = f0(tn, un, vn, result=ku[i])
            kv[i] = f1(tn, un, vn, result=kv[i])

            # Compute truncation error
            TE += dt*CT[i]*ku[i]

        TE = TE.norm()
        dt = 0.9*dt*(eps/TE)**(1/5)

        if TE <= eps:
            # Update time
            t += dt
            step += 1
            print("Time step:", step, t, dt)

            for i in range(n_RK):
                u.vector.axpy(dt * b_runge[i], ku[i])
                v.vector.axpy(dt * b_runge[i], kv[i])

            if step%100 == 0:
                file.write_function(u, t=t)

    file.close()

    return u


def butcher(order):
    """Butcher table data"""
    if order == 2:
        # Explicit trapezium method
        n_RK = 2
        b_runge = [1 / 2, 1 / 2]
        c_runge = [0, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1
    elif order == 3:
        # Third-order RK3
        n_RK = 3
        b_runge = [1 / 6, 4 / 6, 1 / 6]
        c_runge = [0, 1 / 2, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 0] = -1
        a_runge[2, 1] = 2
    elif order == 4:
        # "Classical" 4th-order Runge-Kutta method
        n_RK = 4
        b_runge = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        c_runge = np.array([0, 1 / 2, 1 / 2, 1])
        a_runge = np.zeros((4, 4), dtype=np.float)
        a_runge[1, 0] = 1 / 2
        a_runge[2, 1] = 1 / 2
        a_runge[3, 2] = 1
    elif order == 5:
        # Fehlberg 5th order
        n_RK = 6
        b_runge = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]
        c_runge = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1 / 4
        a_runge[2, 0:2] = [3 / 32, 9 / 32]
        a_runge[3, 0:3] = [1932 / 2197, -7200 / 2197, 7296 / 2197]
        a_runge[4, 0:4] = [439 / 216, -8, 3680 / 513, -845 / 4104]
        a_runge[5, 0:5] = [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
    elif order == "Dormand-Prince":
        # Dormand-Prince method
        n_RK = 7
        b_runge = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        c_runge = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1 / 5
        a_runge[2, 0:2] = [3/40, 9/40]
        a_runge[3, 0:3] = [44/45, -56/15, 32/9]
        a_runge[4, 0:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
        a_runge[5, 0:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        a_runge[6, 0:6] = b_runge[:6]
    elif order == 23:
        # Bogacki-Shampine method
        n_RK = 4
        b_runge = [2/9, 1/3, 4/9, 0]
        c_runge = [0, 1/2, 3/4, 1]
        a_runge = np.zeros((n_RK, n_RK), dtype=np.float)
        a_runge[1, 0] = 1/2
        a_runge[2, 0:2] = [0, 3/4]
        a_runge[3, 0:3] = [2/9, 1/3, 4/9]
    
    return n_RK, a_runge, b_runge, c_runge
