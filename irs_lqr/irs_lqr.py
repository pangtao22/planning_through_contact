import time

import numpy as np
from irs_lqr.tv_lqr import solve_tvlqr, get_solver


class IrsLqrParameters:
    """
    Parmeters class for IrsLqr.

    Q (np.array, shape n x n): cost matrix for state.
    Qd (np.array, shape n x n): cost matrix for final state.
    R (np.array, shape m x m): cost matrix for input.
    x0 (np.array, shape n): initial point in state-space.
    xd_trj (np.array, shape (T+1) x n): desired trajectory.
    u_trj_initial (np.array, shape T x m): initial guess of the input.
    xbound (np.array, shape 2 x n): (lb, ub) bounds on state.
    xbound (np.array, shape 2 x m): (lb, ub) bounds on input.
    solver (str): solver name to use for direct LQR.
    """

    def __init__(self):
        self.Q = None
        self.Qd = None
        self.R = None
        self.x0 = None
        self.xd_trj = None
        self.u_trj_initial = None
        self.xbound = None
        self.ubound = None
        self.solver_name = "osqp"


class IrsLqr:
    def __init__(self, system, params):
        """
        Base class for iterative Randomized Smoothing LQR.

        system (DynamicalSystem class): dynamics class.
        parms (IrsLqrParameters class): parameters class.
        """

        self.system = system
        self.params = params
        self.check_valid_system(self.system)
        self.check_valid_params(self.params, self.system)

        self.Q = params.Q
        self.Qd = params.Qd
        self.R = params.R
        self.x0 = params.x0
        self.xd_trj = params.xd_trj
        self.u_trj = params.u_trj_initial
        self.xbound = params.xbound
        self.ubound = params.ubound
        self.solver = get_solver(params.solver_name)

        self.T = self.u_trj.shape[0]  # horizon of the problem
        self.dim_x = self.system.dim_x
        self.dim_u = self.system.dim_u
        self.x_trj = self.rollout(self.x0, self.u_trj)
        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        # These store iterations for plotting.
        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()

        self.iter = 1

    def check_valid_system(self, system):
        """
        Check if the system is valid. Otherwise, throw exception.
        TODO(terry-suh): we can add more error checking later.        
        """
        if system.dim_x == 0:
            raise RuntimeError(
                "System has zero states. Did you forget to set dim_x?")
        elif system.dim_u == 0:
            raise RuntimeError(
                "System has zero inputs. Did you forget to set dim_u?")
        try:
            system.dynamics(np.zeros(system.dim_x), np.zeros(system.dim_u))
        except:
            raise RuntimeError(
                "Could not evaluate dynamics. Have you implemented it?")

    def check_valid_params(self, params, system):
        """
        Check if the parameter is valid. Otherwise, throw exception.
        TODO(terry-suh): we can add more error checking later.
        """
        if params.Q.shape != (system.dim_x, system.dim_x):
            raise RuntimeError(
                "Q matrix must be diagonal with dim_x x dim_x.")
        if params.Qd.shape != (system.dim_x, system.dim_x):
            raise RuntimeError(
                "Qd matrix must be diagonal with dim_x x dim_x.")
        if params.R.shape != (system.dim_u, system.dim_u):
            raise RuntimeError(
                "R matrix must be diagonal with dim_u x dim_u.")

    def rollout(self, x0, u_trj):
        """
        Given the initial state and an input trajectory, get an open-loop
        state trajectory of the system that is consistent with the dynamics
        of the system.
        - args:
            x0 (np.array, shape n): initial state.
            u_traj (np.array, shape T x m): initial input guess.
        """
        x_trj = np.zeros((self.T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(self.T):
            x_trj[t + 1, :] = self.system.dynamics(x_trj[t, :], u_trj[t, :])

        return x_trj

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n): state trajectory to evaluate cost with.
            u_trj (np.array, shape T x m): state trajectory to evaluate cost with.
        NOTE(terry-suh): this function can be jitted, but we don't do it here to minimize
        dependency.
        """
        cost = 0.0
        for t in range(self.T):
            et = x_trj[t, :] - self.xd_trj[t, :]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t, :]).dot(self.R).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.xd_trj[self.T, :]
        cost += et.dot(self.Q).dot(et)
        return cost

    def get_TV_matrices(self, x_trj, u_trj):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        raise NotImplementedError("This class is virtual.")

    def local_descent(self, x_trj, u_trj):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """
        At, Bt, ct = self.get_TV_matrices(x_trj, u_trj)
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        if self.xbound is not None:
            x_bounds_abs = np.zeros((2, self.T + 1, self.dim_x))
            x_bounds_abs[0] = self.xbound[0]
            x_bounds_abs[1] = self.xbound[1]
        if self.ubound is not None:
            u_bounds_abs = np.zeros((2, self.T, self.dim_u))
            u_bounds_abs[0] = self.ubound[0]
            u_bounds_abs[1] = self.ubound[1]

        for t in range(self.T):
            x_star, u_star = solve_tvlqr(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T],
                self.Q, self.Qd, self.R,
                x0 = x_trj_new[t, :],
                x_trj_d = self.xd_trj[t:self.T + 1],
                solver=self.solver,
                indices_u_into_x=None,
                x_bound_abs=x_bounds_abs[:, t:, :] if(
                    self.xbound is not None) else None,
                u_bound_abs=u_bounds_abs[:, t:, :] if(
                    self.ubound is not None) else None)
            u_trj_new[t, :] = u_star[0]
            x_trj_new[t + 1, :] = self.system.dynamics(x_trj_new[t, :], u_trj_new[t, :])

        return x_trj_new, u_trj_new

    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        NOTE(terry-suh): originally, there is a convergence criteria.
        However, given the "non-local" nature of some randomized smoothing
        algorithms, setting such a criteria might cause it to terminate early.
        Thus we only provide a max iterations input.
        """
        while True:
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.evaluate_cost(x_trj_new, u_trj_new)

            print("Iteration: {:02d} ".format(self.iter) + " || " +
                  "Current Cost: {0:05f} ".format(cost_new) + " || " +
                  "Elapsed time: {0:05f} ".format(
                      time.time() - self.start_time))

            self.x_trj_lst.append(x_trj_new)
            self.u_trj_lst.append(u_trj_new)
            self.cost_lst.append(cost_new)

            if (self.iter > max_iterations):
                break

            # Go over to next iteration.
            self.cost = cost_new
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        return self.x_trj, self.u_trj, self.cost
