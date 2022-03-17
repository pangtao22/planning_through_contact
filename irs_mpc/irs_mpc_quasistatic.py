from typing import Dict
import time
import os

import numpy as np
import spdlog

from pydrake.all import ModelInstanceIndex
import matplotlib.pyplot as plt

from zmq_parallel_cmp.array_io import *

from .irs_mpc_params import (IrsMpcQuasistaticParameters,
                             ParallelizationMode)
from .quasistatic_dynamics import QuasistaticDynamics, GradientMode
from .quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from .mpc import solve_mpc, get_solver


def update_q_start_and_goal(
        q_start: Dict[ModelInstanceIndex, np.ndarray],
        q_goal: Dict[ModelInstanceIndex, np.ndarray],
        params: IrsMpcQuasistaticParameters,
        q_dynamics: QuasistaticDynamics,
        T: int):
    params.x0 = q_dynamics.get_x_from_q_dict(q_start)

    u0 = q_dynamics.get_u_from_q_cmd_dict(q_start)
    params.u_trj_0 = np.tile(u0, (T, 1))

    xd = q_dynamics.get_x_from_q_dict(q_goal)
    params.x_trj_d = np.tile(xd, (T + 1, 1))


class IrsMpcQuasistatic:
    def __init__(self, q_dynamics: QuasistaticDynamics,
                 params: IrsMpcQuasistaticParameters):
        """

        Arguments are similar to those of SqpLsImplicit.
        Only samples u to estimate B.
        A uses the first derivative of the dynamics at x.

        sampling receives sampling(initial_std, iter) and returns the 
        current std.
        """
        self.q_dynamics = q_dynamics
        self.dim_x = q_dynamics.dim_x
        self.dim_u = q_dynamics.dim_u

        self.irs_mpc_params = params

        self.T = params.T
        self.Q_dict = params.Q_dict
        self.Q = self.q_dynamics.get_Q_from_Q_dict(self.Q_dict)
        self.Qd_dict = params.Qd_dict
        self.Qd = self.q_dynamics.get_Q_from_Q_dict(self.Qd_dict)
        self.R_dict = params.R_dict
        self.R = self.q_dynamics.get_R_from_R_dict(self.R_dict)
        self.x_bounds_abs = params.x_bounds_abs
        self.u_bounds_abs = params.u_bounds_abs
        self.x_bounds_rel = params.x_bounds_rel
        self.u_bounds_rel = params.u_bounds_rel
        self.indices_u_into_x = q_dynamics.get_q_a_indices_into_x()

        self.publish_every_iteration = params.publish_every_iteration

        # solver
        self.solver = get_solver(params.solver_name)

        # logger
        try:
            self.logger = spdlog.ConsoleLogger("IrsMpc")
        except RuntimeError as e:
            spdlog.drop("IrsMpc")
            self.logger = spdlog.ConsoleLogger("IrsMpc")

        # parallelization.
        use_zmq_workers = (
                self.irs_mpc_params.parallel_mode ==
                ParallelizationMode.kZmq or
                self.irs_mpc_params.parallel_mode ==
                ParallelizationMode.kZmqDebug)

        self.q_dynamics_parallel = QuasistaticDynamicsParallel(
            q_dynamics=q_dynamics,
            use_zmq_workers=use_zmq_workers)

    def initialize_problem(self, x0, x_trj_d, u_trj_0):
        # initial trajectory.
        self.x0 = x0
        self.x_trj_d = x_trj_d
        self.u_trj_0 = u_trj_0
        self.x_trj = self.rollout(self.x0, self.u_trj_0)
        self.u_trj = self.u_trj_0  # T x m

        # initial cost.
        (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
         cost_R) = self.calc_cost(self.x_trj, self.u_trj)
        self.cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R

        # best cost
        self.idx_best = 0
        self.x_trj_best = None
        self.u_trj_best = None
        self.cost_best = np.inf

        # logging
        self.x_trj_list = [self.x_trj]
        self.u_trj_list = [self.u_trj]

        self.cost_all_list = [self.cost]
        self.cost_Qu_list = [cost_Qu]
        self.cost_Qu_final_list = [cost_Qu_final]
        self.cost_Qa_list = [cost_Qa]
        self.cost_Qa_final_list = [cost_Qa_final]
        self.cost_R_list = [cost_R]

        self.current_iter = 0

    def rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.q_dynamics.dynamics(
                x_trj[t, :], u_trj[t, :])
        return x_trj

    @staticmethod
    def calc_Q_cost(models_list: List[ModelInstanceIndex],
                    x_dict: Dict[ModelInstanceIndex, np.ndarray],
                    xd_dict: Dict[ModelInstanceIndex, np.ndarray],
                    Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        cost = 0.
        for model in models_list:
            x_i = x_dict[model]
            xd_i = xd_dict[model]
            Q_i = Q_dict[model]
            dx_i = x_i - xd_i
            cost += (dx_i * Q_i * dx_i).sum()

        return cost

    def calc_cost(self, x_trj, u_trj):
        T = u_trj.shape[0]
        assert T == self.T and x_trj.shape[0] == T + 1
        idx_u_into_x = self.q_dynamics.get_q_a_indices_into_x()

        # Final cost Qd.
        x_dict = self.q_dynamics.get_q_dict_from_x(x_trj[-1])
        xd_dict = self.q_dynamics.get_q_dict_from_x(self.x_trj_d[-1])
        cost_Qu_final = self.calc_Q_cost(
            models_list=self.q_dynamics.models_unactuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)
        cost_Qa_final = self.calc_Q_cost(
            models_list=self.q_dynamics.models_actuated,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)

        # Q and R costs.
        cost_Qu = 0.
        cost_Qa = 0.
        cost_R = 0.
        for t in range(T):
            x_dict = self.q_dynamics.get_q_dict_from_x(x_trj[t])
            xd_dict = self.q_dynamics.get_q_dict_from_x(self.x_trj_d[t])
            # Q cost.
            cost_Qu += self.calc_Q_cost(
                models_list=self.q_dynamics.models_unactuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)
            cost_Qa += self.calc_Q_cost(
                models_list=self.q_dynamics.models_actuated,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)

            # R cost.
            if t == 0:
                du = u_trj[t] - x_trj[t, idx_u_into_x]
            else:
                du = u_trj[t] - u_trj[t - 1]
            cost_R += du @ self.R @ du

        return cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final, cost_R

    def calc_bundled_ABc(self, x_trj: np.ndarray, u_trj: np.ndarray):
        """
        Calls the owned QuasistaticDynamicsParallel.calcBundledABc with
            parameters defined in self.params.
        :param x_trj:
        :param u_trj:
        :return:
        """
        if self.irs_mpc_params.bundle_mode == BundleMode.kFirst:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1)
            log_barrier_weight = None
        elif self.irs_mpc_params.bundle_mode == BundleMode.kFirstAnalytic:
            std_u = None
            beta = self.irs_mpc_params.log_barrier_weight_multiplier
            log_barrier_weight = (self.irs_mpc_params.log_barrier_weight_initial
                                  * (beta ** self.current_iter))
        else:
            raise NotImplementedError

        return self.q_dynamics_parallel.calc_bundled_ABc(
            x_trj=x_trj, u_trj=u_trj,
            irs_mpc_params=self.irs_mpc_params,
            std_u=std_u, log_barrier_weight=log_barrier_weight)

    def local_descent(self, x_trj: np.ndarray, u_trj: np.ndarray):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj ((T + 1), dim_x): nominal state trajectory.
            u_trj (T, dim_u) : nominal input trajectory
        """
        At, Bt, ct = self.calc_bundled_ABc(x_trj, u_trj)

        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        if self.x_bounds_abs is not None:
            # x_bounds_abs are used to establish a trust region around a current
            # trajectory.
            x_bounds_abs = np.zeros((2, self.T + 1, self.dim_x))
            x_bounds_abs[0] = x_trj + self.x_bounds_abs[0]
            x_bounds_abs[1] = x_trj + self.x_bounds_abs[1]
        if self.u_bounds_abs is not None:
            # u_bounds_abs are used to establish a trust region around a current
            # trajectory.
            u_bounds_abs = np.zeros((2, self.T, self.dim_u))
            u_bounds_abs[0] = (x_trj[:-1, self.indices_u_into_x]
                               + self.u_bounds_abs[0])
            u_bounds_abs[1] = (x_trj[:-1, self.indices_u_into_x]
                               + self.u_bounds_abs[1])
        if self.x_bounds_rel is not None:
            # this should be rarely used.
            x_bounds_rel = np.zeros((2, self.T, self.dim_x))
            x_bounds_rel[0] = self.x_bounds_rel[0]
            x_bounds_rel[1] = self.x_bounds_rel[1]
        if self.u_bounds_rel is not None:
            # u_bounds_rel are used to impose input constraints.
            u_bounds_rel = np.zeros((2, self.T, self.dim_u))
            u_bounds_rel[0] = self.u_bounds_rel[0]
            u_bounds_rel[1] = self.u_bounds_rel[1]

        for t in range(self.T):
            x_star, u_star = solve_mpc(
                At[t:self.T],
                Bt[t:self.T],
                ct[t:self.T], self.Q, self.Qd,
                self.R, x_trj_new[t, :],
                self.x_trj_d[t:],
                solver=self.solver,
                indices_u_into_x=self.indices_u_into_x,
                x_bound_abs=x_bounds_abs[:, t:, :] if (
                        self.x_bounds_abs is not None) else None,
                u_bound_abs=u_bounds_abs[:, t:, :] if (
                        self.u_bounds_abs is not None) else None,
                x_bound_rel=x_bounds_rel[:, t:, :] if (
                        self.x_bounds_rel is not None) else None,
                u_bound_rel=u_bounds_rel[:, t:, :] if (
                        self.u_bounds_rel is not None) else None,
                xinit=None,
                uinit=None)
            u_trj_new[t, :] = u_star[0]
            x_trj_new[t + 1, :] = self.q_dynamics.dynamics(
                x_trj_new[t], u_trj_new[t])

        return x_trj_new, u_trj_new

    def print_iterate_info(self):
        self.logger.info(
            'Iter {:02d}, '.format(self.current_iter) +
            'cost: {:0.4f}, '.format(self.cost) +
            'time: {:0.2f}.'.format(time.time() - self.start_time))

    def iterate(self, max_iterations: int,
                cost_Qu_f_threshold: float = 0):
        """
        Terminates after the trajectory cost is less than cost_threshold or
         max_iterations is reached.
        """
        self.start_time = time.time()
        self.print_iterate_info()

        while True:
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
             cost_R) = self.calc_cost(x_trj_new, u_trj_new)
            cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R
            self.x_trj_list.append(x_trj_new)
            self.u_trj_list.append(u_trj_new)
            self.cost_Qu_list.append(cost_Qu)
            self.cost_Qu_final_list.append(cost_Qu_final)
            self.cost_Qa_list.append(cost_Qa)
            self.cost_Qa_final_list.append(cost_Qa_final)
            self.cost_R_list.append(cost_R)
            self.cost_all_list.append(cost)

            if self.publish_every_iteration:
                self.q_dynamics.publish_trajectory(x_trj_new)

            if self.cost_best > cost:
                self.x_trj_best = x_trj_new
                self.u_trj_best = u_trj_new
                self.cost_best = cost
                self.idx_best = self.current_iter

            # Go over to next iteration.
            self.cost = cost
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.current_iter += 1
            self.print_iterate_info()

            if (self.current_iter > max_iterations
                    or cost_Qu_final < cost_Qu_f_threshold):
                break

        return self.x_trj, self.u_trj, self.cost

    def package_solution(self):
        i_best = self.idx_best
        cost = {
            "Qu": self.cost_Qu_list[i_best],
            "Qu_f": self.cost_Qu_final_list[i_best],
            "Qa": self.cost_Qa_list[i_best],
            "Qa_f": self.cost_Qa_final_list[i_best],
            "R": self.cost_R_list[i_best],
            "all": self.cost_all_list[i_best]}
        result = {'cost': cost,
                  "x_trj": np.array(self.x_trj_best),
                  "u_trj": np.array(self.u_trj_best)}
        return result

    def plot_costs(self):
        """
        Plot the costs stored in self.cost_*, which is generated by the
            previous call to self.iterate.
        """
        plt.figure()
        plt.plot(self.cost_all_list, label='all')
        plt.plot(self.cost_Qa_list, label='Qa')
        plt.plot(self.cost_Qu_list, label='Qu')
        plt.plot(self.cost_Qa_final_list, label='Qa_f')
        plt.plot(self.cost_Qu_final_list, label='Qu_f')
        plt.plot(self.cost_R_list, label='R')

        plt.title('Trajectory cost')
        plt.xlabel('Iterations')
        # plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()

