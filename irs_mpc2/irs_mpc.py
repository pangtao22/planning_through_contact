import copy, time
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import ModelInstanceIndex, GurobiSolver

from qsim.parser import QuasistaticParser
from qsim_cpp import QuasistaticSimulatorCpp, ForwardDynamicsMode, GradientMode

from .irs_mpc_params import (IrsMpcQuasistaticParameters, SmoothingMode,
                             kSmoothingMode2ForwardDynamicsModeMap,
                             RandomizedSmoothingModes, AnalyticSmoothingModes)
from .quasistatic_visualizer import QuasistaticVisualizer
from .mpc import solve_mpc


class IrsMpcQuasistatic:
    def __init__(self,
                 q_sim: QuasistaticSimulatorCpp,
                 parser: QuasistaticParser,
                 params: IrsMpcQuasistaticParameters):
        self.irs_mpc_params = params
        self.parser = parser
        self.q_sim = q_sim
        self.q_sim_batch = parser.make_batch_simulator()
        self.vis = QuasistaticVisualizer(q_parser=self.parser,
                                         q_sim=self.q_sim)
        self.plant = self.q_sim.get_plant()
        self.solver = GurobiSolver()

        # unpack various parameters for convenience.
        self.dim_x = self.q_sim.get_plant().num_positions()
        self.dim_u = self.q_sim.num_actuated_dofs()
        self.indices_u_into_x = self.q_sim.get_q_a_indices_into_q()
        # elements of Q_dict, Qd_dict and R_dict are the diagonals of the Q,
        # Qd and R matrices.
        self.Q_dict = params.Q_dict
        self.Qd_dict = params.Qd_dict
        self.R_dict = params.R_dict
        # the matrices are needed when adding quadratic costs to MPC.
        self.Q = self.get_Q_mat_from_Q_dict(self.Q_dict)
        self.Qd = self.get_Q_mat_from_Q_dict(self.Qd_dict)
        self.R = self.get_R_mat_from_R_dict(self.R_dict)
        self.x_bounds_abs = params.x_bounds_abs
        self.u_bounds_abs = params.u_bounds_abs
        self.x_bounds_rel = params.x_bounds_rel
        self.u_bounds_rel = params.u_bounds_rel

        # QuasistaticSimParameters
        self.sim_params = self.get_q_sim_params(self.irs_mpc_params)

    def get_q_sim_params(self, p_mpc: IrsMpcQuasistaticParameters):
        p = copy.deepcopy(self.parser.q_sim_params)

        p.h = p_mpc.h
        p.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            p_mpc.smoothing_mode]

        if p_mpc.use_A:
            p.gradient_mode = GradientMode.kAB
        else:
            p.gradient_mode = GradientMode.kBOnly

        return p

    def initialize_problem(self,
                           x0: np.ndarray, x_trj_d: np.ndarray,
                           u_trj_0: np.ndarray):
        self.T = u_trj_0.shape[0]
        self.x0 = x0
        self.x_trj_d = x_trj_d

        # best cost
        self.idx_best = None
        self.x_trj_best = None
        self.u_trj_best = None
        self.cost_best = np.inf

        # logging
        self.x_trj_list = []
        self.u_trj_list = []

        self.cost_all_list = []
        self.cost_Qu_list = []
        self.cost_Qu_final_list = []
        self.cost_Qa_list = []
        self.cost_Qa_final_list = []
        self.cost_R_list = []

        self.current_iter = 0

        # Initial trajectory and linearization
        self.x_trj_0 = self.rollout(x0, u_trj_0, ForwardDynamicsMode.kSocpMp)
        self.u_trj_0 = u_trj_0  # T x m
        self.A_trj, self.B_trj, self.c_trj = self.calc_bundled_ABc(
            self.x_trj_0[:self.T], self.u_trj_0)

    def get_Q_mat_from_Q_dict(self,
                              Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        Q = np.eye(self.dim_x)
        for model, idx in self.q_sim.get_position_indices().items():
            Q[idx, idx] = Q_dict[model]
        return Q

    def get_R_mat_from_R_dict(self,
                              R_dict: Dict[ModelInstanceIndex, np.ndarray]):
        R = np.eye(self.dim_u)
        i_start = 0
        for model in self.q_sim.get_actuated_models():
            n_v_i = self.plant.num_velocities(model)
            R[i_start: i_start + n_v_i, i_start: i_start + n_v_i] = \
                np.diag(R_dict[model])
            i_start += n_v_i
        return R

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
        models_u = self.q_sim.get_unactuated_models()
        models_a = self.q_sim.get_actuated_models()

        # Final cost Qd.
        x_dict = self.q_sim.get_q_dict_from_vec(x_trj[-1])
        xd_dict = self.q_sim.get_q_dict_from_vec(self.x_trj_d[-1])
        cost_Qu_final = self.calc_Q_cost(
            models_list=models_u,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)
        cost_Qa_final = self.calc_Q_cost(
            models_list=models_a,
            x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Qd_dict)

        # Q and R costs.
        cost_Qu = 0.
        cost_Qa = 0.
        cost_R = 0.
        for t in range(T):
            x_dict = self.q_sim.get_q_dict_from_vec(x_trj[t])
            xd_dict = self.q_sim.get_q_dict_from_vec(self.x_trj_d[t])
            # Q cost.
            cost_Qu += self.calc_Q_cost(
                models_list=models_u,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)
            cost_Qa += self.calc_Q_cost(
                models_list=models_a,
                x_dict=x_dict, xd_dict=xd_dict, Q_dict=self.Q_dict)

            # R cost.
            if t == 0:
                du = u_trj[t] - x_trj[t, self.indices_u_into_x]
            else:
                du = u_trj[t] - u_trj[t - 1]
            cost_R += du @ self.R @ du

        return cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final, cost_R

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

    def calc_bundled_ABc(self, x_trj: np.ndarray, u_trj: np.ndarray):
        """
        Calls the owned QuasistaticDynamicsParallel.calcBundledABc with
            parameters defined in self.params.
        :param x_trj: (T, dim_x)
        :param u_trj: (T, dim_u)
        :return:
        """
        T = len(u_trj)
        sim_p = copy.deepcopy(self.sim_params)
        if self.irs_mpc_params.smoothing_mode in RandomizedSmoothingModes:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1)
            A_trj, B_trj, c_trj = self.q_sim_batch.calc_bundled_ABc_trj(
                x_trj, u_trj, std_u, sim_p,
                self.irs_mpc_params.n_samples_randomized, None)
        else:
            if self.irs_mpc_params.smoothing_mode in AnalyticSmoothingModes:
                sim_p.log_barrier_weight = (
                    self.irs_mpc_params.calc_log_barrier_weight(
                        self.irs_mpc_params.log_barrier_weight_initial,
                        self.current_iter))

            c_trj, A_trj, B_trj, is_valid = \
                self.q_sim_batch.calc_dynamics_parallel(x_trj, u_trj, sim_p)

            if not all(is_valid):
                raise RuntimeError("analytic smoothing failed.")

        # Convert lists of 2D arrays to 3D arrays.
        if self.irs_mpc_params.use_A:
            A_trj = np.array(A_trj)
        else:
            A_trj = np.zeros((T, self.dim_x, self.dim_x))
            A_trj[:] = np.eye(A_trj.shape[1])
            A_trj[:, :, self.indices_u_into_x] = 0.0

        B_trj = np.array(B_trj)
        c_trj = np.array(c_trj)
        return A_trj, B_trj, c_trj

    def local_descent(self, x_trj: np.ndarray):
        """
        Forward pass using a TV-LQR controller on the linearized dynamics.
        - args:
            x_trj ((T + 1), dim_x): nominal state trajectory.
            u_trj (T, dim_u) : nominal input trajectory
        """
        x_trj_new = np.zeros_like(x_trj)
        x_trj_new[0] = x_trj[0]
        u_trj_new = np.zeros((self.T, self.dim_u))

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

        sim_params_rollout = copy.deepcopy(self.sim_params)
        if self.irs_mpc_params.rollout_forward_dynamics_mode:
            sim_params_rollout.forward_mode = (
                self.irs_mpc_params.rollout_forward_dynamics_mode)
        sim_params_rollout.gradient_mode = GradientMode.kNone

        for t in range(self.T):
            x_star, u_star = solve_mpc(
                At=self.A_trj[t:self.T],
                Bt=self.B_trj[t:self.T],
                ct=self.c_trj[t:self.T], Q=self.Q, Qd=self.Qd,
                R=self.R,
                x0=x_trj_new[t],
                x_trj_d=self.x_trj_d[t:],
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
            u_trj_new[t] = u_star[0]

            # Rollout and compute bundled A, B and c.
            At, Bt, ct = self.calc_bundled_ABc(
                x_trj_new[t][None, :], u_trj_new[t][None, :])

            self.A_trj[t] = At.squeeze()
            self.B_trj[t] = Bt.squeeze()
            self.c_trj[t] = ct.squeeze()

            if self.irs_mpc_params.rollout_forward_dynamics_mode:
                x_trj_new[t + 1, :] = self.q_sim.calc_dynamics(
                    x_trj_new[t], u_trj_new[t], sim_params_rollout)
            else:
                x_trj_new[t + 1] = ct.squeeze()

        return x_trj_new, u_trj_new

    def rollout(self, x0: np.ndarray, u_trj: np.ndarray,
                forward_mode: ForwardDynamicsMode):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0

        sim_p = copy.deepcopy(self.sim_params)
        sim_p.forward_mode = forward_mode
        sim_p.gradient_mode = GradientMode.kNone

        for t in range(T):
            x_trj[t + 1, :] = self.q_sim.calc_dynamics(
                x_trj[t, :], u_trj[t, :], sim_p)
        return x_trj

    def iterate(self, max_iterations: int,
                cost_Qu_f_threshold: float = 0):
        """
        Terminates after the trajectory cost is less than cost_threshold or
         max_iterations is reached.
        """
        start_time = time.time()
        x_trj = np.array(self.x_trj_0)
        u_trj = np.array(self.u_trj_0)

        while True:
            (cost_Qu, cost_Qu_final, cost_Qa, cost_Qa_final,
             cost_R) = self.calc_cost(x_trj, u_trj)
            cost = cost_Qu + cost_Qu_final + cost_Qa + cost_Qa_final + cost_R
            self.x_trj_list.append(x_trj)
            self.u_trj_list.append(u_trj)
            self.cost_Qu_list.append(cost_Qu)
            self.cost_Qu_final_list.append(cost_Qu_final)
            self.cost_Qa_list.append(cost_Qa)
            self.cost_Qa_final_list.append(cost_Qa_final)
            self.cost_R_list.append(cost_R)
            self.cost_all_list.append(cost)

            if self.cost_best > cost:
                self.x_trj_best = x_trj
                self.u_trj_best = u_trj
                self.cost_best = cost
                self.idx_best = self.current_iter

            print('Iter {:02d}, '.format(self.current_iter) +
                  'cost: {:0.4f}, '.format(cost) +
                  'time: {:0.2f}.'.format(time.time() - start_time))

            if (self.current_iter >= max_iterations
                    or cost_Qu_final < cost_Qu_f_threshold):
                break

            x_trj, u_trj = self.local_descent(x_trj)
            self.current_iter += 1
