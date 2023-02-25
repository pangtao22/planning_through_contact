import copy, time
from typing import Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    ModelInstanceIndex,
    GurobiSolver,
    OsqpSolver,
    PiecewisePolynomial,
)

from qsim.parser import QuasistaticParser
from qsim_cpp import (
    QuasistaticSimulatorCpp,
    ForwardDynamicsMode,
    GradientMode,
    QuasistaticSimParameters,
)

from .irs_mpc_params import (
    IrsMpcQuasistaticParameters,
    SmoothingMode,
    kSmoothingMode2ForwardDynamicsModeMap,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
    k0RandomizedSmoothingModes,
)
from .quasistatic_visualizer import QuasistaticVisualizer
from .mpc import solve_mpc


class IrsMpcQuasistatic:
    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp,
        parser: QuasistaticParser,
        params: IrsMpcQuasistaticParameters,
    ):
        self.irs_mpc_params = params
        self.parser = parser
        self.q_sim = q_sim
        self.q_sim_batch = parser.make_batch_simulator()

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
        # rollout
        # Not used if self.irs_mpc_params.rollout_forward_dynamics_mode is None.
        self.sim_params_rollout = copy.deepcopy(self.sim_params)
        if self.irs_mpc_params.rollout_forward_dynamics_mode:
            self.sim_params_rollout.forward_mode = (
                self.irs_mpc_params.rollout_forward_dynamics_mode
            )
        self.sim_params_rollout.gradient_mode = GradientMode.kNone

    def get_q_sim_params(self, p_mpc: IrsMpcQuasistaticParameters):
        p = copy.deepcopy(self.parser.q_sim_params)

        p.h = p_mpc.h
        p.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            p_mpc.smoothing_mode
        ]

        if p_mpc.use_A:
            p.gradient_mode = GradientMode.kAB
        else:
            p.gradient_mode = GradientMode.kBOnly

        return p

    def initialize_problem(
        self, x0: np.ndarray, x_trj_d: np.ndarray, u_trj_0: np.ndarray
    ):
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
        self.x_trj_0 = self.rollout(
            x0, u_trj_0, self.sim_params_rollout.forward_mode
        )
        self.u_trj_0 = u_trj_0  # T x m
        self.A_trj, self.B_trj, self.c_trj = self.calc_bundled_ABc_trj(
            self.x_trj_0[: self.T], self.u_trj_0
        )

    def get_Q_mat_from_Q_dict(
        self, Q_dict: Dict[ModelInstanceIndex, np.ndarray]
    ):
        Q = np.eye(self.dim_x)
        for model, idx in self.q_sim.get_position_indices().items():
            Q[idx, idx] = Q_dict[model]
        return Q

    def get_R_mat_from_R_dict(
        self, R_dict: Dict[ModelInstanceIndex, np.ndarray]
    ):
        R = np.eye(self.dim_u)
        i_start = 0
        for model in self.q_sim.get_actuated_models():
            n_v_i = self.plant.num_velocities(model)
            R[i_start : i_start + n_v_i, i_start : i_start + n_v_i] = np.diag(
                R_dict[model]
            )
            i_start += n_v_i
        return R

    @staticmethod
    def calc_Q_cost(
        models_list: List[ModelInstanceIndex],
        x_dict: Dict[ModelInstanceIndex, np.ndarray],
        xd_dict: Dict[ModelInstanceIndex, np.ndarray],
        Q_dict: Dict[ModelInstanceIndex, np.ndarray],
    ):
        cost = 0.0
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
            x_dict=x_dict,
            xd_dict=xd_dict,
            Q_dict=self.Qd_dict,
        )
        cost_Qa_final = self.calc_Q_cost(
            models_list=models_a,
            x_dict=x_dict,
            xd_dict=xd_dict,
            Q_dict=self.Qd_dict,
        )

        # Q and R costs.
        cost_Qu = 0.0
        cost_Qa = 0.0
        cost_R = 0.0
        for t in range(T):
            x_dict = self.q_sim.get_q_dict_from_vec(x_trj[t])
            xd_dict = self.q_sim.get_q_dict_from_vec(self.x_trj_d[t])
            # Q cost.
            cost_Qu += self.calc_Q_cost(
                models_list=models_u,
                x_dict=x_dict,
                xd_dict=xd_dict,
                Q_dict=self.Q_dict,
            )
            cost_Qa += self.calc_Q_cost(
                models_list=models_a,
                x_dict=x_dict,
                xd_dict=xd_dict,
                Q_dict=self.Q_dict,
            )

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
        plt.plot(self.cost_all_list, label="all")
        plt.plot(self.cost_Qa_list, label="Qa")
        plt.plot(self.cost_Qu_list, label="Qu")
        plt.plot(self.cost_Qa_final_list, label="Qa_f")
        plt.plot(self.cost_Qu_final_list, label="Qu_f")
        plt.plot(self.cost_R_list, label="R")

        plt.title("Trajectory cost")
        plt.xlabel("Iterations")
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
            "all": self.cost_all_list[i_best],
        }
        result = {
            "cost": cost,
            "x_trj": np.array(self.x_trj_best),
            "u_trj": np.array(self.u_trj_best),
        }
        return result

    def calc_B_zero_order(
        self,
        x_nominal: np.ndarray,
        u_nominal: np.ndarray,
        n_samples: int,
        std_u: Union[np.ndarray, float],
        sim_p: QuasistaticSimParameters,
    ):
        """
        Computes B:=df/du using least-square fit, and A:=df/dx using the
            exact gradient at x_nominal and u_nominal.
        :param std_u: standard deviation of the normal distribution when
            sampling u.
        """
        n_x = self.dim_x
        n_u = self.dim_u
        x_next_nominal = self.q_sim.calc_dynamics(x_nominal, u_nominal, sim_p)
        Bhat = np.zeros((n_x, n_u))

        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])
        results = self.q_sim_batch.calc_dynamics_parallel(
            np.tile(x_nominal, (n_samples, 1)), u_nominal + du, sim_p
        )

        x_next, _, _, is_valid = results
        dx_next = x_next - x_next_nominal

        x_next_smooth = np.mean(x_next[is_valid], axis=0)
        Bhat = np.linalg.lstsq(du[is_valid], dx_next[is_valid], rcond=None)[
            0
        ].transpose()
        return Bhat, x_next_smooth

    def calc_AB_zero_order(
        self,
        x_nominal: np.ndarray,
        u_nominal: np.ndarray,
        n_samples: int,
        std_u: Union[np.ndarray, float],
        sim_p: QuasistaticSimParameters,
    ):
        """
        Computes B:=df/du using least-square fit, and A:=df/dx using the
            exact gradient at x_nominal and u_nominal.
        :param std_u: standard deviation of the normal distribution when
            sampling u.
        """
        n_x = self.dim_x
        n_u = self.dim_u
        x_next_nominal = self.q_sim.calc_dynamics(x_nominal, u_nominal, sim_p)

        dx = np.random.normal(0, 0.001, size=[n_samples, self.dim_x])
        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])

        dxdu = np.hstack((dx, du))

        results = self.q_sim_batch.calc_dynamics_parallel(
            x_nominal + dx, u_nominal + du, sim_p
        )

        x_next, _, _, is_valid = results
        dx_next = x_next - x_next_nominal

        x_next_smooth = np.mean(x_next[is_valid], axis=0)

        ABhat = np.linalg.lstsq(dxdu[is_valid], dx_next[is_valid], rcond=None)[
            0
        ].transpose()

        Ahat = ABhat[:, n_x]
        Bhat = ABhat[:, n_x : n_x + n_u]

        return Ahat, Bhat, x_next_smooth

    def calc_bundled_ABc_trj(self, x_trj: np.ndarray, u_trj: np.ndarray):
        """
        Computes bundled linearized dynamics for the given x and u trajectories.
        This function is only used in self.initialize_problem
        :param x_trj: (T, dim_x)
        :param u_trj: (T, dim_u)
        :return: A_trj (T, n_x, n_x), B_trj(T, n_x, n_u), c_trj(T, n_x).
        """
        T = len(u_trj)
        sim_p = copy.deepcopy(self.sim_params)
        sim_p.calc_contact_forces = False
        if self.irs_mpc_params.smoothing_mode in k1RandomizedSmoothingModes:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1
            )
            (
                A_trj,
                B_trj,
                x_next_smooth_trj,
            ) = self.q_sim_batch.calc_bundled_ABc_trj(
                x_trj,
                u_trj,
                std_u,
                sim_p,
                self.irs_mpc_params.n_samples_randomized,
                None,
            )
        elif self.irs_mpc_params.smoothing_mode in kAnalyticSmoothingModes:
            sim_p.log_barrier_weight = (
                self.irs_mpc_params.calc_log_barrier_weight(
                    self.irs_mpc_params.log_barrier_weight_initial,
                    self.current_iter,
                )
            )
            (
                x_next_smooth_trj,
                A_trj,
                B_trj,
                is_valid,
            ) = self.q_sim_batch.calc_dynamics_parallel(x_trj, u_trj, sim_p)

            if not all(is_valid):
                raise RuntimeError("analytic smoothing failed.")
        elif self.irs_mpc_params.smoothing_mode in k0RandomizedSmoothingModes:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1
            )
            sim_p.gradient_mode = GradientMode.kNone
            x_next_smooth_trj = np.zeros((T + 1, self.dim_x))
            A_trj = np.zeros((T, self.dim_x, self.dim_x))
            B_trj = np.zeros((T, self.dim_x, self.dim_u))

            for t in range(T):
                # TODO: handle irs_mpc_params.use_A = True.
                Bhat, x_next_smooth = self.calc_B_zero_order(
                    x_trj[t],
                    u_trj[t],
                    self.irs_mpc_params.n_samples_randomized,
                    std_u,
                    sim_p,
                )
                x_next_smooth_trj[t] = x_next_smooth

                A_trj[t] = np.eye(self.dim_x)
                B_trj[t] = Bhat

        else:
            raise NotImplementedError

        # Convert lists of 2D arrays to 3D arrays.
        B_trj = np.array(B_trj)

        if self.irs_mpc_params.use_A:
            A_trj = np.array(A_trj)
        else:
            A_trj = np.zeros((T, self.dim_x, self.dim_x))
            A_trj[:] = np.eye(A_trj.shape[1])
            A_trj[:, :, self.indices_u_into_x] = 0.0
            B_trj[:, self.indices_u_into_x, :] = np.eye(self.dim_u)

        # compute ct
        c_trj = np.zeros((T, self.dim_x))
        for t in range(T):
            if self.irs_mpc_params.rollout_forward_dynamics_mode:
                x_next_nominal = self.q_sim.calc_dynamics(
                    x_trj[t], u_trj[t], self.sim_params_rollout
                )
            else:
                x_next_nominal = x_next_smooth_trj[t]

            c_trj[t] = (
                x_next_nominal - A_trj[t].dot(x_trj[t]) - B_trj[t].dot(u_trj[t])
            )

        return A_trj, B_trj, c_trj

    def calc_bundled_ABc(self, x_nominal: np.ndarray, u_nominal: np.ndarray):
        """
        Computes bundled linearized dynamics for the given x and u.
        This function is used in self.local_descent at every iteration of
         the iterative MPC.
        :param x_nominal: (dim_x,)
        :param u_nominal: (dim_u,)
        :return: A(n_x, n_x), B(n_x, n_u), c(n_x,), x_next_nominal(n_x,).
        """
        sim_p = copy.deepcopy(self.sim_params)
        sim_p.calc_contact_forces = False
        if self.irs_mpc_params.smoothing_mode in k1RandomizedSmoothingModes:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1
            )
            n_samples = self.irs_mpc_params.n_samples_randomized
            x_batch = np.zeros((n_samples, self.dim_x))
            x_batch[:] = x_nominal
            u_batch = np.random.normal(
                u_nominal, std_u, (n_samples, self.dim_u)
            )
            (
                x_next_batch,
                A_batch,
                B_batch,
                is_valid,
            ) = self.q_sim_batch.calc_dynamics_parallel(x_batch, u_batch, sim_p)

            if self.irs_mpc_params.use_A:
                A_batch = np.array(A_batch)
                A = A_batch[is_valid].mean(axis=0)

            B_batch = np.array(B_batch)
            B = B_batch[is_valid].mean(axis=0)
            x_next_smooth = x_next_batch[is_valid].mean(axis=0)

        elif self.irs_mpc_params.smoothing_mode in kAnalyticSmoothingModes:
            sim_p.log_barrier_weight = (
                self.irs_mpc_params.calc_log_barrier_weight(
                    self.irs_mpc_params.log_barrier_weight_initial,
                    self.current_iter,
                )
            )

            x_next_smooth = self.q_sim.calc_dynamics(
                x_nominal, u_nominal, sim_p
            )
            if self.irs_mpc_params.use_A:
                A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

        elif self.irs_mpc_params.smoothing_mode in k0RandomizedSmoothingModes:
            std_u = self.irs_mpc_params.calc_std_u(
                self.irs_mpc_params.std_u_initial, self.current_iter + 1
            )
            sim_p.gradient_mode = GradientMode.kNone

            if self.irs_mpc_params.use_A:
                A, B, x_next_smooth = self.calc_AB_zero_order(
                    x_nominal,
                    u_nominal,
                    self.irs_mpc_params.n_samples_randomized,
                    std_u,
                    sim_p,
                )

            else:
                B, x_next_smooth = self.calc_B_zero_order(
                    x_nominal,
                    u_nominal,
                    self.irs_mpc_params.n_samples_randomized,
                    std_u,
                    sim_p,
                )

        else:
            raise NotImplementedError

        if not self.irs_mpc_params.use_A:
            A = np.eye(self.dim_x)
            A[:, self.indices_u_into_x] = 0.0
            B[self.indices_u_into_x, :] = np.eye(self.dim_u)

        # c
        if self.irs_mpc_params.rollout_forward_dynamics_mode:
            x_next_nominal = self.q_sim.calc_dynamics(
                x_nominal, u_nominal, self.sim_params_rollout
            )
        else:
            x_next_nominal = x_next_smooth

        c = x_next_nominal - A @ x_nominal - B @ u_nominal

        return A, B, c, x_next_nominal

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
            u_bounds_abs[0] = (
                x_trj[:-1, self.indices_u_into_x] + self.u_bounds_abs[0]
            )
            u_bounds_abs[1] = (
                x_trj[:-1, self.indices_u_into_x] + self.u_bounds_abs[1]
            )
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
                At=self.A_trj[t : self.T],
                Bt=self.B_trj[t : self.T],
                ct=self.c_trj[t : self.T],
                Q=self.Q,
                Qd=self.Qd,
                R=self.R,
                x0=x_trj_new[t],
                x_trj_d=self.x_trj_d[t:],
                solver=self.solver,
                indices_u_into_x=self.indices_u_into_x,
                x_bound_abs=x_bounds_abs[:, t:, :]
                if (self.x_bounds_abs is not None)
                else None,
                u_bound_abs=u_bounds_abs[:, t:, :]
                if (self.u_bounds_abs is not None)
                else None,
                x_bound_rel=x_bounds_rel[:, t:, :]
                if (self.x_bounds_rel is not None)
                else None,
                u_bound_rel=u_bounds_rel[:, t:, :]
                if (self.u_bounds_rel is not None)
                else None,
                xinit=None,
                uinit=None,
            )
            u_trj_new[t] = u_star[0]

            # Rollout and compute bundled A, B and c.
            At, Bt, ct, x_trj_new[t + 1] = self.calc_bundled_ABc(
                x_trj_new[t], u_trj_new[t]
            )

            self.A_trj[t] = At.squeeze()
            self.B_trj[t] = Bt.squeeze()
            self.c_trj[t] = ct.squeeze()

        return x_trj_new, u_trj_new

    def rollout(
        self,
        x0: np.ndarray,
        u_trj: np.ndarray,
        forward_mode: ForwardDynamicsMode,
    ):
        T = u_trj.shape[0]
        assert T == self.T
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0] = x0

        sim_p = copy.deepcopy(self.sim_params)
        sim_p.forward_mode = forward_mode
        sim_p.gradient_mode = GradientMode.kNone

        for t in range(T):
            x_trj[t + 1] = self.q_sim.calc_dynamics(x_trj[t], u_trj[t], sim_p)

        return x_trj

    @staticmethod
    def rollout_smaller_steps(
        x0: np.ndarray,
        u_trj: np.ndarray,
        h_small: float,
        n_steps_per_h: int,
        q_sim: QuasistaticSimulatorCpp,
        sim_params: QuasistaticSimParameters,
    ):
        T = len(u_trj)
        sim_params.h = h_small

        q_trj_small = np.zeros((T * n_steps_per_h + 1, len(x0)))
        q_trj_small[0] = x0
        u_trj_small = IrsMpcQuasistatic.calc_u_trj_small(
            u_trj, h_small, n_steps_per_h
        )
        for t in range(n_steps_per_h * T):
            q_trj_small[t + 1] = q_sim.calc_dynamics(
                q_trj_small[t], u_trj_small[t], sim_params
            )

        return q_trj_small, u_trj_small

    @staticmethod
    def calc_u_trj_small(u_trj: np.ndarray, h_small: float, n_steps_per_h: int):
        T = len(u_trj)
        # Note! PiecewisePolynomial.ZeroOrderHold ignores the last knot point.
        # So we need to append a useless row.
        t_trj = np.arange(T + 1) * h_small * n_steps_per_h
        u_trj_poly = PiecewisePolynomial.ZeroOrderHold(
            t_trj, np.vstack([u_trj, np.zeros(u_trj.shape[1])]).T
        )

        return np.array(
            [
                u_trj_poly.value(h_small * (t + 0.01)).squeeze()
                for t in range(n_steps_per_h * T)
            ]
        )

    def iterate(self, max_iterations: int, cost_Qu_f_threshold: float = 0):
        """
        Terminates after the trajectory cost is less than cost_threshold or
         max_iterations is reached.
        """
        start_time = time.time()
        x_trj = np.array(self.x_trj_0)
        u_trj = np.array(self.u_trj_0)

        while True:
            (
                cost_Qu,
                cost_Qu_final,
                cost_Qa,
                cost_Qa_final,
                cost_R,
            ) = self.calc_cost(x_trj, u_trj)
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

            print(
                f"Iter {self.current_iter:02d}, "
                f"cost: {cost:0.4f}, "
                f"time: {time.time() - start_time:0.2f}."
            )

            if (
                self.current_iter >= max_iterations
                or cost_Qu_final < cost_Qu_f_threshold
            ):
                break

            x_trj, u_trj = self.local_descent(x_trj)
            self.current_iter += 1

    def run_traj_opt_on_rrt_segment(
        self,
        n_steps_per_h: int,
        h_small: float,
        q0: np.ndarray,
        q_final: np.ndarray,
        u_trj: np.ndarray,
        max_iterations: int,
    ):
        """
        T0 = len(u_trj). This function constructs a new trajectory where each
         knot in the original u_trj is expanded into n_steps_per_h knots.
        Each new knot corresponds to a the new, smaller time step h_small,
         which reduces the effect of "hydroplaning" in Anitescu's model.
        """
        indices_q_u_into_x = self.q_sim.get_q_u_indices_into_q()
        q_u_d = q_final[indices_q_u_into_x]
        q_d = np.copy(q0)
        q_d[indices_q_u_into_x] = q_u_d

        T = len(u_trj) * n_steps_per_h
        q_trj_d = np.tile(q_d, (T + 1, 1))

        u_trj_small = IrsMpcQuasistatic.calc_u_trj_small(
            u_trj, h_small, n_steps_per_h
        )

        self.initialize_problem(x0=q0, x_trj_d=q_trj_d, u_trj_0=u_trj_small)
        self.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=0)

        return (
            np.array(self.x_trj_best),
            np.array(self.u_trj_best),
            self.idx_best,
        )
