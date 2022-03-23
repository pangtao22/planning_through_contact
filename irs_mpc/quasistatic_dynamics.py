from typing import Dict, Set, Union
import os

import numpy as np
import logging
from pydrake.all import (ModelInstanceIndex, MultibodyPlant,
                         PiecewisePolynomial)
from qsim.parser import (QuasistaticParser, GradientMode, QuasistaticSimulator,
                         QuasistaticSimParameters)
from qsim.simulator import ForwardDynamicsMode

from irs_mpc.irs_mpc_params import BundleMode


class QuasistaticDynamics:
    def __init__(self, h: float, q_model_path: str,
                 internal_viz: bool):
        super().__init__()
        self.q_model_path = q_model_path
        self.parser = QuasistaticParser(q_model_path)
        self.parser.set_sim_params(
            h=h,
            is_quasi_dynamic=True)

        self.h = h
        self.q_sim_py = self.parser.make_simulator_py(internal_vis=internal_viz)
        self.q_sim = self.parser.make_simulator_cpp()
        self.plant = self.q_sim.get_plant()
        self.dim_x = self.plant.num_positions()
        self.dim_u = self.q_sim.num_actuated_dofs()

        self.models_all = self.q_sim.get_all_models()
        self.models_actuated = self.q_sim.get_actuated_models()
        self.models_unactuated = self.q_sim.get_unactuated_models()

        self.position_indices = self.q_sim.get_position_indices()
        self.velocity_indices = self.q_sim.get_velocity_indices()

        # make sure that q_sim_py and q_sim have the same underlying plant.
        self.check_plants(
            plant_a=self.q_sim.get_plant(),
            plant_b=self.q_sim_py.get_plant(),
            models_all_a=self.q_sim.get_all_models(),
            models_all_b=self.q_sim_py.get_all_models(),
            velocity_indices_a=self.q_sim.get_velocity_indices(),
            velocity_indices_b=self.q_sim.get_velocity_indices())

        self.q_sim_params_default = self.q_sim.get_sim_params()

    @staticmethod
    def check_plants(plant_a: MultibodyPlant, plant_b: MultibodyPlant,
                     models_all_a: Set[ModelInstanceIndex],
                     models_all_b: Set[ModelInstanceIndex],
                     velocity_indices_a: Dict[ModelInstanceIndex, np.ndarray],
                     velocity_indices_b: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Make sure that plant_a and plant_b are identical.
        """
        assert models_all_a == models_all_b
        for model in models_all_a:
            name_a = plant_a.GetModelInstanceName(model)
            name_b = plant_b.GetModelInstanceName(model)
            assert name_a == name_b

            idx_a = velocity_indices_a[model]
            idx_b = velocity_indices_b[model]
            assert idx_a == idx_b

    # TODO (pang): consider moving functions that convert between state
    #  dictionaries and state vectors to QuasistaticSimulator, together with
    #  the relevant tests.
    def get_q_a_indices_into_x(self):
        q_a_indices = np.zeros(self.dim_u, dtype=int)
        i_start = 0
        for model in self.models_actuated:
            indices = self.position_indices[model]
            n_model = len(indices)
            q_a_indices[i_start: i_start + n_model] = indices
            i_start += n_model
        return q_a_indices

    def get_q_u_indices_into_x(self):
        q_u_indices = np.zeros(self.dim_x - self.dim_u, dtype=int)
        i_start = 0
        for model in self.models_unactuated:
            indices = self.position_indices[model]
            n_model = len(indices)
            q_u_indices[i_start: i_start + n_model] = indices
            i_start += n_model
        return q_u_indices

    def get_q_a_cmd_dict_from_u(self, u: np.ndarray):
        q_a_cmd_dict = dict()
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            q_a_cmd_dict[model] = u[i_start: i_start + n_v_i]
            i_start += n_v_i

        return q_a_cmd_dict

    def get_q_dict_from_x(self, x: np.ndarray):
        q_dict = {
            model: x[n_q_indices]
            for model, n_q_indices in self.position_indices.items()}

        return q_dict

    def get_x_from_q_dict(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        x = np.zeros(self.dim_x)
        for model, n_q_indices in self.position_indices.items():
            x[n_q_indices] = q_dict[model]

        return x

    def get_u_from_q_cmd_dict(self,
                              q_cmd_dict: Dict[ModelInstanceIndex, np.ndarray]):
        u = np.zeros(self.dim_u)
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            u[i_start: i_start + n_v_i] = q_cmd_dict[model]
            i_start += n_v_i

        return u

    def get_Q_from_Q_dict(self, Q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        Q = np.eye(self.dim_x)
        for model, idx in self.position_indices.items():
            Q[idx, idx] = Q_dict[model]
        return Q

    def get_R_from_R_dict(self,
                          R_dict: Dict[ModelInstanceIndex, np.ndarray]):
        R = np.eye(self.dim_u)
        i_start = 0
        for model in self.models_actuated:
            n_v_i = self.plant.num_velocities(model)
            R[i_start: i_start + n_v_i, i_start: i_start + n_v_i] = \
                np.diag(R_dict[model])
            i_start += n_v_i
        return R

    def publish_trajectory(self, x_traj, h=None):
        q_dict_traj = [self.get_q_dict_from_x(x) for x in x_traj]
        self.q_sim_py.animate_system_trajectory(h=self.h if h is None else h,
                                                q_dict_traj=q_dict_traj)

    def make_sim_params(self, forward_mode: ForwardDynamicsMode,
                        gradient_mode: GradientMode):
        sim_params = QuasistaticSimulator.copy_sim_params(
            self.q_sim_params_default)
        if forward_mode is not None:
            sim_params.forward_mode = forward_mode
        if gradient_mode is not None:
            sim_params.gradient_mode = gradient_mode

        return sim_params

    def dynamics_py(
            self, x: np.ndarray, u: np.ndarray,
            forward_mode: ForwardDynamicsMode = None,
            gradient_mode: GradientMode = GradientMode.kNone):
        """
        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        q_dict = self.get_q_dict_from_x(x)
        self.q_sim_py.update_mbp_positions(q_dict)

        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)
        tau_ext_dict = self.q_sim_py.calc_tau_ext([])

        sim_params = self.make_sim_params(forward_mode, gradient_mode)

        q_next_dict = self.q_sim_py.step(q_a_cmd_dict, tau_ext_dict, sim_params)

        return self.get_x_from_q_dict(q_next_dict)

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                 forward_mode: ForwardDynamicsMode = None,
                 gradient_mode: GradientMode = GradientMode.kNone):
        """
        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        sim_params = self.make_sim_params(forward_mode, gradient_mode)

        q_dict = self.get_q_dict_from_x(x)
        self.q_sim.update_mbp_positions(q_dict)
        q_a_cmd_dict = self.get_q_a_cmd_dict_from_u(u)
        tau_ext_dict = self.q_sim.calc_tau_ext([])

        self.q_sim.step(
            q_a_cmd_dict=q_a_cmd_dict,
            tau_ext_dict=tau_ext_dict,
            sim_params=sim_params)

        q_next_dict = self.q_sim.get_mbp_positions()
        return self.get_x_from_q_dict(q_next_dict)

    def dynamics_rollout(self, x0: np.ndarray, u_trj: np.ndarray):
        """
        Given an initial state and trajectory of inputs, rollout the
        input and return a trajectory of states using the dynamics function.
        : params x0 (dim_x): initial state of the robot.
        : params u_trj (T, dim_u): input trajectory to be applied
        """
        T = u_trj.shape[0]
        x_trj = np.zeros((T + 1, self.dim_x))
        x_trj[0, :] = x0
        for t in range(T):
            x_trj[t + 1, :] = self.dynamics(x_trj[t, :], u_trj[t, :])
        return x_trj

    def dynamics_more_steps(self, x: np.ndarray, u: np.ndarray,
                            n_steps: int):
        """
        Instead of commanding u in one step self.h, this function
        interpolates the control input between x[index_to_u] and u,
        and simulates to self.h using a smaller time step u/n_steps.

        :param x: the position vector of self.q_sim.plant.
        :param u: commanded positions of models in
            self.q_sim.models_actuated, concatenated into one vector.
        """
        q_dict = self.get_q_dict_from_x(x)
        self.q_sim.update_mbp_positions(q_dict)

        u0 = self.get_u_from_q_cmd_dict(q_dict)
        u_traj = PiecewisePolynomial.FirstOrderHold(
            [0, self.h / 2, self.h], np.vstack([u0, u, u]).T)
        h_small = self.h / n_steps

        sp = self.q_sim.get_sim_params()

        for i in range(1, n_steps + 1):
            t = i / n_steps * self.h
            u_t = u_traj.value(t).ravel()
            qa_dict = self.get_q_a_cmd_dict_from_u(u_t)
            tau_ext_dict = self.q_sim.calc_tau_ext([])

            self.q_sim.step(
                qa_dict, tau_ext_dict, h_small,
                self.q_sim_py.sim_params.contact_detection_tolerance,
                requires_grad=False,
                unactuated_mass_scale=sp.unactuated_mass_scale)

        q_next_dict = self.q_sim.get_mbp_positions()
        return self.get_x_from_q_dict(q_next_dict)

    def jacobian_xu(self, x, u):
        AB = np.zeros((self.dim_x, self.dim_x + self.dim_u))
        self.dynamics(x, u, params=GradientMode.kAB)
        AB[:, :self.dim_x] = self.q_sim.get_Dq_nextDq()
        AB[:, self.dim_x:] = self.q_sim.get_Dq_nextDqa_cmd()

        return AB

    def calc_bundled_AB(
            self, x_nominals: np.ndarray, u_nominals: np.ndarray,
            n_samples: int, std_u: Union[np.ndarray, float],
            bundle_mode: BundleMode):
        """
        x_nominals: (n, n_x) array, n states.
        u_nominals: (n, n_u) array, n inputs.
        mode: "first_order", "zero_order_B", "zero_order_AB", or "exact."
        """
        n = x_nominals.shape[0]
        ABhat_list = np.zeros((n, self.dim_x, self.dim_x + self.dim_u))

        if bundle_mode == BundleMode.kFirstRandomized:
            for i in range(n):
                ABhat_list[i] = self.calc_AB_first_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)
        elif bundle_mode == BundleMode.kZeroB:
            for i in range(n):
                ABhat_list[i] = self.calc_B_zero_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)
        elif bundle_mode == BundleMode.kZeroAB:
            for i in range(n):
                ABhat_list[i] = self.calc_AB_zero_order(
                    x_nominals[i], u_nominals[i], n_samples, std_u)
        elif bundle_mode == BundleMode.kFirstExact:
            for i in range(n):
                ABhat_list[i] = self.calc_AB_exact(
                    x_nominals[i], u_nominals[i])
        else:
            raise RuntimeError(f"AB mode {bundle_mode} is not supported.")

        return ABhat_list

    def calc_AB_exact(self, x_nominal: np.ndarray, u_nominal: np.ndarray):
        return self.jacobian_xu(x_nominal, u_nominal)

    def calc_AB_first_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                            n_samples: int, std_u: Union[np.ndarray, float]):
        """
        x_nominal: (n_x,) array, 1 state.
        u_nominal: (n_u,) array, 1 input.
        """
        # np.random.seed(2021)
        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])
        ABhat = np.zeros((self.dim_x, self.dim_x + self.dim_u))
        is_sample_good = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            try:
                self.dynamics(x_nominal, u_nominal + du[i],
                              params=GradientMode.kBOnly)
                ABhat[:, :self.dim_x] += self.q_sim.get_Dq_nextDq()
                ABhat[:, self.dim_x:] += self.q_sim.get_Dq_nextDqa_cmd()
            except RuntimeError as err:
                is_sample_good[i] = False
                logging.warning(err.__str__())

        ABhat /= is_sample_good.sum()
        return ABhat

    def calc_B_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                          n_samples: int, std_u: Union[np.ndarray, float]):
        """
        Computes B:=df/du using least-square fit, and A:=df/dx using the
            exact gradient at x_nominal and u_nominal.
        :param std_u: standard deviation of the normal distribution when
            sampling u.
        """
        n_x = self.dim_x
        n_u = self.dim_u
        x_next_nominal = self.dynamics(x_nominal, u_nominal,
                                       params=GradientMode.kBOnly)
        ABhat = np.zeros((n_x, n_x + n_u))
        ABhat[:, :n_x] = self.q_sim.get_Dq_nextDq()

        du = np.random.normal(0, std_u, size=[n_samples, self.dim_u])
        x_next = np.zeros((n_samples, self.dim_x))

        for i in range(n_samples):
            x_next[i] = self.dynamics(x_nominal, u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        ABhat[:, n_x:] = np.linalg.lstsq(du, dx_next, rcond=None)[0].transpose()

        return ABhat

    def calc_AB_zero_order(self, x_nominal: np.ndarray, u_nominal: np.ndarray,
                           n_samples: int, std_u: Union[np.ndarray, float],
                           std_x: Union[np.ndarray, float] = 1e-3,
                           damp: float = 1e-2):
        """
        Computes both A:=df/dx and B:=df/du using least-square fit.
        :param std_x (n_x,): standard deviation of the normal distribution
            when sampling x.
        :param damp, weight of norm-regularization when solving for A and B.
        """
        n_x = self.dim_x
        n_u = self.dim_u
        dx = np.random.normal(0, std_x, size=[n_samples, n_x])
        du = np.random.normal(0, std_u, size=[n_samples, n_u])

        x_next_nominal = self.dynamics(x_nominal, u_nominal)
        x_next = np.zeros((n_samples, n_x))

        for i in range(n_samples):
            x_next[i] = self.dynamics(x_nominal + dx[i], u_nominal + du[i])

        dx_next = x_next - x_next_nominal
        # A, B as in AX = B, not the linearized dynamics.
        A = np.zeros((n_samples + n_x + n_u, n_x + n_u))
        A[:n_samples, :n_x] = dx
        A[:n_samples, n_x:] = du
        A[n_samples:] = np.eye(n_x + n_u, n_x + n_u) * damp
        B = np.zeros((n_samples + n_x + n_u, n_x))
        B[:n_samples] = dx_next

        ABhat = np.linalg.lstsq(A, B, rcond=None)[0].transpose()

        return ABhat
