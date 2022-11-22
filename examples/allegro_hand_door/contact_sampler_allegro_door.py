import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics, GradientMode
from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix

from allegro_hand_setup import *


class AllegroHandPlateContactSampler(ContactSampler):
    def __init__(self, q_dynamics: QuasistaticDynamics):
        super().__init__(q_dynamics)

        q_sim_py = q_dynamics.q_sim_py
        self.plant = q_sim_py.get_plant()
        self.idx_a = self.plant.GetModelInstanceByName(robot_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_dynamics.get_q_a_indices_into_x()
        self.knob_body = self.plant.GetBodyByName("sphere")

        self.T = 25

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(19)
        self.qdot_torsion[[3, 8, 11, 15]] = 1.0

        self.qdot_anti_torsion = np.zeros(19)
        self.qdot_anti_torsion[3] = -1.0
        self.qdot_anti_torsion[15] = 1.0

        self.qdot_enveloping_flexion = np.zeros(19)
        self.qdot_enveloping_flexion[
            [4, 5, 6, 6, 9, 10, 12, 13, 14, 16, 17, 18]
        ] = 1.0

        self.qdot_pinch_flexion = np.zeros(19)
        self.qdot_pinch_flexion[[6, 10, 14, 18]] = 1.0

        self.qdot_thumb = np.zeros(19)
        self.qdot_thumb[8] = -1

        self.q_a0_thumb_forward = np.zeros(19)
        self.q_a0_thumb_forward[0] = 0.0
        self.q_a0_thumb_forward[1] = 0.5
        self.q_a0_thumb_forward[2] = 0.5
        self.q_a0_thumb_forward[8] = np.pi / 2  # Default configuration.

        self.q_a0_thumb_upward = np.array(self.q_a0_thumb_forward)
        self.q_a0_thumb_upward[8] = 0

    def get_qa0(self):
        return np.array(
            self.q_a0_thumb_forward
            if np.random.rand() > 0.5
            else self.q_a0_thumb_upward
        )

    def get_knob_center_world_frame(self, q_u: np.ndarray):
        q_dict = {self.idx_a: self.q_a0_thumb_forward, self.idx_u: q_u}
        # K: knob frame.
        self.q_dynamics.q_sim_py.update_mbp_positions(q_dict)
        X_WK = self.plant.EvalBodyPoseInWorld(
            self.q_dynamics.q_sim_py.context_plant, self.knob_body
        )

        return X_WK.translation()

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_dict_lst = []
        for t in range(T):
            ubar = x[self.idx_a_vec]
            q_dict = self.q_dynamics.get_q_dict_from_x(x)
            self.q_sim.update_mbp_positions(q_dict)
            q_a_cmd_dict = self.q_dynamics.get_q_a_cmd_dict_from_u(ubar + qdot)
            tau_ext_dict = self.q_sim.calc_tau_ext([])

            sim_params = self.q_sim.get_sim_params()
            sim_params.unactuated_mass_scale = 1e-4
            sim_params.gradient_mode = GradientMode.kNone

            self.q_sim.step(
                q_a_cmd_dict=q_a_cmd_dict,
                tau_ext_dict=tau_ext_dict,
                sim_params=sim_params,
            )

            q_next_dict = self.q_sim.get_mbp_positions()
            x = self.q_dynamics.get_x_from_q_dict(q_next_dict)

            q_dict = self.q_dynamics.get_q_dict_from_x(x)
            q_dict_lst.append(q_dict)
        return x, q_dict_lst

    def sample_contact(self, q_u):
        while True:
            try:
                p_WK = self.get_knob_center_world_frame(q_u)
                q_a0 = np.array(self.q_a0_thumb_forward)

                q_a0[0] = p_WK[0] - 0.075  # x
                q_a0[1] = p_WK[1] - 0.06 + np.random.rand() * 0.04
                q_a0[2] = p_WK[2] - 0.06 + np.random.rand() * 0.01

                q0_dict = {self.idx_a: q_a0, self.idx_u: q_u}
                x0 = self.q_dynamics.get_x_from_q_dict(q0_dict)
                self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(x0)

                w_torsion = 0.03 * (np.random.rand() - 0.5)
                w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

                w_enveloping_flexion = 0.04 + 0.02 * (np.random.rand() - 0.5)
                w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

                w_thumb = 0.02 * np.random.rand()

                qdot = (
                    w_torsion * self.qdot_torsion
                    + w_anti_torsion * self.qdot_anti_torsion
                    + w_enveloping_flexion * self.qdot_enveloping_flexion
                    + w_pinch_flexion * self.qdot_pinch_flexion
                    + w_thumb * self.qdot_thumb
                )

                xnext, q_dict_lst = self.simulate_qdot(x0, qdot, self.T)
                self.q_dynamics.q_sim_py.update_mbp_positions(q_dict_lst[-1])
                self.q_dynamics.q_sim_py.draw_current_configuration()

                break

            except RuntimeError as e:
                print(e, "contact sampling failure.")

        return xnext
