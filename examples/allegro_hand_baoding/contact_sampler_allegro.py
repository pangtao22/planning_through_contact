import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics, GradientMode
from irs_rrt.contact_sampler import ContactSampler

from planning_through_contact.examples.allegro_hand_baoding.allegro_hand_setup import *


class AllegroHandContactSampler(ContactSampler):
    def __init__(self, q_dynamics: QuasistaticDynamics):
        super().__init__(q_dynamics)

        q_sim_py = q_dynamics.q_sim_py
        plant = q_sim_py.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u_r = plant.GetModelInstanceByName(object_name_1)
        self.idx_u_g = plant.GetModelInstanceByName(object_name_2)
        self.idx_a_vec = q_dynamics.get_q_a_indices_into_x()

        self.T = 20

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(16)
        self.qdot_torsion[[0, 5, 8, 12]] = 1.0

        self.qdot_anti_torsion = np.zeros(16)
        self.qdot_anti_torsion[0] = -1.0
        self.qdot_anti_torsion[12] = 1.0

        self.qdot_enveloping_flexion = np.zeros(16)
        self.qdot_enveloping_flexion[
            [1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15]] = 1.0

        self.qdot_pinch_flexion = np.zeros(16)
        self.qdot_pinch_flexion[[3, 7, 11, 15]] = 1.0

        self.q_a0 = np.zeros(16)
        self.q_a0[5] = np.pi / 2  # Default configuraiton.

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_dict_lst = []
        for t in range(T):
            ubar = x[self.idx_a_vec]
            q_dict = self.q_dynamics.get_q_dict_from_x(x)
            self.q_sim.update_mbp_positions(q_dict)
            q_a_cmd_dict = self.q_dynamics.get_q_a_cmd_dict_from_u(ubar + qdot)
            tau_ext_dict = self.q_sim.calc_tau_ext([])

            sp = self.q_sim.get_sim_params()

            self.q_sim.step(
                q_a_cmd_dict=q_a_cmd_dict,
                tau_ext_dict=tau_ext_dict,
                h=self.q_dynamics.h,
                contact_detection_tolerance=sp.contact_detection_tolerance,
                gradient_mode=GradientMode.kNone,
                unactuated_mass_scale=0)

            q_next_dict = self.q_sim.get_mbp_positions()
            x = self.q_dynamics.get_x_from_q_dict(q_next_dict)

            q_dict = self.q_dynamics.get_q_dict_from_x(x)
            q_dict_lst.append(q_dict)
        return x, q_dict_lst

    def sample_contact(self, q_u):
        q_u_r = q_u[:7]
        q_u_g = q_u[7:]
        q0_dict = {self.idx_a: self.q_a0, self.idx_u_r: q_u_r, self.idx_u_g: q_u_g}
        x0 = self.q_dynamics.get_x_from_q_dict(q0_dict)

        w_torsion = 0.03 * (np.random.rand() - 0.5)
        w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

        w_enveloping_flexion = 0.05 + 0.03 * (np.random.rand() - 0.5)
        w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

        xnext, q_dict_lst = self.simulate_qdot(
            x0,
            w_torsion * self.qdot_torsion +
            w_anti_torsion * self.qdot_anti_torsion +
            w_enveloping_flexion * self.qdot_enveloping_flexion +
            w_pinch_flexion * self.qdot_pinch_flexion, self.T)

        # self.q_dynamics.q_sim_py.animate_system_trajectory(self.q_dynamics.h,
        #     q_dict_lst)

        return xnext
