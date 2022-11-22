import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics, GradientMode
from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix

from allegro_hand_setup import *


class AllegroHandPlateContactSampler(ContactSampler):
    def __init__(self, q_dynamics: QuasistaticDynamics):
        super().__init__(q_dynamics)

        q_sim_py = q_dynamics.q_sim_py
        plant = q_sim_py.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_dynamics.get_q_a_indices_into_x()

        self.T = 20

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

        self.q_a0 = np.zeros(19)
        self.q_a0[0] = 0.0
        self.q_a0[1] = 0.5
        self.q_a0[2] = 0.5
        self.q_a0[8] = np.pi / 2  # Default configuraiton.

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
            sim_params.unactuated_mass_scale = 0
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

        is_success = False
        while not is_success:
            try:
                q_a0 = np.zeros(19)

                rpy = RollPitchYaw(RotationMatrix(Quaternion(q_u[0:4])))
                roll = rpy.vector()[0]

                q_a0[0] = q_u[4] + 0.05 * (np.random.rand() - 0.5)
                q_a0[1] = q_u[5] + 0.0
                q_a0[2] = q_u[6] + 0.06 + 0.1 * np.sin(roll)
                q_a0[8] = np.pi / 2

                q0_dict = {self.idx_a: q_a0, self.idx_u: q_u}
                x0 = self.q_dynamics.get_x_from_q_dict(q0_dict)
                self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(x0)
                self.q_dynamics.q_sim_py.draw_current_configuration()

                w_torsion = 0.03 * (np.random.rand() - 0.5)
                w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

                w_enveloping_flexion = 0.04 + 0.02 * (np.random.rand() - 0.5)
                w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

                qdot = (
                    w_torsion * self.qdot_torsion
                    + w_anti_torsion * self.qdot_anti_torsion
                    + w_enveloping_flexion * self.qdot_enveloping_flexion
                    + w_pinch_flexion * self.qdot_pinch_flexion
                )

                if np.sin(roll) < 0.2:
                    qdot[8] = 0.01 + 0.01 * (np.random.rand() - 0.5)
                    qdot[9] = 0.01 + 0.01 * (np.random.rand() - 0.5)
                    qdot[10] = 0.01 + 0.01 * (np.random.rand() - 0.5)

                xnext, q_dict_lst = self.simulate_qdot(x0, qdot, self.T)
                self.q_dynamics.q_sim_py.animate_system_trajectory(
                    self.q_dynamics.h, q_dict_lst
                )

                is_success = True
            except Exception as e:
                print(e)
                print("contact sampling failure.")
                pass

        return xnext
