import numpy as np
import copy
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix

from qsim_cpp import QuasistaticSimulatorCpp
from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.simulator import QuasistaticSimulator

from allegro_hand_setup import *


class AllegroHandDoorContactSampler(ContactSampler):
    def __init__(self, q_sim: QuasistaticSimulatorCpp, q_sim_py: QuasistaticSimulator):
        super().__init__(q_sim=q_sim, q_sim_py=q_sim_py)

        self.q_sim = q_sim
        self.q_sim_py = q_sim_py
        self.q_vis = QuasistaticVisualizer(self.q_sim, self.q_sim_py)

        plant = q_sim_py.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_sim.get_q_a_indices_into_q()
        self.knob_body = self.q_sim_py.plant.GetBodyByName("sphere")

        self.T = 25

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(19)
        self.qdot_torsion[[3, 16, 7, 11]] = 1.0

        self.qdot_anti_torsion = np.zeros(19)
        self.qdot_anti_torsion[3] = -1.0
        self.qdot_anti_torsion[11] = 1.0

        self.qdot_enveloping_flexion = np.zeros(19)
        self.qdot_enveloping_flexion[[4, 5, 6, 15, 17, 18, 8, 9, 10, 12, 13, 14]] = 1.0

        self.qdot_pinch_flexion = np.zeros(19)
        self.qdot_pinch_flexion[[6, 10, 14, 18]] = 1.0

        self.qdot_thumb = np.zeros(19)
        self.qdot_thumb[16] = -1

        self.q_a0_thumb_forward = np.zeros(19)
        self.q_a0_thumb_forward[0] = 0.0
        self.q_a0_thumb_forward[1] = 0.5
        self.q_a0_thumb_forward[2] = 0.5
        self.q_a0_thumb_forward[16] = np.pi / 2  # Default configuration.

        self.q_a0_thumb_upward = np.array(self.q_a0_thumb_forward)
        self.q_a0_thumb_upward[16] = 0

        # simulation parameters
        self.sim_params = copy.deepcopy(self.q_sim.get_sim_params())
        self.sim_params.unactuated_mass_scale = 0.0
        self.sim_params.forward_mode = ForwardDynamicsMode.kSocpMp
        self.sim_params.gradient_mode = GradientMode.kNone
        self.sim_params.calc_contact_forces = False
        self.sim_params.h = h

    def get_qa0(self):
        return np.array(
            self.q_a0_thumb_forward
            if np.random.rand() > 0.5
            else self.q_a0_thumb_upward
        )

    def get_knob_center_world_frame(self, q_u: np.ndarray):
        q_dict = {self.idx_a: self.q_a0_thumb_forward, self.idx_u: q_u}
        # K: knob frame.
        self.q_sim_py.update_mbp_positions(q_dict)
        X_WK = self.q_sim_py.plant.EvalBodyPoseInWorld(
            self.q_sim_py.context_plant, self.knob_body
        )

        return X_WK.translation()

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_lst = []

        for t in range(T):
            ubar = x[self.idx_a_vec]
            x = self.q_sim.calc_dynamics(x, ubar + qdot, self.sim_params)
            # self.q_vis.draw_configuration(x)
            q_lst.append(np.copy(x))

        return x, q_lst

    def sample_contact(self, q):
        while True:
            try:
                q_u = q[self.q_sim.get_q_u_indices_into_q()]
                p_WK = self.get_knob_center_world_frame(q_u)
                q_a0 = np.array(self.q_a0_thumb_forward)

                q_a0[0] = p_WK[0] - 0.075  # x
                q_a0[1] = p_WK[1] - 0.06 + np.random.rand() * 0.04
                q_a0[2] = p_WK[2] - 0.06 + np.random.rand() * 0.01

                q0_dict = {self.idx_a: q_a0, self.idx_u: q_u}
                x0 = self.q_sim.get_q_vec_from_dict(q0_dict)

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
                self.q_vis.draw_configuration(xnext)

                break

            except RuntimeError as e:
                print(e, "contact sampling failure.")

        return xnext
