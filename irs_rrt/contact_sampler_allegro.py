import copy

import numpy as np
from irs_rrt.contact_sampler import ContactSampler
from qsim_cpp import QuasistaticSimulatorCpp
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from tests.context import examples
from examples.allegro_hand.allegro_hand_setup import *


class AllegroHandContactSampler(ContactSampler):
    def __init__(self, q_sim: QuasistaticSimulatorCpp):
        super().__init__(q_sim)

        plant = q_sim.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_sim.get_q_a_indices_into_q()

        self.T = 20

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(16)
        self.qdot_torsion[[0, 9, 12, 4]] = 1.0

        self.qdot_anti_torsion = np.zeros(16)
        self.qdot_anti_torsion[0] = -1.0
        self.qdot_anti_torsion[4] = 1.0

        self.qdot_enveloping_flexion = np.zeros(16)
        self.qdot_enveloping_flexion[
            [1, 2, 3, 8, 10, 11, 13, 14, 15, 5, 6, 7]
        ] = 1.0

        self.qdot_pinch_flexion = np.zeros(16)
        self.qdot_pinch_flexion[[3, 11, 15, 7]] = 1.0

        self.q_a0 = np.zeros(16)
        self.q_a0[9] = np.pi / 2  # Default configuraiton.

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_lst = []

        sim_params = copy.deepcopy(self.q_sim.get_sim_params())
        sim_params.unactuated_mass_scale = 0.0
        sim_params.gradient_mode = GradientMode.kNone

        for t in range(T):
            ubar = x[self.idx_a_vec]
            x = self.q_sim.calc_dynamics(x, ubar + qdot, sim_params)
            q_lst.append(np.copy(x))

        return x, q_lst

    def sample_contact(self, q_u):
        q0_dict = {self.idx_a: self.q_a0, self.idx_u: q_u}
        x0 = self.q_sim.get_q_vec_from_dict(q0_dict)

        w_torsion = 0.03 * (np.random.rand() - 0.5)
        w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

        w_enveloping_flexion = 0.05 + 0.03 * (np.random.rand() - 0.5)
        w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

        xnext, q_dict_lst = self.simulate_qdot(
            x0,
            w_torsion * self.qdot_torsion
            + w_anti_torsion * self.qdot_anti_torsion
            + w_enveloping_flexion * self.qdot_enveloping_flexion
            + w_pinch_flexion * self.qdot_pinch_flexion,
            self.T,
        )

        return xnext
