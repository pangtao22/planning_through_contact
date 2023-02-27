import copy

import numpy as np
from irs_rrt.contact_sampler import ContactSampler
from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# from tests.context import examples
from examples.allegro_hand.allegro_hand_setup import *


class AllegroHandContactSampler(ContactSampler):
    def __init__(
        self, q_sim: QuasistaticSimulatorCpp, q_sim_py: QuasistaticSimulator
    ):
        super().__init__(q_sim, q_sim_py)

        plant = q_sim.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_sim.get_q_a_indices_into_q()

        self.T = 12

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(16)
        self.qdot_torsion[[0, 13, 4, 8]] = 1.0

        self.qdot_anti_torsion = np.zeros(16)
        self.qdot_anti_torsion[0] = -1.0
        self.qdot_anti_torsion[8] = 1.0

        self.qdot_enveloping_flexion = np.zeros(16)
        self.qdot_enveloping_flexion[
            [1, 2, 3, 12, 14, 15, 5, 6, 7, 9, 10, 11]
        ] = 1.0

        self.qdot_pinch_flexion = np.zeros(16)
        self.qdot_pinch_flexion[[3, 7, 11, 15]] = 1.0

        self.q_a0 = np.zeros(16)
        self.q_a0[13] = np.pi / 2  # Default configuration.

        # simulation parameters
        self.sim_params = copy.deepcopy(self.q_sim.get_sim_params())
        self.sim_params.unactuated_mass_scale = 0.0
        self.sim_params.forward_mode = ForwardDynamicsMode.kQpMp
        self.sim_params.gradient_mode = GradientMode.kNone
        self.sim_params.calc_contact_forces = False
        self.sim_params.h = 0.1

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_lst = []

        for t in range(T):
            ubar = x[self.idx_a_vec]
            x = self.q_sim.calc_dynamics(x, ubar + qdot, self.sim_params)
            q_lst.append(np.copy(x))

        return x, q_lst

    def sample_contact(self, q):
        x0 = np.copy(q)
        x0[self.q_sim.get_q_a_indices_into_q()] = self.q_a0

        w_torsion = 0.03 * (np.random.rand() - 0.5)
        w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

        w_enveloping_flexion = 0.05 + 0.03 * (np.random.rand() - 0.5)
        w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

        q_dot = (
            w_torsion * self.qdot_torsion
            + w_anti_torsion * self.qdot_anti_torsion
            + w_enveloping_flexion * self.qdot_enveloping_flexion
            + w_pinch_flexion * self.qdot_pinch_flexion
        )

        q_dot *= 2

        xnext, q_lst = self.simulate_qdot(
            x0,
            q_dot,
            self.T,
        )

        return xnext
