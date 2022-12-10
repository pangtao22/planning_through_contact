import numpy as np
import copy

from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix

from qsim_cpp import QuasistaticSimulatorCpp
from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.simulator import QuasistaticSimulator

from allegro_hand_setup import *


class AllegroHandPenContactSampler(ContactSampler):
    def __init__(self, 
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator):
        super().__init__(q_sim=q_sim, q_sim_py=q_sim_py)

        self.q_sim = q_sim
        self.q_sim_py = q_sim_py

        plant = q_sim.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_sim.get_q_a_indices_into_q()

        self.T = 20

        # Basis vectors for generating eigengrasps.
        self.qdot_torsion = np.zeros(19)
        self.qdot_torsion[[3, 16, 7, 11]] = 1.0

        self.qdot_anti_torsion = np.zeros(19)
        self.qdot_anti_torsion[3] = -1.0
        self.qdot_anti_torsion[11] = 1.0

        self.qdot_enveloping_flexion = np.zeros(19)
        self.qdot_enveloping_flexion[
            [4, 5, 6, 15, 17, 18, 8, 9, 10, 12, 13, 14]
        ] = 1.0

        self.qdot_pinch_flexion = np.zeros(19)
        self.qdot_pinch_flexion[[6, 10, 14, 18]] = 1.0

        self.q_a0 = np.zeros(19)
        self.q_a0[0] = 0.0
        self.q_a0[1] = 0.5
        self.q_a0[2] = 0.5
        self.q_a0[16] = np.pi / 2  # Default configuraiton.

        # simulation parameters
        self.sim_params = copy.deepcopy(self.q_sim.get_sim_params())
        self.sim_params.unactuated_mass_scale = 0.0
        self.sim_params.forward_mode = ForwardDynamicsMode.kQpMp
        self.sim_params.gradient_mode = GradientMode.kNone
        self.sim_params.calc_contact_forces = False
        self.sim_params.h = h        

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_lst = []

        for t in range(T):
            ubar = x[self.idx_a_vec]
            x = self.q_sim.calc_dynamics(x, ubar + qdot, self.sim_params)
            q_lst.append(np.copy(x))

        return x, q_lst

    def sample_contact(self, q):

        is_success = False
        while not is_success:
            try:
                q_a0 = np.zeros(19)
                q_u = q[self.q_sim.get_q_u_indices_into_q()]

                rpy = RollPitchYaw(
                    RotationMatrix(Quaternion(q_u[0:4])).matrix()
                )
                rpy_vec = rpy.vector()

                q_a0[0] = -q_u[5] - (0.01 * (np.random.rand() - 0.5))
                q_a0[1] = -q_u[4] - (+0.07 + 0.02 * (np.random.rand() - 0.5))

                q_a0[2] = q_u[6] - 0.05  # - 0.05 * np.abs(np.sin(max_rp))
                q_a0[16] = np.pi / 2

                q0_dict = {self.idx_a: q_a0, self.idx_u: q_u}
                x0 = self.q_sim.get_q_vec_from_dict(q0_dict)

                w_torsion = 0.03 * (np.random.rand() - 0.5)
                w_anti_torsion = 0.03 * (np.random.rand() - 0.5)

                w_enveloping_flexion = 0.06 + 0.02 * (np.random.rand() - 0.5)
                w_pinch_flexion = 0.01 + 0.03 * (np.random.rand() - 0.5)

                qdot = (
                    w_torsion * self.qdot_torsion
                    + w_anti_torsion * self.qdot_anti_torsion
                    + w_enveloping_flexion * self.qdot_enveloping_flexion
                    + w_pinch_flexion * self.qdot_pinch_flexion
                )

                print("A")
                xnext, q_dict_lst = self.simulate_qdot(x0, qdot, self.T)
                #self.q_sim_py.update_mbp_positions_from_vector(xnext)
                #self.q_sim_py.draw_current_configuration()
                # self.q_dynamics.q_sim_py.animate_system_trajectory(
                #     self.q_dynamics.h, q_dict_lst)

                # print(q_dict_lst[-1])

                is_success = True
            except Exception as e:
                print(e)
                print("contact sampling failure.")
                pass

        return xnext
