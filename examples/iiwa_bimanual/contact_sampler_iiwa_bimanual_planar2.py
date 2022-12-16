import numpy as np
import copy
from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix, RigidTransform

from iiwa_bimanual_setup import *


class IiwaBimanualPlanarContactSampler(ContactSampler):
    def __init__(self, q_sim, q_sim_py):
        super().__init__(q_sim, q_sim_py)

        self.q_sim = q_sim
        self.q_sim_py = q_sim_py
        self.plant = self.q_sim.get_plant()
        self.idx_a_l = self.plant.GetModelInstanceByName(iiwa_l_name)
        self.idx_a_r = self.plant.GetModelInstanceByName(iiwa_r_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)
        self.idx_q_a = q_sim.get_q_a_indices_into_q()
        self.idx_q_u = q_sim.get_q_u_indices_into_q()
        self.T = 6

        self.mu_r = np.array([0.1, -0.1, 0.1])
        self.mu_l = np.array([-0.1, 0.1, -0.1])
        self.std = np.array([0.05, 0.1, 0.1])

        # simulation parameters
        self.sim_params = copy.deepcopy(self.q_sim.get_sim_params())
        self.sim_params.unactuated_mass_scale = 0.0
        self.sim_params.forward_mode = ForwardDynamicsMode.kQpMp
        self.sim_params.gradient_mode = GradientMode.kNone
        self.sim_params.calc_contact_forces = False
        self.sim_params.h = 0.05

    def sample_qdot(self, arm):
        """
        Sample qdot.
        """
        if arm == "r":
            return np.random.normal(self.mu_r, self.std)
        else:
            return np.random.normal(self.mu_l, self.std)

    def simulate_qdot(self, x0, qdot, T):
        x = np.copy(x0)
        q_lst = []

        for t in range(T):
            ubar = x[self.idx_q_a]
            x = self.q_sim.calc_dynamics(x, ubar + qdot, self.sim_params)
            q_lst.append(np.copy(x))

        return x, q_lst

    def sample_contact(self, q):

        is_success = False
        while not is_success:
            try:
                q_u = q[self.idx_q_u]
                q_a = q[self.idx_q_a]

                q_a_r = q_a[:3]
                q_a_l = q_a[3:]

                q_dict = self.q_sim.get_q_dict_from_vec(q)
                self.q_sim.update_mbp_positions(q_dict)

                idx_a_r = self.plant.GetModelInstanceByName(iiwa_r_name)
                ee_a_r = self.plant.GetBodyByName("iiwa_link_7", idx_a_r)
                ee_a_r_pos = self.plant.EvalBodyPoseInWorld(
                    self.q_sim_py.context_plant, ee_a_r
                )

                idx_a_l = self.plant.GetModelInstanceByName(iiwa_l_name)
                ee_a_l = self.plant.GetBodyByName("iiwa_link_7", idx_a_l)
                ee_a_l_pos = self.plant.EvalBodyPoseInWorld(
                    self.q_sim_py.context_plant, ee_a_l
                )

                qdot = np.zeros(6)
                q0 = np.copy(q)

                cointoss = np.random.randint(2)

                if cointoss:
                    qdot[:3] = self.sample_qdot("l")
                    q0[self.idx_q_a[:3]] = np.array(
                        [np.pi / 4, np.pi / 4, 0.0]
                    )
                else:
                    qdot[:3] = -self.sample_qdot("l")
                    q0[self.idx_q_a[:3]] = np.array(
                        [np.pi / 4, np.pi - 0.3, 0.0]
                    )

                cointoss = np.random.randint(2)

                if cointoss:
                    qdot[3:] = self.sample_qdot("r")
                    q0[self.idx_q_a()[3:]] = np.array(
                        [-np.pi / 4, -np.pi / 4, 0.0]
                    )
                else:
                    qdot[3:] = -self.sample_qdot("r")
                    q0[self.idx_q_a()[3:]] = np.array(
                        [-np.pi / 4, -np.pi + 0.3, 0.0]
                    )

                xnext, q_dict_lst = self.simulate_qdot(q0, qdot, 3)
                is_success = True
            except Exception as e:
                print(e)
                print("contact sampling failure.")
                pass

        return xnext
