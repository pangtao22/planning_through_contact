import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics, GradientMode
from irs_rrt.contact_sampler import ContactSampler
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix, RigidTransform


from iiwa_bimanual_setup import *


class IiwaBimanualContactSampler(ContactSampler):
    def __init__(self, q_dynamics: QuasistaticDynamics):
        super().__init__(q_dynamics)

        self.q_sim_py = q_dynamics.q_sim_py
        self.plant = self.q_sim_py.get_plant()
        self.idx_a_l = self.plant.GetModelInstanceByName(iiwa_l_name)
        self.idx_a_r = self.plant.GetModelInstanceByName(iiwa_r_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)
        self.idx_a_vec = q_dynamics.get_q_a_indices_into_x()
        self.T = 6

        self.mu_r = np.array([0.3, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0])
        self.mu_l = np.array([-0.3, -0.0, 0.0, -0.1, 0.0, 0.1, 0.0])
        self.std = np.array([0.0, 0.05, 0.001, 0.1, 0.001, 0.1, 0.0])

    def sample_qdot(self, arm):
        """
        Sample qdot.
        """
        if arm == "r":
            return np.random.normal(self.mu_r, self.std)
        else:
            return np.random.normal(self.mu_l, self.std)

    def simulate_qdot(self, x0, qdot, T, unactuated_mass_scale=0):
        x = np.copy(x0)
        q_dict_lst = []

        for t in range(T):
            ubar = x[self.idx_a_vec]

            q_dict = self.q_dynamics.get_q_dict_from_x(x)
            self.q_sim.update_mbp_positions(q_dict)
            q_a_cmd_dict = self.q_dynamics.get_q_a_cmd_dict_from_u(ubar + qdot)

            tau_ext_dict = self.q_sim.calc_tau_ext([])

            sim_params = self.q_sim.get_sim_params()
            sim_params.unactuated_mass_scale = unactuated_mass_scale
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

        if unactuated_mass_scale == 0:
            # self.q_sim_py.animate_system_trajectory(
            #    0.1, q_dict_lst)
            # input()
            pass
        return x, q_dict_lst

    def get_corner(self, qu):
        """
        Given a pose of the box qu, return bottommost corner of the box.
        """
        p_B = np.array(
            [
                [-0.308, -0.232, -0.315],
                [-0.308, 0.232, -0.315],
                [0.308, -0.232, -0.315],
                [0.308, 0.232, -0.315],
                [-0.308, -0.232, 0.315],
                [-0.308, 0.232, 0.315],
                [0.308, -0.232, 0.315],
                [0.308, 0.232, 0.315],
            ]
        )

        X_WB = RigidTransform(Quaternion(qu[:4]), qu[4:]).GetAsMatrix4()

        homog_p_B = np.hstack((p_B, np.ones((8, 1))))
        p_W = X_WB.dot(homog_p_B.T).T[:, :3]

        min_ind = np.argmin(p_W[:, 2])
        p_W_min = p_W[min_ind, :]

        meshcat_vis = self.q_dynamics.q_sim_py.viz.vis
        import meshcat
        import meshcat.geometry as g

        meshcat_vis.set_object(
            g.Points(
                g.PointsGeometry(
                    p_W_min[:, None], color=np.array([1.0, 1.0, 1.0])
                ),
                g.PointsMaterial(size=0.05),
            )
        )

        return p_W_min

    def sample_contact(self, q):

        is_success = False
        while not is_success:
            try:
                q_u = q[self.q_dynamics.get_q_u_indices_into_x()]
                q_a = q[self.q_dynamics.get_q_a_indices_into_x()]

                q_a_r = q_a[:7]
                q_a_l = q_a[7:]

                q_dict = self.q_dynamics.get_q_dict_from_x(q)
                self.q_sim_py.update_mbp_positions(q_dict)

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

                z_a_r_pos = ee_a_r_pos.translation()[2]
                z_a_l_pos = ee_a_l_pos.translation()[2]

                qdot = np.zeros(14)
                q0 = np.copy(q)

                """
                p_W_min = self.get_corner(q_u)
                if (p_W_min[1] > q_u[5]):
                    cointoss = 0.0
                else:
                    cointoss = 1.0
                """

                """
                qdot[0] = -0.1
                xnext_r, q_dict_lst = self.simulate_qdot(
                    q0, qdot, self.T, 10)
                q_next_r = xnext_r[self.q_dynamics.get_q_u_indices_into_x()]
                diff_next_r = np.linalg.norm(q_next_r - q_u)
                qdot[0] = 0.0

                qdot[7] = 0.1
                xnext_l, q_dict_lst = self.simulate_qdot(
                    q0, qdot, self.T, 10)    
                q_next_l = xnext_l[self.q_dynamics.get_q_u_indices_into_x()]
                diff_next_l = np.linalg.norm(q_next_l - q_u)
                qdot = np.zeros(14)

                if (diff_next_r < diff_next_l):
                    qdot[:7] = self.sample_qdot('r')
                    q0[self.q_dynamics.get_q_a_indices_into_x()[:7]] = np.array(
                        [-0.7, 1.8, 0.0, 0.0, 0, 0, 0])

                else:
                    qdot[7:] = self.sample_qdot('l')
                    q0[self.q_dynamics.get_q_a_indices_into_x()[7:]] = np.array(
                        [0.7, 1.8, 0., 0., 0, 0, 0])

                xnext, qdict_lst = self.simulate_qdot(q0, qdot, 5)
                """

                cointoss = np.random.randint(2)

                if cointoss:
                    qdot[:7] = self.sample_qdot("r")
                    q0[self.q_dynamics.get_q_a_indices_into_x()[:7]] = np.array(
                        [-0.7, 1.8, np.pi / 2, -0.5, 0, 0, 0]
                    )
                else:
                    qdot[:7] = -self.sample_qdot("r")
                    q0[self.q_dynamics.get_q_a_indices_into_x()[:7]] = np.array(
                        [-0.3, 1.8, np.pi / 2, -2.5, 0, 0.0, 0]
                    )

                cointoss = np.random.randint(2)

                if cointoss:
                    qdot[7:] = self.sample_qdot("l")
                    q0[self.q_dynamics.get_q_a_indices_into_x()[7:]] = np.array(
                        [0.7, 1.8, -np.pi / 2, -0.5, 0, 0, 0]
                    )
                else:
                    qdot[7:] = -self.sample_qdot("l")
                    q0[self.q_dynamics.get_q_a_indices_into_x()[7:]] = np.array(
                        [0.3, 1.8, -np.pi / 2, -2.5, 0, 0, 0]
                    )

                xnext, q_dict_lst = self.simulate_qdot(q0, qdot, 2)

                print(q_u)
                if q_u[6] >= 0.23:
                    xnext = np.copy(q)

                is_success = True
            except Exception as e:
                print(e)
                print("contact sampling failure.")
                pass

        return xnext
