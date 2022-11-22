import unittest

import numpy as np
from .context import examples, irs_mpc
from examples.allegro_hand.allegro_hand_setup import *
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel


class TestQuasistaticDynamicsParallel(unittest.TestCase):
    def setUp(self):
        self.q_dynamics = QuasistaticDynamics(
            h=h, q_model_path=q_model_path, internal_viz=False
        )
        self.q_dynamics_p = QuasistaticDynamicsParallel(
            q_dynamics=self.q_dynamics, use_zmq_workers=False
        )

        q_sim_py = self.q_dynamics.q_sim_py
        plant = q_sim_py.get_plant()
        idx_a = plant.GetModelInstanceByName(robot_name)
        idx_u = plant.GetModelInstanceByName(object_name)

        # initial conditions.
        q_a0 = np.array(
            [
                0.03501504,
                0.75276565,
                0.74146232,
                0.83261002,
                0.63256269,
                1.02378254,
                0.64089555,
                0.82444782,
                -0.1438725,
                0.74696812,
                0.61908827,
                0.70064279,
                -0.06922541,
                0.78533142,
                0.82942863,
                0.90415436,
            ]
        )
        q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
        q0_dict = {idx_a: q_a0, idx_u: q_u0}

        x0 = self.q_dynamics.get_x_from_q_dict(q0_dict)
        u0 = self.q_dynamics.get_u_from_q_cmd_dict(q0_dict)

        n_batch = 10
        x_batch = np.zeros((n_batch, self.q_dynamics.dim_x))
        u_batch = np.zeros((n_batch, self.q_dynamics.dim_u))

        x_batch[:] = x0
        u_batch[:] = u0
        np.random.seed(900407)
        u_batch += np.random.normal(0, 0.1, u_batch.shape)

        self.x_batch = x_batch
        self.u_batch = u_batch

    def test_forward_dynamics(self):
        """
        Compares batch forwards dynamics computed by
            - calling QuasistaticDynamics.dynamics in a loop in python.
            - calling BatchQuasistaticCpp.dynamics_batch.
        (This test is also done in c++.)
        """
        x_next_batch_py = self.q_dynamics_p.dynamics_batch_serial(
            self.x_batch, self.u_batch
        )
        x_next_batch_cpp = self.q_dynamics_p.dynamics_batch(
            self.x_batch, self.u_batch
        )

        self.assertTrue(
            np.allclose(x_next_batch_py, x_next_batch_cpp, atol=1e-6)
        )

    def test_bundled_B(self):
        """
        Compares the bundled B computed by
            - the C++ batch simulator, and
            - QuasistaticDynamics::dynamics, which is used by ZMQ workers.
        """
        T = len(self.x_batch)
        n_samples = 10
        dim_u = self.u_batch.shape[1]
        du_samples = np.random.normal(0, 0.01, [T, n_samples, dim_u])
        Bt_cpp = self.q_dynamics_p.calc_bundled_B_cpp_debug(
            self.x_batch, self.u_batch, du_samples
        )

        Bt_py = self.q_dynamics_p.calc_bundled_B_serial_debug(
            self.x_batch, self.u_batch, du_samples
        )

        self.assertTrue(np.allclose(Bt_py, Bt_cpp, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
