import unittest
from typing import Dict, Any

import numpy as np

from context import examples, irs_mpc, irs_rrt
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from irs_rrt.rrt_base import Node
from irs_rrt.reachable_set import ReachableSet3D
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from examples.allegro_hand.allegro_hand_setup import *


class TestReachableSet3D(unittest.TestCase):
    """
    The tests compare the numpy storage used for batch computations against
    information in the invidiual nodes of network x to make sure they are
    consistent.
    """

    def setUp(self):
        self.params = IrsRrtParams(q_model_path, None)
        self.q_dynamics = QuasistaticDynamics(
            h=h, q_model_path=q_model_path, internal_viz=False
        )
        self.q_dynamics_p = QuasistaticDynamicsParallel(self.q_dynamics)

        self.params.n_samples = 100
        self.params.std_u = 0.01

        self.reachable_set = ReachableSet3D(
            self.q_dynamics, self.params, self.q_dynamics_p
        )

        self.q_a0 = np.array(
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
        self.q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
        plant = self.q_dynamics.q_sim_py.get_plant()
        idx_a = plant.GetModelInstanceByName(robot_name)
        idx_u = plant.GetModelInstanceByName(object_name)
        self.q0_dict = {idx_a: self.q_a0, idx_u: self.q_u0}

        self.x0 = self.q_dynamics.get_x_from_q_dict(self.q0_dict)
        self.ubar = self.q_dynamics.get_u_from_q_cmd_dict(self.q0_dict)

    def test_run(self):
        """
        Test if the values can be computed and the shapes are as expected.
        """
        Bhat, chat = self.reachable_set.calc_bundled_Bc(self.x0, self.ubar)
        self.assertEqual(
            Bhat.shape, (self.q_dynamics.dim_x, self.q_dynamics.dim_u)
        )
        self.assertEqual(chat.shape, (self.q_dynamics.dim_x,))


if __name__ == "__main__":
    unittest.main()
