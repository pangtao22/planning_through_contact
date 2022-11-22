import unittest
from typing import Dict, Any

import time

import numpy as np

from context import examples, irs_mpc, irs_rrt
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.rrt_base import Node

from examples.planar_pushing.contact_sampler import PlanarPushingContactSampler
from examples.planar_pushing.planar_pushing_setup import *


class TestPlanarPushingContactSampler(unittest.TestCase):
    """
    Test if contact sampler runs correctly for planar pushing. For visual
    inspection, turn on meshcat.
    """

    def setUp(self):

        self.q_dynamics = QuasistaticDynamics(
            h=h, q_model_path=q_model_path, internal_viz=True
        )

        self.contact_sampler = PlanarPushingContactSampler(self.q_dynamics)

    def test_sampler(self):
        """
        Test if sampler is running correctly.
        """

        for i in range(100):
            q_u = np.random.rand(3)
            q_u[0] = 2.0 * (q_u[0] - 0.5)
            q_u[1] = 2.0 * (q_u[1] - 0.5)
            q_u[2] = 2.0 * np.pi * q_u[2]

            sample = self.contact_sampler.sample_contact(q_u)
            self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(sample)
            self.q_dynamics.q_sim_py.draw_current_configuration()

            query_object = self.q_dynamics.q_sim_py.query_object
            sdp = query_object.ComputeSignedDistancePairwiseClosestPoints(0.3)
            for i, sdp_i in enumerate(sdp):
                self.assertTrue(sdp_i.distance < 0.05)


if __name__ == "__main__":
    unittest.main()
