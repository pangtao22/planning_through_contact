import unittest
from typing import Dict, Any

import numpy as np

from context import examples, irs_mpc, irs_rrt
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.rrt_base import Node

from examples.allegro_hand.contact_sampler import AllegroHandContactSampler
from examples.allegro_hand.allegro_hand_setup import *
    
class TestAllegroContactSampler(unittest.TestCase):
    """
    Test if contact sampler runs correctly for allegro. For visual
    inspection, turn on meshcat.    
    """
    def setUp(self):

        self.q_dynamics = QuasistaticDynamics(
            h=h,
            q_model_path=q_model_path_fixqu,
            internal_viz=True)

        self.contact_sampler = AllegroHandContactSampler(self.q_dynamics, 100)

    def test_sampler(self):
        """
        Test if sampler is running correctly.
        """

        for i in range(self.contact_sampler.n_samples):
            q_u = np.array([1, 0, 0, 0,
                -0.08 + 0.01 * np.random.rand(),
                0.01 * np.random.rand() - 0.005, 
                0.05 + 0.03 * np.random.rand()])

            sample = self.contact_sampler.sample_contact(q_u)
            self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(sample)
            self.q_dynamics.q_sim_py.draw_current_configuration()
    
if __name__ == '__main__':
    unittest.main()
