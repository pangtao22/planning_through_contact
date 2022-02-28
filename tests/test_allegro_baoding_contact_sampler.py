import unittest

import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from planning_through_contact.examples.allegro_hand_baoding.contact_sampler_allegro import AllegroHandContactSampler
from planning_through_contact.examples.allegro_hand_baoding.allegro_hand_setup import *


class TestAllegroContactSampler(unittest.TestCase):
    """
    Test if contact sampler runs correctly for allegro. For visual
    inspection, turn on meshcat.
    """
    def setUp(self):

        self.q_dynamics = QuasistaticDynamics(
            h=h,
            q_model_path=q_model_path,
            internal_viz=True)

        self.contact_sampler = AllegroHandContactSampler(self.q_dynamics)

    def test_sampler(self):
        """
        Test if sampler is running correctly.
        """

        for i in range(10):
            q_u_g = np.array([1, 0, 0, 0,
                -0.05,
                0.022,
                0.045])

            q_u_r = np.array([1, 0, 0, 0,
                -0.03,
                - 0.043,
                0.045])

            # q_u_r = np.array([1, 0, 0, 0, -0.018, -0.01, 0.045])
            # q_u_g = np.array([1, 0, 0, 0, -0.081, 0.01, 0.045])

            # q_u_r = np.array([1, 0, 0, 0,
            #     -0.04,
            #     0.02,
            #     0.041])
            
            # q_u_g = np.array([1, 0, 0, 0,
            #     -0.04,
            #     - 0.03,
            #     0.041])

            x_sample = self.contact_sampler.sample_contact(np.concatenate((q_u_r, q_u_g)))
            self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(x_sample)
            # q_u_sample = x_sample[self.q_dynamics.get_q_u_indices_into_x()]

            # Make sure that the object does not move.
            # self.assertTrue(np.allclose(q_u_sample, q_u))


if __name__ == '__main__':
    unittest.main()
