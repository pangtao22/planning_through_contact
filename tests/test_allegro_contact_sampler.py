import unittest

import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler
from examples.allegro_hand.allegro_hand_setup import *


class TestAllegroContactSampler(unittest.TestCase):
    """
    Test if contact sampler runs correctly for allegro. For visual
    inspection, turn on meshcat.
    """

    def setUp(self):

        self.q_dynamics = QuasistaticDynamics(
            h=h, q_model_path=q_model_path, internal_viz=True
        )

        self.contact_sampler = AllegroHandContactSampler(self.q_dynamics)

    def test_sampler(self):
        """
        Test if sampler is running correctly.
        """

        for i in range(10):
            q_u = np.array(
                [
                    1,
                    0,
                    0,
                    0,
                    -0.08 + 0.01 * np.random.rand(),
                    0.01 * np.random.rand() - 0.005,
                    0.05 + 0.03 * np.random.rand(),
                ]
            )

            x_sample = self.contact_sampler.sample_contact(q_u)
            self.q_dynamics.q_sim_py.update_mbp_positions_from_vector(x_sample)
            q_u_sample = x_sample[self.q_dynamics.get_q_u_indices_into_x()]

            # Make sure that the object does not move.
            self.assertTrue(np.allclose(q_u_sample, q_u))


if __name__ == "__main__":
    unittest.main()
