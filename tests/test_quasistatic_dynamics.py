import unittest
from typing import Dict, Any

import numpy as np

from .context import examples, irs_mpc
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from examples.allegro_hand.allegro_hand_setup import *


class TestQuasistaticDynamics(unittest.TestCase):
    """
    The tests compare various conversions between state dictionaries and
    state vectors are consistent between the python and cpp implementations.
    """

    def setUp(self):
        self.q_dynamics = QuasistaticDynamics(
            h=h, q_model_path=q_model_path, internal_viz=False
        )
        self.q_sim_cpp = self.q_dynamics.q_sim

        n2i_map = self.q_sim_cpp.get_model_instance_name_to_index_map()
        self.idx_a = n2i_map[robot_name]
        self.idx_u = n2i_map[object_name]

    def is_dict_of_arrays_equal(
        self, d1: Dict[Any, np.ndarray], d2: Dict[Any, np.ndarray]
    ):
        self.assertTrue(d1.keys() == d2.keys())
        for key in d1.keys():
            self.assertTrue(np.allclose(d1[key], d2[key]))

    def test_x_and_u_vec_dict_conversion(self):
        plant = self.q_sim_cpp.get_plant()
        n_q_a = plant.num_positions(self.idx_a)
        n_q_u = plant.num_positions(self.idx_u)
        n_v_a = plant.num_velocities(self.idx_a)
        n_v_u = plant.num_velocities(self.idx_u)

        q_a = np.arange(n_q_a)
        q_u = np.arange(n_q_u) * 100
        q_dict = {self.idx_a: q_a, self.idx_u: q_u}
        q = np.arange(n_q_a + n_q_u)

        # q_a_cmd, dict --> vec
        self.assertTrue(
            np.allclose(
                self.q_sim_cpp.get_q_a_cmd_vec_from_dict(q_dict),
                self.q_dynamics.get_u_from_q_cmd_dict(q_dict),
            )
        )

        # q_a_cmd, vec --> dict
        self.is_dict_of_arrays_equal(
            self.q_sim_cpp.get_q_a_cmd_dict_from_vec(q_a),
            self.q_dynamics.get_q_a_cmd_dict_from_u(q_a),
        )

        # q, dict --> vec
        self.assertTrue(
            np.allclose(
                self.q_sim_cpp.get_q_vec_from_dict(q_dict),
                self.q_dynamics.get_x_from_q_dict(q_dict),
            )
        )

        # q, vec --> dict
        self.is_dict_of_arrays_equal(
            self.q_sim_cpp.get_q_dict_from_vec(q),
            self.q_dynamics.get_q_dict_from_x(q),
        )

        # v, vec --> dict
        # TODO (pang): self.q_sim_cpp.get_v_dict_from_vec() exists, but its
        #  equivalent in QuasistaticSimulator has not been abstracted into a
        #  function yet.


if __name__ == "__main__":
    unittest.main()
