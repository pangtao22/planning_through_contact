import unittest
from typing import Dict, Any

import numpy as np

from .context import examples, irs_mpc, irs_rrt
from irs_rrt.rrt_base import Node

from examples.euclidean_tree.euclidean_tree import (
    EuclideanRrtParams,
    EuclideanRrt,
)


class TestRRTBase(unittest.TestCase):
    """
    The tests compare the numpy storage used for batch computations against
    information in the invidiual nodes of network x to make sure they are
    consistent.
    """

    def setUp(self):
        np.random.seed(940923)
        root_node = Node(np.zeros(2))
        goal = 10.0 * np.ones(2)

        self.params = EuclideanRrtParams()
        self.params.root_node = root_node
        self.params.goal = goal
        self.params.goal_as_subgoal_prob = 0.1
        self.params.max_size = 200

        self.tree = EuclideanRrt(self.params)
        self.tree.iterate()

    def test_q_matrix(self):
        """
        Test if the q matrix is still valid through iterations of rewiring.
        """
        node_lst = []
        for i in range(self.tree.size):
            node_lst.append(self.tree.graph.nodes[i]["node"].q)
        node_lst = np.array(node_lst)

        self.assertTrue(np.allclose(node_lst, self.tree.get_q_matrix_up_to()))

    def test_value_lst(self):
        """
        Test if the value list is tracked correctly.
        """
        node_lst = []
        for i in range(self.tree.size):
            node_lst.append(self.tree.graph.nodes[i]["node"].value)
        node_lst = np.array(node_lst)

        self.assertTrue(np.allclose(node_lst, self.tree.get_valid_value_lst()))

    def test_node_id(self):
        """
        Test if the nodes all have valid ids.
        """
        node_lst = []
        for i in range(self.tree.size):
            node_lst.append(self.tree.graph.nodes[i]["node"].id)
        node_lst = np.array(node_lst)

        self.assertTrue(np.allclose(node_lst, np.array(range(self.tree.size))))


if __name__ == "__main__":
    unittest.main()
