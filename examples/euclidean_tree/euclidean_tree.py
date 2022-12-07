import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from irs_rrt.rrt_base import Node, Edge, RrtParams, Rrt


class EuclideanRrtParams(RrtParams):
    def __init__(self):
        super().__init__()
        self.x_lb = np.array([-20, -20])
        self.x_ub = np.array([20, 20])
        self.radius = 0.1  # radius of maximum extension


class EuclideanRrt(Rrt):
    def __init__(self, params: RrtParams):
        super().__init__(params)

    def sample_subgoal(self):
        subgoal = np.random.rand(2)
        subgoal = (
            self.rrt_params.x_lb
            + (self.rrt_params.x_ub - self.rrt_params.x_lb) * subgoal
        )
        return subgoal

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        x_parent = parent_node.q
        x_child = child_node.q
        return np.linalg.norm(x_parent - x_child)

    def extend_towards_q(self, node: Node, q: np.array):
        if np.linalg.norm(q - node.q) < self.rrt_params.radius:
            child_q = q
        else:
            child_q = node.q + self.rrt_params.radius * (
                q - node.q
            ) / np.linalg.norm(q - node.q)
        return Node(child_q)

    def calc_distance_batch(self, q_query: np.array):
        q_batch = self.get_q_matrix_up_to()
        error_batch = q_query[None, :] - q_batch
        return np.linalg.norm(error_batch, axis=1)


def plot_result():
    root_node = Node(np.zeros(2))

    goal = 10.0 * np.ones(2)

    params = EuclideanRrtParams()
    params.root_node = root_node
    params.goal = goal
    params.goal_as_subgoal_prob = 0.01
    params.max_size = 3000

    tree = EuclideanRrt(params)
    tree.iterate()

    node_lst = []
    for i in range(tree.size):
        node_lst.append(tree.graph.nodes[i]["node"].q)
    node_lst = np.array(node_lst)

    plt.figure()
    for edge_tuple in tree.graph.edges:
        qu = tree.get_node_from_id(edge_tuple[0]).q
        qv = tree.get_node_from_id(edge_tuple[1]).q
        plt.plot([qu[0], qv[0]], [qu[1], qv[1]], "k-")

    """
    for node_idx in tree.graph.nodes:
        plt.annotate(str(tree.get_node_from_id(node_idx).value),
            xy=tree.get_node_from_id(node_idx).q)
    """

    plt.plot(goal[0], goal[1], "ro")
    plt.scatter(node_lst[:, 0], node_lst[:, 1])
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])

    plt.show()
