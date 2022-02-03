import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from irs_rrt.rrt_base import Node, Edge, TreeParams, Tree

class PoissonParams(TreeParams):
    def __init__(self):
        super().__init__()
        self.radius = 1.0

class PoissonTree(Tree):
    def __init__(self, params: TreeParams):
        super().__init__(params)

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        x_parent = parent_node.q
        x_child = child_node.q
        return np.linalg.norm(x_parent - x_child)

    def select_node_from_tree(self):
        # Choose the sample furthest away from origin.

        max_sample_dist = 0.0
        max_sample = None
        
        for i in range(self.size):
            dist = np.linalg.norm(self.graph.nodes[i]["node"].q)
            if dist >= max_sample_dist:
                max_sample_dist = dist
                max_sample = i

        max_sample = np.random.randint(self.size)
        return self.graph.nodes[max_sample]["node"]

    def extend(self, node:Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """
        q_now = node.q
        theta_samples = 2.0 * np.pi * np.random.rand(100)

        sample_coords = q_now[:,None] + np.array(
            [np.cos(theta_samples), np.sin(theta_samples)])
        sample_coords = sample_coords.transpose()

        existing_coords = self.get_valid_q_matrix()

        pairwise_distance = np.linalg.norm(
            sample_coords[:,None,:] - existing_coords[None,:,:], axis=-1)

        best_idx = np.argmax(np.min(pairwise_distance, axis=1), axis=0)

        new_node = Node(sample_coords[best_idx,:])
        return new_node

    def termination(self):
        for i in range(self.size):
            dist = np.linalg.norm(self.graph.nodes[i]["node"].q - self.goal)
            if (dist < 0.1):
                return True
        return False

root_node = Node(np.zeros(2))
root_node.value = 0.0

goal = 100.0 * np.ones(2)

params = PoissonParams()
params.root_node = root_node
params.goal = goal
params.eps = 1.5

tree = PoissonTree(params)
tree.iterate()

node_lst = []
for i in range(tree.size):
    node_lst.append(tree.graph.nodes[i]["node"].q)
node_lst = np.array(node_lst)

plt.figure()
for edge_tuple in tree.graph.edges:
    qu = tree.get_node_from_id(edge_tuple[0]).q
    qv = tree.get_node_from_id(edge_tuple[1]).q
    plt.plot([qu[0], qv[0]], [qu[1], qv[1]], 'k-')

for node_idx in tree.graph.nodes:
    plt.annotate(str(tree.get_node_from_id(node_idx).value),
        xy=tree.get_node_from_id(node_idx).q)


plt.scatter(node_lst[:,0], node_lst[:,1])
plt.show()