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

    def sample_node_from_tree(self):
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

        max_sample_dist = 0.0
        max_sample = None

        for i in range(len(theta_samples)):
            # For each sample, find minimum distance.
            theta = theta_samples[i]
            min_dist = np.inf
            for j in range(self.size):
                dist = np.linalg.norm(
                    np.array(q_now + [np.cos(theta), np.sin(theta)]) - 
                    self.graph.nodes[j]["node"].q
                )

                if dist < min_dist:
                    min_dist = dist

            if min_dist > max_sample_dist:
                max_sample_dist = min_dist
                max_sample = i

        # Construct node.
        theta_best = theta_samples[max_sample]
        new_node = Node()
        new_node.q = q_now + np.array([np.cos(theta_best), np.sin(theta_best)])
        return new_node

    def termination(self):
        for i in range(self.size):
            dist = np.linalg.norm(self.graph.nodes[i]["node"].q - self.goal)
            if (dist < 0.1):
                return True
        return False

root_node = Node()
root_node.q = np.zeros(2)
root_node.value = 0.0

goal = 100.0 * np.ones(2)

params = PoissonParams()
params.root_node = root_node
params.goal = goal

tree = PoissonTree(params)
tree.iterate()

node_lst = []
for i in range(tree.size):
    node_lst.append(tree.graph.nodes[i]["node"].q)
node_lst = np.array(node_lst)

plt.figure()
plt.scatter(node_lst[:,0], node_lst[:,1])
plt.show()