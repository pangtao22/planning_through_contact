import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from irs_rrt.rrt_base import Node, Edge, TreeParams, Tree

class EuclideanTreeParams(TreeParams):
    def __init__(self):
        super().__init__()

        self.n_samples = 100
        self.termination_tolerance = 1.0        
        self.cvar_bestk = 20
        
        self.select_prob = {
            "explore_max": 0.0,
            "explore_cvar": 1.0,
            "towards_goal_max": 0.0,
            "towards_goal_cvar": 0.0,
            "random": 0.0
        }

        self.extend_prob = {
            "explore_max": 0.7,
            "explore_cvar": 0.0,
            "towards_goal_max": 0.3,
            "towards_goal_cvar": 0.0,
            "random": 0.0
        }

class EuclideanTree(Tree):
    def __init__(self, params: TreeParams):
        super().__init__(params)

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        x_parent = parent_node.q
        x_child = child_node.q
        return np.linalg.norm(x_parent - x_child)

    def get_dist_to_root(self):
        dist_lst = []
        for i in range(self.size):
            dist_lst.append(
                np.linalg.norm(
                    self.get_node_from_id(i).q - self.root_node.q))
        return dist_lst

    def get_dist_to_goal(self):
        dist_lst = []
        for i in range(self.size):
            dist_lst.append(
                np.linalg.norm(self.get_node_from_id(i).q - self.goal))
        return dist_lst

    def choose_cvar(self, dist_lst, order='max'):
        k = np.min((self.params.cvar_bestk, self.size))
        if (order == 'max'):
            best_k_idx = np.argpartition(dist_lst, -k)[-k:]
        else:
            best_k_idx = np.argpartition(dist_lst, k-1)[:k]
        idx = best_k_idx[np.random.randint(len(best_k_idx))]
        return idx

    def select_node_explore_max(self):
        # Choose the sample furthest away from origin.
        dist_lst = self.get_dist_to_root()
        max_idx = np.argmax(dist_lst)
        return self.get_node_from_id(max_idx)

    def select_node_explore_cvar(self):
        # Randomly sample from k furthest samples from origin.
        dist_lst = self.get_dist_to_root()
        print(dist_lst)
        idx = self.choose_cvar(dist_lst, order='max')
        return self.get_node_from_id(idx)

    def select_node_towards_goal_max(self):
        # Choose the sample closest to the goal.
        dist_lst = self.get_dist_to_goal()
        idx = np.argmin(dist_lst)
        return self.get_node_from_id(idx)

    def select_node_towards_goal_cvar(self):
        # Randomly sample from k closest samples from goal.
        dist_lst = self.get_dist_to_goal()
        idx = self.choose_cvar(dist_lst, order='min')
        return self.get_node_from_id(idx)

    def select_node_random(self):
        # Uniformly sample from existing nodes.
        idx = np.random.randint(self.size)
        return self.get_node_from_id(idx)

    def select_node(self):
        """
        Wrapper method that tosses a dice to decide between the strategies 
        according to self.params.select_prob dictionary.
        """
        mode = np.random.choice(
            list(self.params.select_prob.keys()), 1,
            p = list(self.params.select_prob.values()))

        if (mode == "explore_max"):
            selected_node = self.select_node_explore_max()
        elif (mode == "explore_cvar"):
            selected_node = self.select_node_explore_cvar()
        elif (mode == "towards_goal_max"):
            selected_node = self.select_node_towards_goal_max()
        elif (mode == "towards_goal_cvar"):
            selected_node = self.select_node_towards_goal_cvar()
        elif (mode == "random"):
            selected_node = self.select_node_random()
        else:
            selected_node = self.select_node_random()

        return selected_node

    def compute_pairwise_distance(self, sample_coords, existing_coords):
        return np.linalg.norm(
            sample_coords[:,None,:] - existing_coords[None,:,:], axis=-1)

    def sample_from_node(self, node: Node):
        q_now = node.q
        theta_samples = 2.0 * np.pi * np.random.rand(self.params.n_samples)
        sample_coords = q_now[:,None] + np.array(
            [np.cos(theta_samples), np.sin(theta_samples)])
        sample_coords = sample_coords.transpose()
        return sample_coords

    def compute_chamfer_distance(self, node):
        sample_coords = self.sample_from_node(node)
        existing_coords = self.get_valid_q_matrix()

        pairwise_distance = self.compute_pairwise_distance(
            sample_coords, existing_coords)
        chamfer_distance = np.min(pairwise_distance, axis=1)
        return chamfer_distance, sample_coords
    
    def extend_explore_max(self, node:Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """
        chamfer_distance, sample_coords = self.compute_chamfer_distance(node)
        idx = np.argmax(chamfer_distance, axis=0)
        return Node(sample_coords[idx,:])

    def extend_explore_cvar(self, node: Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """        
        chamfer_distance, sample_coords = self.compute_chamfer_distance(node)
        idx = self.choose_cvar(chamfer_distance, order='max')
        return Node(sample_coords[idx,:])

    def sample_distance_to_goal(self, node):
        sample_coords = self.sample_from_node(node)
        dist_lst = np.linalg.norm(sample_coords - self.goal[None,:], axis=1)
        return dist_lst, sample_coords

    def extend_towards_goal_max(self, node:Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """
        dist_lst, sample_coords = self.sample_distance_to_goal(node)
        idx = np.argmin(dist_lst, axis=0)
        return Node(sample_coords[idx,:])

    def extend_towards_goal_cvar(self, node:Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """
        dist_lst, sample_coords = self.sample_distance_to_goal(node)
        idx = self.choose_cvar(dist_lst, order='min')
        return Node(sample_coords[idx,:])

    def extend_random(self, node:Node):
        """
        Among samples from radius 1, choose one that is the furthest away from
        all the existing nodes.
        """
        sample_coords = self.sample_from_node(node)
        idx = np.random.randint(self.params.n_samples)
        return Node(sample_coords[idx,:])

    def extend(self, node:Node):
        """
        Wrapper method that tosses a dice to decide between the strategies 
        according to self.params.select_prob dictionary.
        """
        mode = np.random.choice(
            list(self.params.extend_prob.keys()), 1,
            p = list(self.params.extend_prob.values()))

        if (mode == "explore_max"):
            selected_node = self.extend_explore_max(node)
        elif (mode == "explore_cvar"):
            selected_node = self.extend_explore_cvar(node)
        elif (mode == "towards_goal_max"):
            selected_node = self.extend_towards_goal_max(node)
        elif (mode == "towards_goal_cvar"):
            selected_node = self.extend_towards_goal_cvar(node)
        elif (mode == "random"):
            selected_node = self.extend_random(node)
        else:
            selected_node = self.extend_random(node)

        return selected_node

    def is_close_to_goal(self):
        dist_lst = self.get_dist_to_goal()
        if np.min(dist_lst) < self.params.termination_tolerance:
            return True
        return False

root_node = Node(np.zeros(2))
root_node.value = 0.0

goal = 10.0 * np.ones(2)

params = EuclideanTreeParams()
params.root_node = root_node
params.goal = goal
params.eps = 1.5

tree = EuclideanTree(params)
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

#for node_idx in tree.graph.nodes:
#    plt.annotate(str(tree.get_node_from_id(node_idx).value),
#        xy=tree.get_node_from_id(node_idx).q)


plt.plot(goal[0], goal[1], 'ro')
plt.scatter(node_lst[:,0], node_lst[:,1])
plt.xlim([-15, 15])
plt.ylim([-15, 15])

plt.show()