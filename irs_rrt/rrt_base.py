import pickle

import networkx as nx
import numpy as np
from irs_rrt.rrt_params import RrtParams
from tqdm import tqdm

from dash_vis.dash_common import trace_nodes_to_root_from


class Node:
    """
    Base node class. Owns all the attributes and methods related to a
    single node. Add to nx.Digraph() using G.add_node(1, node=Node())
    """

    def __init__(self, q):
        self.q = q  # np.array of states.
        self.value = np.nan  # float.
        self.id = np.nan  # int
        # To extend the tree, a subgoal is sampled first, and then a new node
        # that is "as close as possible" to the subgoal is added to the tree.
        # This field stores the subgoal associated with the new node.
        self.subgoal = None


class Edge:
    """
    Base edge class. Owns all the attributes and methods related to an edge.
    Add to nx.Digraph() using G.add_edge(1, 2, edge=Edge())
    """

    def __init__(self):
        self.parent = None  # Node class.
        self.child = None  # Node class.
        self.cost = np.nan  # float


class Rrt:
    def __init__(self, params: RrtParams):
        self.graph = nx.DiGraph()
        self.size = 0  # variable to keep track of nodes.
        self.max_size = params.max_size
        self.goal = params.goal
        self.root_node = params.root_node
        self.termination_tolerance = params.termination_tolerance
        self.subgoal_prob = params.goal_as_subgoal_prob
        self.rrt_params = params

        self.dim_q = len(self.root_node.q)

        # We keep a matrix over node configurations for batch computation.
        # This is a N x n matrix where N is # of nodes, and
        # n is dim(q). Used for batch computation. Note that we initialize to
        # max_size to save computation time while adding nodes, but only the
        # first N columns of this matrix are "valid".
        self.q_matrix = np.full((self.max_size, self.dim_q), np.nan)

        # Additionally, keep a storage of values.
        self.value_lst = np.full(self.max_size, np.nan)
        self.root_node.value = 0.0  # by definition, root node has value of 0.
        self.value_lst[self.size] = self.root_node.value

        # Add root node to the graph to finish initialization.
        self.add_node(self.root_node)

        self.goal_node_idx = None

    def get_node_from_id(self, id: int):
        """Return node from the graph given id."""
        node = self.graph.nodes[id]["node"]
        return node

    def get_edge_from_id(self, parent_id: int, child_id: int):
        """Return edge from the graph given id of parent and child."""
        edge = self.graph.edges[parent_id, child_id]["edge"]
        return edge

    def get_q_matrix_up_to(self, size: int = None):
        """Get slice of q matrix with valid components."""
        if size is None:
            size = self.size
        return self.q_matrix[:size, :]

    def get_valid_value_lst(self):
        """Get slice of value_lst with valid components."""
        return self.value_lst[: self.size]

    def add_node(self, node: Node):
        """
        Add a new node to the networkx graph and the relevant data structures
        that does batch computation. This also populates the id parameter of
        the node.
        """
        self.graph.add_node(self.size, node=node)
        self.q_matrix[self.size, :] = node.q
        node.id = self.size
        self.size += 1

    def replace_node(self, node: Node, id: int):
        """
        Replaces a node in a graph with id with a new given node. Also
        changes the relevant data structures.
        """
        self.graph.remove_node(id)
        self.graph.add_node(id, node=node)
        self.q_matrix[id, :] = node.q
        node.id = id

    def add_edge(self, edge: Edge):
        """
        Add an edge to the graph. This computes the cost of the edge and
        populates the value of the connected child.
        """
        if np.isnan(edge.cost):
            raise ValueError(
                "Attempting to add edge, but the edge does not have a cost "
                + "assigned."
            )

        self.graph.add_edge(edge.parent.id, edge.child.id, edge=edge)
        edge.child.value = edge.parent.value + edge.cost
        self.value_lst[edge.child.id] = edge.child.value

    def remove_edge(self, edge: Edge):
        """Remove edge from the graph."""
        self.graph.remove_edge(edge.parent.id, edge.child.id)
        edge.child.value = np.nan
        self.value_lst[edge.child.value] = edge.child.value

    def cointoss_for_goal(self):
        if np.random.rand() < self.rrt_params.goal_as_subgoal_prob:
            return True
        return False

    def sample_subgoal(self):
        """Provide a method to sample the a subgoal."""
        raise NotImplementedError("This method is virtual.")

    def select_closest_node(
        self, subgoal: np.array, print_distance: bool = False
    ):
        """
        Given a subgoal, and find the node that is closest from the subgoal.
        """
        d_batch = self.calc_distance_batch(subgoal)
        selected_node = self.get_node_from_id(np.argmin(d_batch))
        if print_distance:
            print("closest distance to subgoal", d_batch[selected_node.id])
        return selected_node

    def find_node_closest_to_goal(self):
        return self.select_closest_node(
            self.rrt_params.goal, print_distance=True
        )

    def trace_nodes_to_root_from(self, i_node: int):
        return trace_nodes_to_root_from(i_node, self.graph)

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """Extend current node towards a specified configuration q."""
        raise NotImplementedError("This method is virtual.")

    def extend(self, node: Node, subgoal: np.array):
        """
        Extend towards a self.subgoal object. This is evaluated analytically.
        """
        return self.extend_towards_q(node, subgoal)

    def calc_distance_batch(self, q_query: np.array):
        """
        Given q_query, return a np.array of \|q_query - q\| according to some
        local distances from all the existing nodes in the tree to q_query.
        """
        raise NotImplementedError("This method is virtual.")

    def is_close_to_goal(self):
        """
        Evaluate termination criteria for RRT using global distance metric.
        """
        dist_batch = self.calc_distance_batch(self.rrt_params.goal)
        return np.min(dist_batch) < self.rrt_params.termination_tolerance

    def rewire(self, parent_node: Node, child_node: Node):
        """
        Rewiring step. Loop over neighbors, query for the new value after
        rewiring to candidate parents, and rewire if the value is lower.
        """
        # Compute distance from candidate child_node to all the nodes.
        dist_lst = self.calc_distance_batch(child_node.q)
        value_candidate_lst = self.get_valid_value_lst() + dist_lst

        # Get neighboring indices.
        # NOTE(terry-suh): this is "conservative rewiring" that does not
        # require user input of rewiring tolerance. In practice, this feels
        # better since there is one less arbitrary hyperparameter.
        neighbor_idx = np.argwhere(dist_lst <= dist_lst[parent_node.id])
        min_idx = np.argmin(value_candidate_lst[neighbor_idx])

        new_parent = self.get_node_from_id(neighbor_idx[min_idx][0])
        new_child, new_edge = self.extend_towards_q(new_parent, child_node.q)

        return new_parent, new_child, new_edge

    def iterate(self):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        while self.size < self.rrt_params.max_size:
            pbar.update(1)

            # 1. Sample a subgoal.
            if self.cointoss_for_goal():
                subgoal = self.rrt_params.goal
            else:
                subgoal = self.sample_subgoal()

            # 2. Sample closest node to subgoal
            parent_node = self.select_closest_node(subgoal)

            # 3. Extend to subgoal.
            child_node, edge = self.extend(parent_node, subgoal)

            # 4. Attempt to rewire a candidate child node.
            if self.rrt_params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node
                )

            # 5. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_close_to_goal():
                self.goal_node_idx = child_node
                break

        pbar.close()

    def save_tree(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f)

    def save_final_path(self, filename):
        # Find closest to the goal.
        q_final = self.select_closest_node(self.rrt_params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id
        )

        dim_u = len(self.get_edge_from_id(path[0], path[1]).u)
        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_q))
        u_trj = np.zeros((path_T - 1, dim_u))

        for i in range(path_T - 1):
            x_trj[i, :] = self.get_node_from_id(path[i]).q
            u_trj[i, :] = self.get_edge_from_id(path[i], path[i + 1]).u
        x_trj[path_T - 1, :] = self.get_node_from_id(path[path_T - 1]).q

        path_dict = {"x_trj": x_trj, "u_trj": u_trj}

        with open(filename, "wb") as f:
            pickle.dump(path_dict, f)
