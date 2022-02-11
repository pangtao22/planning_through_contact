from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import pickle


class Node:
    """
    Base node class. Owns all the attributes and methods related to a 
    single node. Add to nx.Digraph() using G.add_node(1, node=Node())
    """    
    def __init__(self, q):
        self.q = q # np.array of states.
        self.value = np.nan # float.
        self.id = np.nan # int


class Edge:
    """
    Base edge class. Owns all the attributes and methods related to an edge.
    Add to nx.Digraph() using G.add_edge(1, 2, edge=Edge())
    """    
    def __init__(self):
        self.parent = None # Node class.
        self.child = None # Node class.
        self.cost = np.nan # float


class RrtParams:
    """
    Base tree parameters class. Only "yaml'-able parameters should be stored
    in the parameters.
    """
    def __init__(self):
        self.max_size = 100
        self.goal = None # q_goal.
        self.root_node = None
        self.subgoal_prob = 0.5
        self.termination_tolerance = 0.1
        self.rewire = True


class Rrt:
    """
    Base tress class.
    """
    def __init__(self, params: RrtParams):
        self.graph = nx.DiGraph()
        self.size = 0 # variable to keep track of nodes.
        self.max_size = params.max_size
        self.goal = params.goal
        self.root_node = params.root_node
        self.termination_tolerance = params.termination_tolerance
        self.subgoal_prob = params.subgoal_prob        
        self.params = params

        self.dim_q = len(self.root_node.q)


        # We keep a matrix over node configurations for batch computation.
        # This is a N x n matrix where N is # of nodes, and 
        # n is dim(q). Used for batch computation. Note that we initialize to
        # max_size to save computation time while adding nodes, but only the
        # first N columns of this matrix are "valid".
        self.q_matrix = np.nan * np.zeros((self.max_size, self.dim_q))

        # Additionally, keep a storage of values.
        self.value_lst = np.nan * np.zeros(self.max_size)
        self.root_node.value = 0.0 # by definition, root node has value of 0.
        self.value_lst[self.size] = self.root_node.value

        # Add root node to the graph to finish initialization.
        self.add_node(self.root_node)

    def get_node_from_id(self, id: int):
        """ Return node from the graph given id. """
        node = self.graph.nodes[id]["node"]
        return node

    def get_edge_from_id(self, parent_id: int, child_id: int):
        """ Return edge from the graph given id of parent and child. """
        edge = self.graph.edges[parent_id, child_id]["edge"]
        return edge

    def get_valid_q_matrix(self):
        """ Get slice of q matrix with valid components."""
        return self.q_matrix[:self.size, :]

    def get_valid_value_lst(self):
        """ Get slice of value_lst with valid components."""
        return self.value_lst[:self.size]

    def add_node(self, node: Node):
        """ 
        Add a new node to the networkx graph and the relevant data structures
        that does batch computation. This also populates the id parameter of
        the node.
        """
        self.graph.add_node(self.size, node=node)
        self.q_matrix[self.size,:] = node.q
        node.id = self.size
        self.size += 1

    def replace_node(self, node: Node, id: int):
        """
        Replaces a node in a graph with id with a new given node. Also 
        changes the relevant data structures.
        """
        self.graph.remove_node(id)
        self.graph.add_node(id, node=node)
        self.q_matrix[id,:] = node.q
        node.id = id

    def add_edge(self, edge: Edge):
        """
        Add an edge to the graph. This computes the cost of the edge and
        populates the value of the connected child.
        """
        if np.isnan(edge.cost):
            raise ValueError(
                "Attempting to add edge, but the edge does not have a cost "
             +  "assigned.")
            
        self.graph.add_edge(edge.parent.id, edge.child.id, edge=edge)
        edge.child.value = edge.parent.value + edge.cost
        self.value_lst[edge.child.id] = edge.child.value

    def remove_edge(self, edge: Edge):
        """ Remove edge from the graph. """
        self.graph.remove_edge(edge.parent.id, edge.child.id)
        edge.child.value = np.nan
        self.value_lst[edge.child.value] = edge.child.value

    def cointoss_for_goal(self):
        sample_goal = np.random.choice(
            [0, 1], 1, p=[
                1 - self.params.subgoal_prob, self.params.subgoal_prob])
        return sample_goal

    def sample_subgoal(self):
        """ Provide a method to sample the a subgoal. """
        raise NotImplementedError("This method is virtual.")

    def select_closest_node(self, subgoal: np.array):
        """
        Given a subgoal, and find the node that is closest from the subgoal.
        """
        metric_batch = self.calc_metric_batch(subgoal)
        selected_node = self.get_node_from_id(np.argmin(metric_batch))
        return selected_node

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """ Extend current node towards a specified configuration q. """
        raise NotImplementedError("This method is virtual.")

    def extend(self, node: Node, subgoal: np.array):
        """
        Extend towards a self.subgoal object. This is evaluated analytically.
        """
        return self.extend_towards_q(node, subgoal)

    def calc_metric_batch(self, q_query: np.array):
        """
        Given q_query, return a np.array of \|q_query - q\| according to some
        local distances from all the existing nodes in the tree to q_query.
        """
        raise NotImplementedError("This method is virtual.")

    def is_close_to_goal(self):
        """
        Evaluate termination criteria for RRT using global distance metric.
        """
        dist_batch = self.calc_metric_batch(self.params.goal)
        return np.min(dist_batch) < self.params.termination_tolerance        
        
    def rewire(self, parent_node: Node, child_node: Node):
        """
        Rewiring step. Loop over neighbors, query for the new value after
        rewiring to candidate parents, and rewire if the value is lower.
        """
        # Compute distance from candidate child_node to all the nodes.
        dist_lst = self.calc_metric_batch(child_node.q)
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
        pbar = tqdm(total = self.max_size)

        while(self.size < self.params.max_size):
            pbar.update(1)

            # 1. Sample a subgoal.
            sample_goal = self.cointoss_for_goal()
            if (sample_goal):
                subgoal = self.params.goal
            else:
                subgoal = self.sample_subgoal()

            # 2. Sample closest node to subgoal
            parent_node = self.select_closest_node(subgoal)

            # 3. Extend to subgoal.
            child_node, edge = self.extend(parent_node, subgoal)

            # 4. Attempt to rewire a candidate child node.
            if (self.params.rewire):
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node)

            # 5. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_close_to_goal():
                break

        pbar.close()

    def save_tree(self, filename):
        """
        NOTE: it seems possible to convert both the Node and Edge classes in
         rrt_base.py into dictionaries, which can be saved as attributes as
         part of a networkx node/edge. The computational methods in the
         Node class can be moved into a "reachable set" class.
         This conversion has a few benefits:
         - accessing attributes becomes easier:
            self.graph[node_id]['node'].attribute_name  will become
            self.graph[node_id]['attribute_name']
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f)

    def save_final_path(self, filename):
        # Find closest to the goal.
        q_final = self.select_closest_node(self.params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id)

        dim_u = len(self.get_edge_from_id(path[0], path[1]).u)
        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_q))
        u_trj = np.zeros((path_T - 1, dim_u))

        for i in range(path_T - 1):
            x_trj[i,:] = self.get_node_from_id(path[i]).q
            u_trj[i,:] = self.get_edge_from_id(path[i], path[i+1]).u
        x_trj[path_T-1,:] = self.get_node_from_id(path[path_T-1]).q

        path_dict = {
            "x_trj": x_trj, "u_trj": u_trj}

        with open(filename, 'wb') as f:
            pickle.dump(path_dict, f)
