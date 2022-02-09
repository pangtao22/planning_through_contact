from typing import Dict
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from tqdm import tqdm
import time

"""
Base node class. Owns all the attributes and methods related to a single node.
Add to nx.Digraph() using 
G.add_node(1, node=Node())
"""
class Node:
    def __init__(self, q):
        self.q = q # np.array of states.
        self.value = None # float.
        self.id = None # int

    def is_reachable(self, q_query):
        return True


"""
Base edge class. Owns all the attributes and methods related to an edge. 
Add to nx.Digraph() using
G.add_edge(1, 2, edge=Edge())
"""
class Edge:
    def __init__(self):
        self.parent = None # Node class.
        self.child = None # Node class.
        self.cost = None # float


"""
Base tree class.
"""
class TreeParams:
    def __init__(self):
        self.max_size = 100
        self.goal = None # q_goal.
        self.root_node = None
        self.eps = np.inf # radius of norm ball for NN queries.


class Tree:
    def __init__(self, params: TreeParams):
        self.graph = nx.DiGraph()
        self.size = 0 # variable to keep track of nodes.
        self.max_size = params.max_size
        self.goal = params.goal
        self.root_node = params.root_node
        self.eps = params.eps
        self.params = params

        self.dim_q = len(self.root_node.q)

        # We keep two data structures over nodes in memory to allow fast
        # batch computation / fast nearest-neighbor queries.

        # First is q_matrix, which is N x n matrix where N is # of nodes, and 
        # n is dim(q). Used for batch computation. Note that we initialize to
        # max_size to save computation time while adding nodes, but only the
        # first N columns of this matrix are "valid".
        self.q_matrix = np.zeros((self.max_size, self.dim_q))
        self.kdtree = None # initialize to empty.

        # Add root node to the graph to finish initialization.
        self.add_node(self.root_node)

    def get_node_from_id(self, id: int):
        node = self.graph.nodes[id]["node"]
        return node

    def get_edge_from_id(self, parent_id: int, child_id: int):
        edge = self.graph.edges[parent_id, child_id]["edge"]
        return edge

    def get_valid_q_matrix(self):
        return self.q_matrix[:self.size, :]

    def add_node(self, node: Node, id=None):
        # Check for missing fields and add them.
        if (id == None):
            self.graph.add_node(self.size, node=node)
            self.q_matrix[self.size,:] = node.q
            node.id = self.size
            self.size += 1                        

        else:
            self.graph.add_node(id, node=node)
            self.q_matrix[id,:] = node.q            
            node.id = id

        # NOTE(terry-suh): We construct a KDTree from scratch because the tree
        # is not built for incremental updates. This results in increased 
        # computation time for adding a node linearly in the size of the tree.
        self.kdtree = KDTree(self.get_valid_q_matrix())

    def replace_node(self, node: Node, id: int):
        self.graph.remove_node(id)
        self.add_node(node, id)

    def add_edge(self, edge: Edge):
        if (edge.cost == None):
            edge.cost = self.compute_edge_cost(edge.parent, edge.child)

        self.graph.add_edge(edge.parent.id, edge.child.id, edge=edge)
        edge.child.value = edge.parent.value + edge.cost

    def remove_edge(self, edge: Edge):
        self.graph.remove_edge(edge.parent.id, edge.child.id)
        # This is like an assert statement that disconnected nodes do not have
        # any value properties.
        edge.child.value = np.nan

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        raise NotImplementedError("This method is virtual.")

    def select_node(self):
        raise NotImplementedError("This method is virtual.")

    def extend(self, node: Node):
        raise NotImplementedError("This method is virtual.")

    def is_close_to_goal(self, node: Node):
        raise NotImplementedError("This method is virtual.")

    """ Get neighbors of the current node in the norm ball. """
    def get_neighbors(self, node: Node, eps: float):
        neighbor_idx = self.kdtree.query_ball_point(node.q, eps)
        return neighbor_idx
        
    def rewire(self, child_node: Node):
        parent_node_id = list(self.graph.predecessors(child_node.id))[0]
        parent_node = self.get_node_from_id(parent_node_id)
        edge = self.get_edge_from_id(parent_node.id, child_node.id)

        best_value = parent_node.value + edge.cost
        best_cost = edge.cost
        new_parent = parent_node

        neighbor_idx = self.get_neighbors(child_node, self.eps)

        # Linear search over the neighbors.
        # TODO(terry-suh): Can this be replaced with batch computation?
        # Maybe not, if reachability cannot be checked in batch....
        for idx in neighbor_idx:
            parent_candidate_node = self.get_node_from_id(idx)
            # Check for reachability from the child node.
            if parent_candidate_node.is_reachable(child_node):
                candidate_edge_cost = self.compute_edge_cost(
                    parent_candidate_node, child_node)

                candidate_value = (
                    parent_candidate_node.value + candidate_edge_cost)

                if (candidate_value < best_value):
                    best_value = candidate_value
                    best_cost = candidate_edge_cost
                    new_parent = parent_candidate_node

        # Replace edge.
        self.remove_edge(
            self.graph.edges[parent_node.id, child_node.id]["edge"])

        new_edge = Edge()
        new_edge.parent = new_parent
        new_edge.child = child_node
        new_edge.cost = best_cost

        self.add_edge(new_edge)

    def iterate(self):
        while(self.size < self.params.max_size):
            # 1. Sample some node from the current tree.
            parent_node = self.select_node()

            # 2. Sample a new child node from the selected node and add 
            #    to the graph.
            child_node = self.extend(parent_node)

            self.add_node(child_node)
            
            edge = Edge()
            edge.parent = parent_node
            edge.child = child_node
            edge.cost = self.compute_edge_cost(edge.parent, edge.child)
            
            child_node.value = parent_node.value + edge.cost

            self.add_edge(edge)

            # 3. Attempt to rewire the extended node.
            # NOTE(terry-suh): In order to guarantee optimality, rewiring
            # should not only run on the current node, but also on the child 
            # nodes. We skip this step to save computation.

            #size_before = np.copy(self.size)
            #self.rewire(child_node)

            # 4. Terminate
            if self.is_close_to_goal():
                print("done!")
                break

class ContactSampler():
    def __init__(self, system):
        # TODO(terry-suh): all the grasp methods should come here.
        raise NotImplementedError("")
