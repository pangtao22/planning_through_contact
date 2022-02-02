from typing import Dict
import numpy as np
import networkx as nx

"""
Base node class. Owns all the attributes and methods related to a single node.
Add to nx.Digraph() using 
G.add_node(1, node=Node())
"""
class Node:
    def __init__(self):
        self.q = None # np.array of states.
        self.value = None # float.
        self.id = None # int

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
        self.max_size = 300
        self.system = None
        self.goal = None # q_goal.
        self.root_node = None

class Tree:
    def __init__(self, params: TreeParams):
        self.graph = nx.DiGraph()
        self.size = 0 # variable to keep track of nodes.
        self.system = params.system
        self.max_size = params.max_size
        self.goal = params.goal
        self.root_node = params.root_node

        # Add root node to the graph to finish initialization.
        self.add_node(self.root_node)

    def add_node(self, node: Node):
        # Check for missing fields and add them.
        if (node.id == None):
            node.id = self.size

        self.graph.add_node(self.size, node=node)
        self.size += 1

    def add_edge(self, edge: Edge):
        if (edge.cost == None):
            edge.cost = self.compute_edge_cost(edge.parent, edge.child)

        self.graph.add_edge(edge.parent.id, edge.child.id, edge=edge)

    def remove_edge(self, edge: Edge):
        self.graph.add_edge(edge.parent.id, edge.child.id, edge=edge)        

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        raise NotImplementedError("This method is virtual")

    def sample_node_from_tree(self):
        raise NotImplementedError("This method is virtual")

    def extend(self, node: Node):
        raise NotImplementedError("This method is virtual")

    def termination(self, node: Node):
        raise NotImplementedError("This method is virtual")        
        
    def rewire(self, child_node: Node):
        parent_node_idx = list(self.graph.predecessors(child_node.id))[0]
        parent_node = self.graph.nodes[parent_node_idx]["node"]

        edge = self.graph.edges[parent_node.id, child_node.id]["edge"]

        best_value = parent_node.value + edge.cost
        best_cost = edge.cost
        new_parent = parent_node

        for idx in list(self.graph):
            if (idx == child_node.id): pass
            else:
                parent_candidate_node = self.graph.nodes[idx]["node"]
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
        new_edge.cost = candidate_edge_cost

        self.add_edge(new_edge)

    def iterate(self):
        while (self.size < self.max_size):
            # 1. Sample some node from the current tree.
            parent_node = self.sample_node_from_tree()

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
            self.rewire(child_node)

            # 4. Terminate
            if self.termination():
                break
