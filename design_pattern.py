"""
Documentation for discussing design patterns.

Scope of abstraction: Our base classes should be applicable to general
graphs of trajectories of dynamical systems. The quasistatic case is a very
big and main child class.
"""

"""===========================================================================
Abstraction of the system class.
This is exactly the irs_lqr.dynamical_system class.
"""
class DynamicalSystem:
    def __init__(self):
        self.h = 0
        self.dim_x = 0
        self.dim_u = 0

    def dynamics(self, x, u):
        return None

"""
Abstraction of the quasistatic system class.
This is exactly the irs_lqr.quasistatic_dynamics.QuasistaticDynamics class.
"""
class QuasistaticSystem(DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.q_sim_py = None
        self.q_sim = None 
        # so on and so forth.
    

"""===========================================================================
Context class where we store "q" that defines a single node.
Everything specific to the application (such as configuration, value)
should be stored in the context.

Outside the context, all of the topological information about the graph 
(e.g. parent-child) that is actually supposed be a part of TreeNode class
will be handled by the NetworkX class.

NOTE: This class should be WITHOUT a method, such that we can call
context.__dict__ and pass it onto node attribute of NetworkX.

NOTE: In the current implementation, the Node class is responsible for quite
a lot of things that should not be owned by the node, including:

- Node class has ownership of the implementation of how to sample, but this
  should be done by the Planner class.
- Node class is responsible for checking if it is reachable to q_goal. This 
  should be done by the Planner class.
- The optimal trajectory should not be owned by the Node, but the Edge class.

We should carefully think through and refactor to make sure node class only
owns what defines a node.
"""
class NodeContext:
    def __init__(self):
        self.q = None
        self.value = None

""" 
Example of a child class that specializes to manipulation.
"""
class ManipNodeContext(NodeContext):
    def __init__(self):
        super().__init__()
        self.in_contact = False

"""===========================================================================
EdgeContext class. Very similar to node class.
Remember that parents and children are handled by NetworkX so it should not be
an attirube of the EdgeContext class. (Rather, it should be an attribute of the
Edge class of NetworkX).
"""
class EdgeContext:
    def __init__(self):
        self.x_trj = None

class ManipEdgeContext(EdgeContext):
    def __init__(self):
        super().__init__()

"""===========================================================================
Planner class. Main class that does tree expansion.
"""
class PlannerParams:
    def __init__(self):
        self.max_size = 1000
        self.system = None # DynamicalSystems 
        self.root_context = None # NodeContext
        # TrajectoryOptimizer Class (e.g. IrsLqrQuasistatic)
        self.trajopt = None         

class Planner:
    def __init__(self, params: PlannerParams):
        self.graph = Digraph() # network x
        self.size = 1 # variable to keep track of nodes
        self.root_node = self.graph.add_node(params.root_context)
        self.system = params.system

    """
    Adds node to the graph owned by the planner.
    only the context should be provided here, the rest should be handled by 
    the internal implementation of the planner.
    """
    def add_node(self, context: NodeContext):
        return None

    """
    Selects node in the existing graph. The child class is responsible for
    deciding the criteria of selection.
    """
    def select_node(self):
        return None

    """
    Extends the node with node_id to some goal context.
    """
    def extend(self, node_id: int, goal_context: NodeContext):
        return None

    """
    Rewires the node with node_id.
    """
    def rewire(self, node_id: int):
        return None

    """
    Iterate the planner until some max_size has been achieved.
    (Similar to the iterate function in irs_lqr, the hope is that this method
    can just be inherited without much changes. It's only a hope...
    """
    def iterate(self):
        return None

"""===========================================================================
GraspSampler class. This is really what ConfigurationSpace should be named 
as.
"""
class GraspSamplerParams:
    def __init__(self):
        self.system = None

class GraspSampler:
    def __init__(self, params: GraspSamplerParams):
        self.system = params.system

    # rest is almost exactly the same as ConfigurationSpace.
    # NOTE: no method of the class should ever depend on the tree.
