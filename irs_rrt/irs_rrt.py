from typing import Dict
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Tree, TreeParams
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel


class IrsTreeParams(TreeParams):
    def __init__(self, q_dynamics):
        super().__init__()
        self.q_dynamics = q_dynamics  # QuasistaticDynamics class.
        self.q_dynamics_p = QuasistaticDynamicsParallel(
            self.q_dynamics)

        # Options for computing bundled dynamics.
        self.n_samples = 100
        self.std_u = 0.1 * np.array(self.q_dynamics.dim_u)  # std_u for input.
        self.decouple_AB = True
        self.bundle_mode = BundleMode.kFirst
        self.parallel_mode = ParallelizationMode.kZmq

        # Regularization for computing inverse of covariance matrices.
        # NOTE(terry-suh): Note that if the covariance matrix is singular,
        # then the Mahalanobis distance metric is infinity. One interpretation
        # of the regularization term is to cap the infinity distance to some
        # value that scales with inverse of regularization.
        self.regularization = 1e-5

        # Rewiring tolerance. If the distance from qi to qj is less than this
        # tolerance as evaluated by the local distance metric on qi, then
        # qi will be included in the candidate for rewiring.
        self.rewire_tolerance = 10.0

        # Termination tolerance. Algorithm will terminate if there exists a
        # node such that the norm between the node and the goal is less than
        # this threshold.
        self.termination_tolerance = 10.0

        # The global distance metric that compares qi and qj in configuration
        # space. This metric is not used for reachability criterion, but is
        # used for assigning costs to each edge of RRT, as well as termination
        # criteria. The metric should be a vector of dim_x, which will be 
        # diagonalized and put into matrix norm form.
        self.global_metric = np.ones(self.q_dynamics.dim_x)

        # Stepsize.
        # TODO(terry-suh): the selection of this parameter should be automated.
        self.stepsize = 0.3

        # Probability to choose between different strategies for selection.
        self.select_prob = {
            "explore": 0.3,
            "random": 0.2,
            "towards_goal": 0.5
        }

        # Probability to choose between different strategies for extend.
        self.extend_prob = {
            "explore": 0.7,
            "random": 0.1,
            "contact": 0.0,
            "towards_goal": 0.2
        }


class IrsNode(Node):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """
    def __init__(self, q: np.array, params: IrsTreeParams):
        super().__init__(q)
        # Bundled dynamics parameters.
        self.params = params
        self.q_dynamics = params.q_dynamics  # QuasistaticDynamics class.
        self.q_dynamics_p = params.q_dynamics_p  # QuasistaticDynamics class.
        self.std_u = params.std_u  # Injected noise for input.

        self.ubar = self.q[self.q_dynamics.get_u_indices_into_x()]

        # NOTE(terry-suh): This should ideally use calc_Bundled_ABc from the
        # quasistatic dynamics class, but we are not doing this here because
        # the ct returned by calc_bundled_ABc works on exact dynamics.
        self.Bhat, self.chat = self.compute_bundled_Bc()

        # Parameters of the Gaussian based on computed Bhat / chat.
        self.mu, self.cov = self.compute_metric_parameters()
        self.covinv = np.linalg.inv(self.cov)

    def compute_bundled_Bc(self):
        """
        Compute bundled dynamics on Bc. 
        TODO(terry-suh): accept user input on BundleMode and 
        ParallelizationMode.
        """
        # Compute bundled dynamics.
        fhat = self.q_dynamics_p.dynamics_bundled(
            self.q, self.ubar, self.params.n_samples, self.params.std_u)

        # Get the B matrix.
        x_trj = np.zeros((2, self.q_dynamics_p.dim_x))
        u_trj = np.zeros((1, self.q_dynamics_p.dim_u))

        x_trj[0, :] = self.q
        u_trj[0, :] = self.ubar

        Ahat, Bhat = self.q_dynamics_p.calc_bundled_AB_cpp(
            x_trj, u_trj, self.params.std_u, self.params.n_samples, False)
        chat = fhat

        return Bhat[0, :, :], chat

    def compute_metric_parameters(self):
        """
        Compute parameters of the Gaussian associated with this node,
        which parametrizes teh Mehalanobis distance.
        This computation is separated out as a method for users to allow
        first-order (B @ B.T) or zero-order (SVD) variants.
        TODO(terry-suh): This should always be B @ B.T. The difference should 
        be whether or not B comes from first / zero order computation.
        Ask me for derivation of why.
        """
        cov = self.Bhat @ self.Bhat.T + self.params.regularization * np.eye(
            self.q_dynamics.dim_x)
        mu = self.chat
        return mu, cov

    def eval_bundled_dynamics(self, du: np.array):
        """
        Evaluate effect of applying du on the bundle-linearized dynamics.
        du: shape (dim_u,)
        """
        xhat_next = self.Bhat.dot(du) + self.chat
        return xhat_next

    def eval_bundled_dynamics_batch(self, du_batch: np.array):
        """
        Evaluate effect of applying du on the bundle-linearized dynamics.
        du: shape (dim_b, dim_u)
        """
        dim_b = du_batch.shape[0]
        xhat_next_batch = (
                self.Bhat.dot(du_batch.transpose()).transpose() + self.chat)
        return xhat_next_batch

    def eval_metric(self, q_query):
        """
        Evaluate the distance between the q_query and self.q, as informed by
        the L2 norm on the basis that spans the bundled B matrix.
        This is also known as the Mahalanobis distance.
        """
        return (q_query - self.mu).T @ self.covinv @ (q_query - self.mu)

    def eval_metric_batch(self, q_query_batch):
        """
        Do eval_metric in batch.
        q_query_batch: (dim_b, dim_x)
        returns: (dim_b)
        """
        batch_error = q_query_batch - self.mu[None, :]
        metric_batch = np.diagonal(
            batch_error @ self.covinv @ batch_error.T)
        return metric_batch

    def sample_exact_gaussian(self, n_samples: int, std_u: np.array):
        """
        Given number of samples and standard deviation, sample around the node
        according to exact dynamic rollouts.
        """
        u_batch = np.random.normal(self.ubar, std_u)
        x_batch = np.tile(self.q[:, None], (1, n_samples)).transpose()
        xnext_batch = self.dynamics_p.dynamics_batch(x_batch, u_batch)
        return xnext_batch

    def sample_bundled_gaussian(self, n_samples: int, std_u: np.array):
        """
        Given number of samples and standard deviation, sample around the node
        according to bundled dynamics.
        """
        du_batch = np.random.normal(np.zeros(self.q_dynamics.dim_u), std_u)
        xnext_hat_batch = self.eval_bundled_dynamics_batch(du_batch)
        return xnext_hat_batch

    def sample_exact_step(self, n_samples: int, step_size: float):
        """
        Given a number of samples and step_size, sample around the node
        according to exact dynamic rollouts.
        """
        # Normalize du by step_size then add back ubar to bring to nominal
        # coordinates.
        u_batch = np.random.rand(n_samples, self.q_dynamics.dim_u) - 0.5
        u_batch = u_batch / np.linalg.norm(u_batch, axis=1)[:, None]
        u_batch = step_size * u_batch + self.u_bar

        x_batch = np.tile(self.q[:, None], (1, n_samples)).transpose()
        xnext_batch = self.dynamics_p.dynamics_batch(x_batch, u_batch)
        return xnext_batch

    def sample_bundled_step(self, n_samples: int, step_size: float):
        """
        Given a number of samples and step_size, sample around the node
        according to bundled dynamic rollouts.
        """
        du_batch = np.random.rand(n_samples, self.q_dynamics.dim_u) - 0.5
        du_batch = du_batch / np.linalg.norm(du_batch, axis=1)[:, None]
        du_batch = step_size * du_batch

        xnext_hat_batch = self.eval_bundled_dynamics_batch(du_batch)
        return xnext_hat_batch

    def find_step_size(self, n_samples: int, max_tolerance: float):
        """
        Using the discrepeancy between bundled and exact dynamics, find a 
        tolerable step size according to some max_tolerance parameter.

        This method attempts to approximately compute the following stepsize:
        
        max (step_size) subject to (exact - bundled <= max_tolerance)
        and some bounding box constraints on step_size.
        """
        raise NotImplementedError("Need to implement this.")

    def is_reachable(self, child_node):
        """
        Is q_query reachable from self.q? Implement thresholding according to
        the Mahalanobis distance on bundled B matrix.
        """
        dist = self.eval_metric(child_node.q)
        return (dist < self.params.rewire_tolerance)


class IrsEdge(Edge):
    """
    IrsEdge.
    """
    def __init__(self):
        super().__init__()
        self.du = None
        # NOTE(terry-suh): It is possible to store trajectories in the edge
        # class. We won't do that here because we don't solve trajopt during
        # extend.


class IrsTree(Tree):
    def __init__(self, params: TreeParams):
        super().__init__(params)
        self.q_dynamics = params.q_dynamics

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        diff = parent_node.q - child_node.q
        return diff.T @ np.diag(self.params.global_metric) @ diff

    def is_close_to_goal(self):
        diff = self.get_valid_q_matrix() - self.params.goal
        dist_batch = np.diagonal(
            diff @ np.diag(self.params.global_metric) @ diff.T)
        return np.min(dist_batch) < self.params.termination_tolerance

    """
    Methods for selecting node from the existing tree.
    """

    def select_node_random(self):
        """
        Select randomly from existing nodes in the tree.
        """
        node_idx = np.random.randint(self.size)
        return self.get_node_from_id(node_idx)

    def select_node_explore(self):
        """
        Select node that is furthest away from the origin by the global
        distance metric.
        """
        # Compute distance metric in batch.
        diff = self.get_valid_q_matrix() - self.root_node.q[None, :]
        dist_batch = np.diagonal(
            diff @ np.diag(self.params.global_metric) @ diff.T)
        max_idx = np.argmax(dist_batch)
        return self.get_node_from_id(max_idx)

    def select_node_towards_goal(self):
        """
        Select node that is closest from the goal as judged by the global
        distance metric.
        """
        # Compute distance metric in batch.
        diff = self.get_valid_q_matrix() - self.params.goal
        dist_batch = np.diagonal(
            diff @ np.diag(self.params.global_metric) @ diff.T)
        min_idx = np.argmin(dist_batch)
        return self.get_node_from_id(min_idx)

    def select_node(self):
        """
        Wrapper method that tosses a dice to decide between the strategies 
        according to self.params.select_prob dictionary.
        """
        mode = np.random.choice(
            list(self.params.select_prob.keys()), 1,
            p=list(self.params.select_prob.values()))

        if (mode == "explore"):
            selected_node = self.select_node_explore()
        elif (mode == "random"):
            selected_node = self.select_node_random()
        elif (mode == "towards_goal"):
            selected_node = self.select_node_towards_goal()
        else:
            selected_node = self.select_node_random()

        return selected_node

    """
    Methods for extending node from the existing tree.
    """

    def extend_explore(self, node: Node):
        """
        Extend from selected node using the separation criteria.
        Given q', the samples from this node, and q, the existing nodes
        in the tree, solve a problem to maximize separation distance from any
        existing nodes from the tree:

        max_q' min_q dist(q',q)
        
        the distance metric here is evaluated based on the local distance
        metric.
        """
        xnext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)

        pairwise_distance = np.zeros((self.size, self.params.n_samples))

        for i in range(self.size):
            node = self.get_node_from_id(i)
            pairwise_distance[i, :] = node.eval_metric_batch(xnext_batch)

        # Evaluate inner minimum over q.
        sample_distance = np.min(pairwise_distance, axis=0)

        # Evaluate outer maximum over q'.
        idx = np.argmax(sample_distance)

        return IrsNode(xnext_batch[idx, :], self.params)

    def extend_contact(self, node: Node):
        """
        From the selected node, extend towards a contacting configuration.
        """
        raise NotImplementedError("not implemented yet.")

    def extend_towards_goal(self, node: Node):
        """
        Extend from the selected node using the goal criteria.
        Given q', the samples from this node, choose the one that is closest
        to the goal configuration using the global distance metric.
        """
        xnext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)
        diff = xnext_batch - self.goal[None, :]
        dist_batch = np.diagonal(
            diff @ np.diag(self.params.global_metric) @ diff.T)
        idx = np.argmin(dist_batch)
        return IrsNode(xnext_batch[idx, :], self.params)

    def extend_random(self, node: Node):
        """
        Extend from the selected node using the goal criteria.
        Given q', the samples from this node, choose the one that is closest
        to the goal configuration using the global distance metric.
        """
        xnext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)
        idx = np.random.randint(self.params.n_samples)
        return IrsNode(xnext_batch[idx, :], self.params)

    def extend(self, node: Node):
        mode = np.random.choice(
            list(self.params.extend_prob.keys()), 1,
            p=list(self.params.extend_prob.values()))

        if (mode == "explore"):
            selected_node = self.extend_explore(node)
        elif (mode == "random"):
            selected_node = self.extend_random(node)
        elif (mode == "towards_goal"):
            selected_node = self.extend_towards_goal(node)
        elif (mode == "contact"):
            selected_node = self.extend_contact(node)
        else:
            selected_node = self.extend_random(node)

        return selected_node

    def serialize(self):
        """
        Converts self.graph into a hashable object with necessary node
        attributes, such as q, and reachable set information.

        TODO: it seems possible to convert both the Node and Edge classes in
         rrt_base.py into dictionaries, which can be saved as attributes as
         part of a networkx node/edge. The computational methods in the
         Node class can be moved into a "reachable set" class.
         This conversion has a few benefits:
         - self.graph becomes hashable, rendering this function unnecessary,
         - accessing attributes becomes easier:
            self.graph[node_id]['node'].attribute_name  will become
            self.graph[node_id]['attribute_name']
         - saves a little bit of memory?
        """
        # TODO: include information about the system (q_sys yaml file?) in
        #  graph attributes.
        hashable_graph = nx.DiGraph()

        for node in self.graph.nodes:
            node_object = self.graph.nodes[node]["node"]
            node_attributes = dict(
                q=node_object.q,
                value=node_object.value,
                std_u=node_object.std_u,
                ubar=node_object.ubar,
                Bhat=node_object.Bhat,
                chat=node_object.chat,
                mu=node_object.mu,
                cov=node_object.cov,
                covinv=node_object.covinv)
            hashable_graph.add_node(node, **node_attributes)

        for edge in self.graph.edges:
            # edge is a 2-tuple of integer node indices.
            hashable_graph.add_edge(edge[0], edge[1],
                                    weight=self.graph.edges[edge]["edge"].cost)

        return hashable_graph

