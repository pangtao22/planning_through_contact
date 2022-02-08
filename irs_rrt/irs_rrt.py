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
from qsim_cpp import GradientMode


class IrsTreeParams(TreeParams):
    def __init__(self, q_dynamics, joint_limits):
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

        # State-space limits for sampling, provided as a bounding box.
        # During tree expansion, samples that go outside of this limit will be
        # rejected, implicitly enforcing the constraint.
        self.x_lb, self.x_ub = self.joint_limit_to_x_bounds(joint_limits)

        # Regularization for computing inverse of covariance matrices.
        # NOTE(terry-suh): Note that if the covariance matrix is singular,
        # then the Mahalanobis distance metric is infinity. One interpretation
        # of the regularization term is to cap the infinity distance to some
        # value that scales with inverse of regularization.
        self.regularization = 1e-5

        # Rewiring tolerance. If the distance from qi to qj is less than this
        # tolerance as evaluated by the local distance metric on qi, then
        # qi will be included in the candidate for rewiring.
        self.rewire_tolerance = 1e3

        # Termination tolerance. Algorithm will terminate if there exists a
        # node such that the norm between the node and the goal is less than
        # this threshold.
        self.termination_tolerance = 1.0

        # The global distance metric that compares qi and qj in configuration
        # space. This metric is not used for reachability criterion, but is
        # used for assigning costs to each edge of RRT, as well as termination
        # criteria. The metric should be a vector of dim_x, which will be 
        # diagonalized and put into matrix norm form.
        self.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 3.0])

        # If this mode is local, it uses
        self.metric_mode = "local"

        # Stepsize.
        # TODO(terry-suh): the selection of this parameter should be automated.
        self.stepsize = 0.3

        # If the "subgoal" strategy is selected, probability of setting the 
        # actual goal as the subgoal.
        self.subgoal_prob = 0.6

        # Probability to choose between different strategies for selection.
        # NOTE(terry-suh): if you choose subgoal as your strategy, you must
        # have 1.0 as subgoal probability here, and as 1.0 as extend
        # probability. This will be relaxed in later iterations.
        self.select_prob = {
            "explore": 0.0,
            "random": 0.0,
            "towards_goal": 0.0,
            "subgoal": 1.0
        }

        # Probability to choose between different strategies for extend.
        self.extend_prob = {
            "explore": 0.0,
            "random": 0.0,
            "contact": 0.0,
            "subgoal": 1.0,
            "towards_goal": 0.0
        }

    def joint_limit_to_x_bounds(self, joint_limits):
        joint_limit_ub = {}
        joint_limit_lb = {}

        for model_idx in joint_limits.keys():
            joint_limit_lb[model_idx] = joint_limits[model_idx][:,0]
            joint_limit_ub[model_idx] = joint_limits[model_idx][:,1]

        x_lb = self.q_dynamics.get_x_from_q_dict(joint_limit_lb)
        x_ub = self.q_dynamics.get_x_from_q_dict(joint_limit_ub)
        return x_lb, x_ub
        

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
        self.Bhat, self.chat = self.calc_bundled_Bc()

        # Parameters of the Gaussian based on computed Bhat / chat.
        self.mu, self.cov = self.compute_metric_parameters()
        self.covinv = np.linalg.inv(self.cov)

    def calc_bundled_Bc(self):
        """
        Compute bundled dynamics on Bc. 
        TODO(terry-suh): accept user input on BundleMode and 
        ParallelizationMode.
        """

        x_batch = np.tile(self.q[None,:], (self.params.n_samples,1))
        u_batch = np.random.normal(self.ubar, self.std_u, (
            self.params.n_samples, self.q_dynamics.dim_u))

        (x_next_batch, B_batch, is_valid_batch
        ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_dynamics.h, GradientMode.kBOnly
        )

        is_valid_batch = np.array(is_valid_batch)
        x_next_batch = np.array(x_next_batch)
        B_batch = np.array(B_batch)

        fhat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.mean(B_batch[is_valid_batch], axis=0)

        return Bhat, fhat

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

    def calc_bundled_dynamics(self, du: np.array):
        """
        Evaluate effect of applying du on the bundle-linearized dynamics.
        du: shape (dim_u,)
        """
        xhat_next = self.Bhat.dot(du) + self.chat
        return xhat_next

    def calc_bundled_dynamics_batch(self, du_batch: np.array):
        """
        Evaluate effect of applying du on the bundle-linearized dynamics.
        du: shape (dim_b, dim_u)
        """
        xhat_next_batch = (
                self.Bhat.dot(du_batch.transpose()).transpose() + self.chat)
        return xhat_next_batch

    def calc_metric(self, q_query):
        """
        Evaluate the distance between the q_query and self.q, as informed by
        the L2 norm on the basis that spans the bundled B matrix.
        This is also known as the Mahalanobis distance.
        """
        return (q_query - self.mu).T @ self.covinv @ (q_query - self.mu)

    def calc_metric_batch(self, q_query_batch):
        """
        Do calc_metric in batch.
        q_query_batch: (dim_b, dim_x)
        returns: (dim_b)
        """
        batch_error = q_query_batch - self.mu[None, :]

        intsum = np.einsum('Bj,ij->Bi', batch_error, self.covinv)
        metric_batch = np.einsum('Bi,Bi->B', intsum, batch_error)
        return metric_batch

    def get_sample_idx_within_joint_limit(self, x_batch):
        """
        Given x_batch of shape (N, n), return x_new_batch that obey joint
        limit constraints between self.params.x_lb and self.params.x_ub.
        """
        inidx = np.all(np.logical_and(self.params.x_lb <= x_batch,
                                      x_batch <= self.params.x_ub), axis=1)
        return inidx

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
        return xnext_batch, u_batch

    def sample_bundled_step(self, n_samples: int, step_size: float):
        """
        Given a number of samples and step_size, sample around the node
        according to bundled dynamic rollouts.
        """
        du_batch = np.random.rand(n_samples, self.q_dynamics.dim_u) - 0.5
        du_batch = du_batch / np.linalg.norm(du_batch, axis=1)[:, None]
        du_batch = step_size * du_batch
        u_batch = self.ubar + du_batch

        xnext_hat_batch = self.calc_bundled_dynamics_batch(du_batch)
        inidx = self.get_sample_idx_within_joint_limit(xnext_hat_batch)
        return xnext_hat_batch, u_batch

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
        dist = self.calc_metric(child_node.q)
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

        # Initialize tensors for batch computation.
        self.Bhat_tensor = np.zeros((self.max_size,
            self.q_dynamics.dim_x, self.q_dynamics.dim_u))
        self.covinv_tensor = np.zeros((self.max_size,
            self.q_dynamics.dim_x, self.q_dynamics.dim_x))
        self.chat_matrix = np.zeros((self.max_size,
            self.q_dynamics.dim_x))

        self.Bhat_tensor[0] = self.root_node.Bhat
        self.covinv_tensor[0] = self.root_node.covinv
        self.chat_matrix[0] = self.root_node.chat

    def get_valid_Bhat_tensor(self):
        return self.Bhat_tensor[:self.size]

    def get_valid_covinv_tensor(self):
        return self.covinv_tensor[:self.size]

    def get_valid_chat_matrix(self):
        return self.chat_matrix[:self.size]

    def add_node(self, node: Node):
        super().add_node(node)
        # In addition to the add_node operation, we'll have to add the
        # B and c matrices of the node into our batch tensor.

        # Note we use self.size-1 here since the parent method increments
        # size by 1.
        if (self.size > 1):
            self.Bhat_tensor[self.size-1,:] = node.Bhat
            self.covinv_tensor[self.size-1,:] = node.covinv
            self.chat_matrix[self.size-1,:] = node.chat

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        if (self.params.metric_mode == "global"):
            diff = parent_node.q - child_node.q
            cost = diff.T @ np.diag(self.params.global_metric) @ diff
        else:
            cost = parent_node.calc_metric(child_node.q)
        return cost 

    def is_close_to_goal(self):
        """
        Evaluate termination criteria for RRT using global distance metric.
        """
        # If the metric is global, terminate if there is a node in the tree
        # such that \|q - q_g\| \leq termination.
        if (self.params.metric_mode == "global"):
            diff = self.get_valid_q_matrix() - self.params.goal
            dist_batch = np.diagonal(
                diff @ np.diag(self.params.global_metric) @ diff.T)
            return np.min(dist_batch) < self.params.termination_tolerance
        # If the metric is global, terminate if there is a node in the tree
        # such that \|q_g - q\|_q \leq termination.
        else:
            dist_batch = self.calc_metric_batch(self.params.goal)
            return np.min(dist_batch) < self.params.termination_tolerance

    def calc_metric_batch(self, q_query):
        """
        Given q_query, return a np.array of \|q_query - q\|_{\Sigma}^{-1}_q,
        local distances from all the existing nodes in the tree to q_query.
        """

        mu_batch = self.get_valid_chat_matrix() # B x n
        covinv_tensor = self.get_valid_covinv_tensor() # B x n x n

        error_batch = q_query[None,:] - mu_batch # B x n

        int_batch = np.einsum('Bij,Bi -> Bj', covinv_tensor, error_batch)
        metric_batch = np.einsum('Bi,Bi -> B', int_batch, error_batch)

        return metric_batch

    def calc_metric_batch_query(self, q_query_batch):
        """
        Given q_query_batch of shape (N,n), 
        return a np.array of \|q_query - q\|_{\Sigma}^{-1}_q, a (N,B) pairwise
        local distances from all the existing nodes in the tree to q_query.
        """
        # q_query_batch is of shape (N,n)
        mu_batch = self.get_valid_chat_matrix() # (B,n)

        # (N,B,n)
        pairwise_error_batch = q_query_batch[:,None,:] - mu_batch[None,:,:]
        covinv_tensor = self.get_valid_covinv_tensor() # B x n x n

        pairwise_int_batch = np.einsum('NBi,Bij->NBj',
            pairwise_error_batch, covinv_tensor) # N x B x n
        pairwise_metric_batch = np.einsum('NBi,NBi->NB',
            pairwise_int_batch, pairwise_error_batch)

        return pairwise_metric_batch

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

    def select_node_subgoal(self):
        """
        Select a subgoal from a configuration space, and find the node that
        is closest from the subgoal.
        """
        # 1. Generate subgoal.
        sample_goal = np.random.choice(
            [0, 1], 1, p=[
                1 - self.params.subgoal_prob, self.params.subgoal_prob])
        
        if (sample_goal):
            self.subgoal = self.params.goal
        else:
            self.subgoal = np.random.rand(self.q_dynamics.dim_x)
            self.subgoal = self.params.x_lb + (
                self.params.x_ub - self.params.x_lb) * self.subgoal

        # 2. Find a node that is closest to the subgoal.
        metric_batch = self.calc_metric_batch(self.subgoal)
        selected_node = self.get_node_from_id(np.argmin(metric_batch))
    
        return selected_node

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
        elif (mode == "subgoal"):
            selected_node = self.select_node_subgoal()
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
        xnext_batch, unext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)

        batch_size = xnext_batch.shape[0]

        pairwise_distance = np.zeros((self.size, batch_size))

        for i in range(self.size):
            node_i = self.get_node_from_id(i)
            pairwise_distance[i, :] = node_i.calc_metric_batch(xnext_batch)

        # Evaluate inner minimum over q.
        sample_distance = np.min(pairwise_distance, axis=0)

        # Evaluate outer maximum over q'.
        idx = np.argmax(sample_distance)

        # Evaluate action and roll out dynamics. 
        ustar = unext_batch[idx, :]
        xnext = self.q_dynamics.dynamics(node.q, ustar)

        return IrsNode(xnext, self.params)

    def extend_contact(self, node: Node):
        """
        From the selected node, extend towards a contacting configuration.
        """
        raise NotImplementedError("not implemented yet.")

    def extend_subgoal(self, node: Node):
        """
        Extend towards a self.subgoal object. This is evaluated analytically.
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(node.Bhat, self.subgoal - node.q)[0]

        # Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = node.ubar + self.params.stepsize * du

        # Roll out dynamics and unregister subgoal.
        xnext = self.q_dynamics.dynamics(node.q, ustar)

        return IrsNode(xnext, self.params)

    def extend_towards_goal(self, node: Node):
        """
        Extend from the selected node using the goal criteria.
        Given q', the samples from this node, choose the one that is closest
        to the goal configuration using the global distance metric.
        """
        xnext_batch, unext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)
        diff = xnext_batch - self.goal[None, :]
        dist_batch = np.diagonal(
            diff @ np.diag(self.params.global_metric) @ diff.T)
        idx = np.argmin(dist_batch)

        # Evaluate action and roll out dynamics.
        ustar = unext_batch[idx, :]
        xnext = self.q_dynamics.dynamics(node.q, ustar)

        return IrsNode(xnext, self.params)

    def extend_random(self, node: Node):
        """
        Extend from the selected node using the goal criteria.
        Given q', the samples from this node, choose the one that is closest
        to the goal configuration using the global distance metric.
        """
        xnext_batch, unext_batch = node.sample_bundled_step(
            self.params.n_samples, self.params.stepsize)
        
        batch_size = xnext_batch.shape[0]
        idx = np.random.randint(batch_size)

        # Evaluate action and roll out dynamics.
        ustar = unext_batch[idx,:]
        xnext = self.q_dynamics.dynamics(node.q, ustar)
        
        return IrsNode(xnext, self.params)

    def extend(self, node: Node):
        mode = np.random.choice(
            list(self.params.extend_prob.keys()), 1,
            p=list(self.params.extend_prob.values()))[0]

        try:
            if (mode == "explore"):
                selected_node = self.extend_explore(node)
            elif (mode == "random"):
                selected_node = self.extend_random(node)
            elif (mode == "towards_goal"):
                selected_node = self.extend_towards_goal(node)
            elif (mode == "subgoal"):
                selected_node = self.extend_subgoal(node)
            elif (mode == "contact"):
                selected_node = self.extend_contact(node)
            else:
                selected_node = self.extend_random(node)
        except:
            selected_node = node

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
