from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.reachable_set import ReachableSet
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from qsim_cpp import GradientMode


class IrsRrtParams(RrtParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__()
        # Options for computing bundled dynamics.
        self.h = 0.1
        self.n_samples = 100
        self.q_model_path = q_model_path
        self.std_u = 0.1

        # kFirst and kExact are supported.
        self.bundle_mode = BundleMode.kFirst

        # State-space limits for sampling, provided as a bounding box.
        # During tree expansion, samples that go outside of this limit will be
        # rejected, implicitly enforcing the constraint.
        self.joint_limits = joint_limits

        # Regularization for computing inverse of covariance matrices.
        # NOTE(terry-suh): Note that if the covariance matrix is singular,
        # then the Mahalanobis distance metric is infinity. One interpretation
        # of the regularization term is to cap the infinity distance to some
        # value that scales with inverse of regularization.
        self.regularization = 1e-5

        # Stepsize.
        # TODO(terry-suh): the selection of this parameter should be automated.
        self.stepsize = 0.3
        

class IrsNode(Node):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """
    def __init__(self, q: np.array):
        super().__init__(q)
        # Bundled dynamics parameters.
        self.ubar = np.nan
        self.Bhat = np.nan
        self.chat = np.nan
        self.cov = np.nan
        self.covinv = np.nan
        self.mu = np.nan


class IrsEdge(Edge):
    """
    IrsEdge.
    """
    def __init__(self):
        super().__init__()
        self.du = np.nan
        self.u = np.nan
        # NOTE(terry-suh): It is possible to store trajectories in the edge
        # class. We won't do that here because we don't solve trajopt during
        # extend.


class IrsRrt(Rrt):
    def __init__(self, params: RrtParams):

        self.q_dynamics = QuasistaticDynamics(
            h = params.h,
            q_model_path = params.q_model_path,
            internal_viz=True)

        self.reachable_set = ReachableSet(self.q_dynamics, params)
        self.max_size = params.max_size

        self.x_lb, self.x_ub = self.joint_limit_to_x_bounds(
            params.joint_limits)

        # Initialize tensors for batch computation.
        self.Bhat_tensor = np.zeros((self.max_size,
            self.q_dynamics.dim_x, self.q_dynamics.dim_u))
        self.covinv_tensor = np.zeros((self.max_size,
            self.q_dynamics.dim_x, self.q_dynamics.dim_x))
        self.chat_matrix = np.zeros((self.max_size,
            self.q_dynamics.dim_x))

        super().__init__(params)

    def joint_limit_to_x_bounds(self, joint_limits):
        joint_limit_ub = {}
        joint_limit_lb = {}

        for model_idx in joint_limits.keys():
            joint_limit_lb[model_idx] = joint_limits[model_idx][:,0]
            joint_limit_ub[model_idx] = joint_limits[model_idx][:,1]

        x_lb = self.q_dynamics.get_x_from_q_dict(joint_limit_lb)
        x_ub = self.q_dynamics.get_x_from_q_dict(joint_limit_ub)
        return x_lb, x_ub

    def populate_node_parameters(self, node):
        """
        Given a node which has a q, this method populates the rest of the
        node parameters using reachable set computations.
        """
        node.ubar = node.q[self.q_dynamics.get_u_indices_into_x()]

        if self.params.bundle_mode == BundleMode.kExact:
            node.Bhat, node.chat = self.reachable_set.calc_exact_Bc(
                node.q, node.ubar)
        else:
            node.Bhat, node.chat = self.reachable_set.calc_bundled_Bc(
                node.q, node.ubar)
                
        node.cov, node.mu = self.reachable_set.calc_metric_parameters(
            node.Bhat, node.chat)
        node.covinv = np.linalg.inv(node.cov)

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
        self.populate_node_parameters(node)

        self.Bhat_tensor[node.id] = node.Bhat
        self.covinv_tensor[node.id] = node.covinv
        self.chat_matrix[node.id] = node.chat

    def sample_subgoal(self):
        """
        Sample a subgoal from the configuration space.
        """
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal
        return subgoal

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node, 
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(
            parent_node.Bhat, q - parent_node.chat, rcond=None)[0]

        # Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = parent_node.ubar + self.params.stepsize * du
        xnext = self.q_dynamics.dynamics(parent_node.q, ustar)
        cost = self.reachable_set.calc_node_metric(
            parent_node.covinv, parent_node.mu, xnext)

        child_node = IrsNode(xnext)

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge

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
