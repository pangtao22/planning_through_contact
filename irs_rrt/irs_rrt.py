import copy
import pickle

import numpy as np
from irs_mpc.irs_mpc_params import BundleMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.reachable_set import ReachableSet
from irs_rrt.rrt_base import Node, Edge, Rrt
from irs_rrt.rrt_params import IrsRrtParams


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
        # NOTE(terry-suh): It is possible to store trajectories in the edge
        # class. We won't do that here because we don't solve trajopt during
        # extend.


class IrsRrt(Rrt):
    def __init__(self, params: IrsRrtParams):
        self.params = params

        self.q_dynamics = QuasistaticDynamics(
            h=params.h,
            q_model_path=params.q_model_path,
            internal_viz=True)

        self.reachable_set = ReachableSet(self.q_dynamics, params)
        self.max_size = params.max_size

        self.x_lb, self.x_ub = self.joint_limit_to_x_bounds(
            params.joint_limits)

        # Initialize tensors for batch computation.
        self.Bhat_tensor = np.zeros((self.max_size,
                                     self.q_dynamics.dim_x,
                                     self.q_dynamics.dim_u))
        self.covinv_tensor = np.zeros((self.max_size,
                                       self.q_dynamics.dim_x,
                                       self.q_dynamics.dim_x))
        self.chat_matrix = np.zeros((self.max_size,
                                     self.q_dynamics.dim_x))

        super().__init__(params)

    def joint_limit_to_x_bounds(self, joint_limits):
        joint_limit_ub = {}
        joint_limit_lb = {}

        for model_idx in joint_limits.keys():
            joint_limit_lb[model_idx] = joint_limits[model_idx][:, 0]
            joint_limit_ub[model_idx] = joint_limits[model_idx][:, 1]

        x_lb = self.q_dynamics.get_x_from_q_dict(joint_limit_lb)
        x_ub = self.q_dynamics.get_x_from_q_dict(joint_limit_ub)
        return x_lb, x_ub

    def populate_node_parameters(self, node):
        """
        Given a node which has a q, this method populates the rest of the
        node parameters using reachable set computations.
        """
        node.ubar = node.q[self.q_dynamics.get_q_a_indices_into_x()]

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

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = self.reachable_set.calc_node_metric(
            parent_node.covinv, parent_node.mu, child_node.q)
        return cost

    def sample_subgoal(self):
        """
        Sample a subgoal from the configuration space.
        """
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal
        return subgoal

    def extend_towards_q(self, node: Node, q: np.array):
        """
        Extend towards a specified configuration q.
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(
            node.Bhat, q - node.chat, rcond=None)[0]

        # Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = node.ubar + self.params.stepsize * du
        xnext = self.q_dynamics.dynamics(node.q, ustar)
        return IrsNode(xnext)

    def calc_metric_batch(self, q_query):
        """
        Given q_query, return a np.array of \|q_query - q\|_{\Sigma}^{-1}_q,
        local distances from all the existing nodes in the tree to q_query.
        """
        mu_batch = self.get_valid_chat_matrix()  # B x n
        covinv_tensor = self.get_valid_covinv_tensor()  # B x n x n

        error_batch = q_query[None, :] - mu_batch  # B x n

        int_batch = np.einsum('Bij,Bi -> Bj', covinv_tensor, error_batch)
        metric_batch = np.einsum('Bi,Bi -> B', int_batch, error_batch)

        return metric_batch

    def save_tree(self, filename):
        """
        self.params.joint_limits are keyed by ModelInstanceIndex,
         which pickle does not like. Here we create a copy of self.params
         with a joint_limits dictionary keyed by model instance names.
        """
        picklable_params_dict = {key: copy.deepcopy(value)
                                 for key, value in self.params.__dict__.items()
                                 if key != "joint_limits"}
        hashable_joint_limits = {
            self.q_dynamics.plant.GetModelInstanceName(model):
                copy.deepcopy(value)
            for model, value in self.params.joint_limits.items()}
        picklable_params_dict['joint_limits'] = hashable_joint_limits

        picklable_params = IrsRrtParams(None, None)
        for key, value in picklable_params_dict.items():
            setattr(picklable_params, key, value)

        self.graph.graph['irs_rrt_params'] = picklable_params
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f)
