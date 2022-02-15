import copy
import pickle

import networkx
import numpy as np
from irs_mpc.irs_mpc_params import BundleMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.reachable_set import ReachableSet, ReachableSet3D
from irs_rrt.rrt_base import Node, Edge, Rrt
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel

from scipy.spatial.transform import Rotation as R

class IrsNode(Node):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """

    def __init__(self, q: np.array):
        super().__init__(q)
        # Bundled dynamics parameters.
        self.ubar = np.nan

        # For q_u and q_a.
        self.Bhat = np.nan
        self.chat = np.nan
        self.cov = np.nan
        self.covinv = np.nan
        self.mu = np.nan

        # For q_u only.
        self.Bhat_u = None
        self.chat_u = None
        self.cov_u = None
        self.covinv_u = None
        self.mu_u = None


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
    def __init__(self, params: IrsRrtParams):
        self.q_dynamics = QuasistaticDynamics(
            h=params.h,
            q_model_path=params.q_model_path,
            internal_viz=True)

        self.params = self.load_params(params)
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

        self.dim_q_u = self.q_dynamics.dim_x - self.q_dynamics.dim_u
        self.covinv_u_tensor = np.zeros((self.max_size,
                                         self.dim_q_u, self.dim_q_u))

        self.q_u_indices_into_x = self.q_dynamics.get_q_u_indices_into_x()

        super().__init__(params)

    @staticmethod
    def make_from_pickled_tree(tree: networkx.DiGraph):
        # Factory method for making an IrsRrt object from a pickled tree.
        irs_rrt_param = tree.graph['irs_rrt_params']
        irs_rrt = IrsRrt(irs_rrt_param)
        irs_rrt.graph = tree

        for i_node in tree.nodes:
            node = tree.nodes[i_node]["node"]
            irs_rrt.Bhat_tensor[i_node] = node.Bhat
            irs_rrt.covinv_tensor[i_node] = node.covinv
            irs_rrt.chat_matrix[i_node] = node.chat
            irs_rrt.covinv_u_tensor[i_node] = node.covinv_u
            irs_rrt.q_matrix[i_node] = node.q

        return irs_rrt

    def load_params(self, params: IrsRrtParams):
        for key in params.joint_limits.keys():
            break
        if isinstance(key, str):
            joint_limits_keyed_by_model_instance_index = {
                self.q_dynamics.plant.GetModelInstanceByName(name): value
                for name, value in params.joint_limits.items()}
            params.joint_limits = joint_limits_keyed_by_model_instance_index

        return params

    def joint_limit_to_x_bounds(self, joint_limits):
        joint_limit_ub = {}
        joint_limit_lb = {}

        for model_idx in joint_limits.keys():
            joint_limit_lb[model_idx] = joint_limits[model_idx][:, 0]
            joint_limit_ub[model_idx] = joint_limits[model_idx][:, 1]

        x_lb = self.q_dynamics.get_x_from_q_dict(joint_limit_lb)
        x_ub = self.q_dynamics.get_x_from_q_dict(joint_limit_ub)
        return x_lb, x_ub

    def populate_node_parameters(self, node: IrsNode):
        """
        Given a node which has a q, this method populates the rest of the
        node parameters using reachable set computations.
        """
        node.ubar = node.q[self.q_dynamics.get_q_a_indices_into_x()]

        # For q_u and q_a.
        if self.params.bundle_mode == BundleMode.kExact:
            Bhat, chat = self.reachable_set.calc_exact_Bc(
                node.q, node.ubar)
        else:
            Bhat, chat = self.reachable_set.calc_bundled_Bc(
                node.q, node.ubar)

        node.Bhat = Bhat
        node.chat = chat
        node.cov, node.mu = self.reachable_set.calc_metric_parameters(
            node.Bhat, node.chat)
        node.covinv = np.linalg.inv(node.cov)

        # For q_u only.
        node.Bhat_u = Bhat[self.q_u_indices_into_x, :]
        node.chat_u = chat[self.q_u_indices_into_x]
        node.cov_u, node.mu_u = (
            self.reachable_set.calc_unactuated_metric_parameters(Bhat, chat))
        node.covinv_u = np.linalg.inv(node.cov_u)

    def get_Bhat_tensor_up_to(self, n_nodes: int):
        return self.Bhat_tensor[:n_nodes]

    def get_covinv_tensor_up_to(self, n_nodes: int, is_q_u_only: bool):
        if is_q_u_only:
            return self.covinv_u_tensor[:n_nodes]
        return self.covinv_tensor[:n_nodes]

    def get_chat_matrix_up_to(self, n_nodes: int, is_q_u_only: bool):
        if is_q_u_only:
            return self.chat_matrix[:n_nodes, self.q_u_indices_into_x]
        return self.chat_matrix[:n_nodes]

    def add_node(self, node: IrsNode):
        super().add_node(node)
        # In addition to the add_node operation, we'll have to add the
        # B and c matrices of the node into our batch tensor.

        # Note we use self.size-1 here since the parent method increments
        # size by 1.
        self.populate_node_parameters(node)

        self.Bhat_tensor[node.id] = node.Bhat
        self.covinv_tensor[node.id] = node.covinv
        self.chat_matrix[node.id] = node.chat
        self.covinv_u_tensor[node.id] = node.covinv_u

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
        child_node.subgoal = q

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge

    def calc_metric_batch_local(self, q_query: np.ndarray, n_nodes: int,
                                is_q_u_only: bool):
        if is_q_u_only:
            q_query = q_query[self.q_u_indices_into_x]
        # B x n
        mu_batch = self.get_chat_matrix_up_to(n_nodes, is_q_u_only)
        # B x n x n
        covinv_tensor = self.get_covinv_tensor_up_to(n_nodes, is_q_u_only)
        error_batch = q_query - mu_batch
        int_batch = np.einsum('Bij,Bi -> Bj', covinv_tensor, error_batch)
        metric_batch = np.einsum('Bi,Bi -> B', int_batch, error_batch)

        return metric_batch

    def calc_metric_batch_global(self, q_query: np.ndarray, n_nodes: int,
                                 is_q_u_only: bool):
        q_batch = self.get_q_matrix_up_to(n_nodes)

        if is_q_u_only:
            error_batch = (q_query[self.q_u_indices_into_x]
                           - q_batch[:, self.q_u_indices_into_x])
            metric_mat = np.diag(
                self.params.global_metric[self.q_u_indices_into_x])
        else:
            error_batch = q_query - q_batch
            metric_mat = np.diag(self.params.global_metric)

        intsum = np.einsum('Bi,ij->Bj', error_batch, metric_mat)
        metric_batch = np.einsum('Bi,Bi->B', intsum, error_batch)

        return metric_batch

    def calc_metric_batch(self, q_query: np.ndarray, n_nodes=None,
                          distance_metric=None):
        """
        Given q_query, return a np.array of \|q_query - q\|_{\Sigma}^{-1}_q,
        local distances from all the existing nodes in the tree to q_query.

        This function computes the batch metric for the first n_nodes nodes
        in the tree. If n_nodes is None, the batch metric to all nodes in the
        tree is computed.

        Metric computation supports a 3 combinations of modes:
         - full_q OR un_actuated
         - full_tree OR up_to_n_nodes
         - global_metric OR local_metric
        """
        assert len(q_query) == self.q_dynamics.dim_x

        if distance_metric is None:
            distance_metric = self.params.distance_metric

        if n_nodes is None:
            n_nodes = self.size

        if distance_metric == "global":
            return self.calc_metric_batch_global(q_query, n_nodes,
                                                 is_q_u_only=False)
        elif distance_metric == "global_u":
            return self.calc_metric_batch_global(q_query, n_nodes,
                                                 is_q_u_only=True)
        elif distance_metric == "local":
            return self.calc_metric_batch_local(q_query, n_nodes,
                                                is_q_u_only=False)
        elif distance_metric == "local_u":
            return self.calc_metric_batch_local(q_query, n_nodes,
                                                is_q_u_only=True)
        else:
            raise RuntimeError(f"distance metric {distance_metric} is not "
                               f"supported.")

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


class IrsRrtGlobal3D(IrsRrt):
    def __init__(self, params: IrsRrtParams):
        super().__init__(params)
        self.qa_dim = self.q_dynamics.dim_u
        self.params.stepsize = 0.2
        q_dynamics_p = QuasistaticDynamicsParallel(self.q_dynamics)
        self.reachable_set = ReachableSet3D(self.q_dynamics,
            self.params, q_dynamics_p)

        # Global metric 
        assert (self.params.global_metric[self.qa_dim:self.qa_dim+4] == 0).all()
    
    def sample_subgoal(self):
        # Sample translation
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal

        # Sample quaternion uniformly following https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.128.8767&rep=rep1&type=pdf
        quat_xyzw = R.random().as_quat()
        subgoal[self.qa_dim:self.qa_dim+4] = np.concatenate((quat_xyzw[-1, None], quat_xyzw[:-1]))
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
        cost = self.compute_edge_cost(parent_node.q, xnext, parent_node.id, )

        child_node = IrsNode(xnext)

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge        

    def compute_edge_cost(self, parent_q: Node, child_q: Node, parent_id):
        error = parent_q - child_q

        if self.params.distance_metric == "global":
            metric_mat = np.diag(self.params.global_metric)
            cost = error @ metric_mat @ error

            parent_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(parent_q))
            child_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(child_q))
            quat_mul_diff = (child_quat * parent_quat.inv()).as_quat()
            cost += self.params.quat_metric * np.linalg.norm(quat_mul_diff[:-1])
        elif self.params.distance_metric == "global_u":
            error = error[self.q_u_indices_into_x]
            metric_mat = np.diag(
                self.params.global_metric[self.q_u_indices_into_x])
            cost = error @ metric_mat @ error

            parent_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(parent_q))
            child_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(child_q))
            quat_mul_diff = (child_quat * parent_quat.inv()).as_quat()
            cost += self.params.quat_metric * np.linalg.norm(quat_mul_diff[:-1])
        elif self.params.distance_metric == "local":
            covinv_tensor = self.get_covinv_tensor_up_to(self.max_size, False)
            metric_mat = covinv_tensor[parent_id]
            cost = error @ metric_mat @ error
        elif self.params.distance_metric == "local_u":
            error = error[self.q_u_indices_into_x]
            covinv_tensor = self.get_covinv_tensor_up_to(self.max_size, True)
            metric_mat = covinv_tensor[parent_id]
            cost = error @ metric_mat @ error

        return cost
    
    def calc_metric_batch_global(self, q_query, n_nodes: int,
                                 is_q_u_only: bool):
        """
        Given q_query, return a np.array of \|q_query - q\|.
        In the EuclidRrt implementation, the global distance metric is
        used as opposed to the local Mahalanobis metric.
        """

        q_batch = self.get_q_matrix_up_to(n_nodes)
        

        if is_q_u_only:
            error_batch = (q_query[self.q_u_indices_into_x]
                           - q_batch[:, self.q_u_indices_into_x])
            metric_mat = np.diag(
                self.params.global_metric[self.q_u_indices_into_x])
        else:
            error_batch = q_query[None,:] - q_batch
            metric_mat = np.diag(self.params.global_metric)

        intsum = np.einsum('Bi,ij->Bj', error_batch, metric_mat)
        metric_batch = np.einsum('Bi,Bi->B', intsum, error_batch)

        # scipy accepts (x, y, z, w)
        q_query_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(q_query))
        quat_batch = R.from_quat(self.convert_quat_wxyz_to_xyzw(q_batch, batch_mode=True))
        quat_mul_diff = (quat_batch * q_query_quat.inv()).as_quat()
        metric_batch += self.params.quat_metric * np.linalg.norm(quat_mul_diff[:, :-1], axis=1)

        return metric_batch

    def convert_quat_wxyz_to_xyzw(self, q, batch_mode=False):
        if batch_mode:
            return np.hstack((q[:, self.qa_dim + 1:self.qa_dim + 4], q[:, self.qa_dim, None]))
        else:
            return np.concatenate((q[self.qa_dim + 1:self.qa_dim + 4], q[self.qa_dim, None]))