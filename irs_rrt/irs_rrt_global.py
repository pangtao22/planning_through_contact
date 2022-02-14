from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.reachable_set import ReachableSet3D
from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode, IrsEdge
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from scipy.spatial.transform import Rotation as R

"""
This is a baseline class that uses the Eucliden norm for distance computation
as opposed to using the Mahalanobis metric.

For documentation of most functions, one should refer to IrsRrt.
"""


class IrsRrtGlobal(IrsRrt):
    def __init__(self, params: IrsRrtParams):
        self.metric_mat = np.diag(params.global_metric)
        super().__init__(params)

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
        cost = self.compute_edge_cost(parent_node.q, xnext)

        child_node = IrsNode(xnext)

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge        

    def compute_edge_cost(self, parent_q: Node, child_q: Node):
        error = parent_q - child_q
        cost = error @ self.metric_mat @ error
        return cost

    def calc_metric_batch(self, q_query):
        """
        Given q_query, return a np.array of \|q_query - q\|.
        In the EuclidRrt implementation, the global distance metric is
        used as opposed to the local Mahalanobis metric.
        """

        q_batch = self.get_valid_q_matrix()
        error_batch = q_query[None,:] - q_batch

        intsum = np.einsum('Bi,ij->Bj', error_batch, self.metric_mat)
        metric_batch = np.einsum('Bi,Bi->B', intsum, error_batch)

        return metric_batch


class IrsRrtGlobal3D(IrsRrtGlobal):
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
        cost = self.compute_edge_cost(parent_node.q, xnext)

        child_node = IrsNode(xnext)

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge        

    def compute_edge_cost(self, parent_q: Node, child_q: Node):
        error = parent_q - child_q
        cost = error @ self.metric_mat @ error

        parent_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(parent_q))
        child_quat = R.from_quat(self.convert_quat_wxyz_to_xyzw(child_q))
        quat_mul_diff = (child_quat * parent_quat.inv()).as_quat()
        cost += self.params.quat_metric * np.linalg.norm(quat_mul_diff[:-1])
        return cost
    
    def calc_metric_batch(self, q_query):
        """
        Given q_query, return a np.array of \|q_query - q\|.
        In the EuclidRrt implementation, the global distance metric is
        used as opposed to the local Mahalanobis metric.
        """

        q_batch = self.get_valid_q_matrix()
        error_batch = q_query[None,:] - q_batch

        intsum = np.einsum('Bi,ij->Bj', error_batch, self.metric_mat)
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

