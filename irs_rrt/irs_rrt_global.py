from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.reachable_set import ReachableSet
from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from qsim_cpp import GradientMode

"""
This is a baseline class that uses the Eucliden norm for distance computation
as opposed to using the Mahalanobis metric.

For documentation of most functions, one should refer to IrsRrt.
"""

class IrsRrtGlobalParams(IrsRrtParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__(q_model_path, joint_limits)
        # Global distance metric for defining an adequate notion of distance.
        self.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 3.0])


class IrsRrtGlobal(IrsRrt):
    def __init__(self, params: IrsRrtParams):
        self.metric_mat = np.diag(params.global_metric)
        super().__init__(params)

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        error = parent_node.q - child_node.q
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
