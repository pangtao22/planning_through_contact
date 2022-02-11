from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import pickle

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode, IrsEdge
from irs_rrt.irs_rrt_global import IrsRrtGlobalParams, IrsRrtGlobal
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from qsim_cpp import GradientMode

"""
This is a baseline class that uses the Eucliden norm for distance computation
as opposed to using the Mahalanobis metric.

For documentation of most functions, one should refer to IrsRrt.
"""

class IrsRrtRolloutParams(IrsRrtGlobalParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__(q_model_path, joint_limits)
        # Global distance metric for defining an adequate notion of distance.
        self.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 3.0])
        self.rollout_horizon = 3


class IrsRolloutEdge(IrsEdge):
    def __init__(self):
        super().__init__()
        self.u_trj = np.nan
        self.du_trj = np.nan


class IrsRrtRollout(IrsRrtGlobal):
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

        # Linearly inteprolate between ubar and ustar with specified 
        # rollout horizon.

        u_trj = np.linspace(
            parent_node.ubar, ustar, self.params.rollout_horizon)

        xnext_trj = self.q_dynamics.dynamics_rollout(parent_node.q, u_trj)
        xnext = xnext_trj[self.params.rollout_horizon,:]

        cost = self.compute_edge_cost(parent_node.q, xnext)

        child_node = IrsNode(xnext)

        edge = IrsRolloutEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.u_trj = u_trj
        edge.cost = cost

        return child_node, edge        

    def save_final_path(self, filename):
        # Find closest to the goal.
        q_final = self.select_closest_node(self.params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id)

        dim_u = self.q_dynamics.dim_u
        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_q))
        u_trj = np.zeros((path_T - 1, self.params.rollout_horizon, dim_u))

        for i in range(path_T - 1):
            x_trj[i,:] = self.get_node_from_id(path[i]).q
            u_trj[i,:,:] = self.get_edge_from_id(path[i], path[i+1]).u_trj
        x_trj[path_T-1,:] = self.get_node_from_id(path[path_T-1]).q

        path_dict = {
            "x_trj": x_trj, "u_trj": u_trj}

        with open(filename, 'wb') as f:
            pickle.dump(path_dict, f)
