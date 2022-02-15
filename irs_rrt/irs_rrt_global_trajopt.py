from typing import Dict
import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.reachable_set import ReachableSet
from irs_rrt.irs_rrt import IrsNode, IrsEdge
from irs_rrt.irs_rrt_global import IrsRrtGlobal
from irs_rrt.rrt_params import IrsRrtGlobalTrajoptParams
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode
from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from qsim_cpp import GradientMode

"""
This is a baseline class that uses the Eucliden norm for distance computation
as opposed to using the Mahalanobis metric.

For documentation of most functions, one should refer to IrsRrt.
"""

class IrsRrtGlobalTrajopt(IrsRrtGlobal):
    def __init__(self, params: IrsRrtGlobalTrajoptParams, contact_sampler,
        mpc_params):
        self.metric_mat = np.diag(params.global_metric)
        self.contact_sampler = contact_sampler
        self.mpc_params = mpc_params
        super().__init__(params)
        self.irs_mpc = IrsMpcQuasistatic(
            q_dynamics = self.q_dynamics,
            params = self.mpc_params)        

    def cointoss_for_grasp(self):
        sample_grasp = np.random.choice(
            [0, 1], 1, p=[
                1 - self.params.grasp_sampling_prob,
                self.params.grasp_sampling_prob])
        return sample_grasp

    def sample_grasp(self, q_u_goal):
        """
        Given a q_goal, sample a grasp using the contact sampler.
        """
        pinch_grasp = self.cointoss_for_grasp()
        if (pinch_grasp):
            q_dict = self.contact_sampler.sample_pinch_grasp(
                q_u_goal, 1000)[0]
        else:
            q_dict = self.contact_sampler.calc_enveloping_grasp(q_u_goal)

        return self.q_dynamics.get_x_from_q_dict(q_dict)

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node, 
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(
            parent_node.Bhat, q - parent_node.chat, rcond=None)[0]

        # 1. Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = self.params.stepsize * du
        xnext = parent_node.Bhat.dot(ustar) + parent_node.chat

        # 2. Compute grasp on the next goal configuration.
        q_goal = self.sample_grasp(
            xnext[self.q_dynamics.dim_u:self.q_dynamics.dim_x])

        # 3. Attempt trajopt to q_goal.
        T = self.mpc_params.T
        x_trj_d = np.tile(q_goal, (T + 1, 1))
        u_trj_0 = np.tile(parent_node.ubar, (T,1))
        self.irs_mpc.initialize_problem(
            parent_node.q, x_trj_d=x_trj_d, u_trj_0=u_trj_0)
        self.irs_mpc.iterate(self.params.num_iters)

        x_trj = self.irs_mpc.x_trj_best
        cost = self.irs_mpc.cost_best        
        x_reached = x_trj[T]

        print("SUBGOAL")
        print(q[-3:])
        print("GOAL")
        print(q_goal[-3:])
        print("REACHED")
        print(x_reached[-3:])

        child_node = IrsNode(x_reached)

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
