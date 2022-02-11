from typing import Dict
import numpy as np
import networkx as nx
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from qsim_cpp import GradientMode


class ReachableSet:
    """
    Computation class that computes parameters and metrics of reachable sets.
    """

    def __init__(self, q_dynamics: QuasistaticDynamics, params: IrsRrtParams,
                 q_dynamics_p: QuasistaticDynamicsParallel = None):
        self.q_dynamics = q_dynamics
        if q_dynamics_p is None:
            q_dynamics_p = QuasistaticDynamicsParallel(self.q_dynamics)
        self.q_dynamics_p = q_dynamics_p

        self.q_u_indices_into_x = self.q_dynamics.get_q_u_indices_into_x()

        self.params = params
        self.n_samples = self.params.n_samples
        self.std_u = self.params.std_u
        self.regularization = self.params.regularization

    def calc_exact_Bc(self, q, ubar):
        """
        Compute exact dynamics.
        """
        x = q[None, :]
        u = ubar[None, :]

        (x_next, B, is_valid
         ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x, u, self.q_dynamics.h, GradientMode.kBOnly)

        c = np.array(x_next).squeeze(0)
        B = np.array(B).squeeze(0)
        return B, c

    def calc_bundled_Bc(self, q, ubar):
        """
        Compute bundled dynamics on Bc. 
        """
        x_batch = np.tile(q[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(ubar, self.std_u, (
            self.params.n_samples, self.q_dynamics.dim_u))

        (x_next_batch, B_batch, is_valid_batch
         ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_dynamics.h, GradientMode.kBOnly)

        B_batch = np.array(B_batch)

        chat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.mean(B_batch[is_valid_batch], axis=0)
        return Bhat, chat

    def calc_metric_parameters(self, Bhat, chat):
        cov = Bhat @ Bhat.T + self.params.regularization * np.eye(
            self.q_dynamics.dim_x)
        mu = chat
        return cov, mu

    def calc_unactuated_metric_parameters(self, Bhat, chat):
        """
        Bhat: (n_a + n_u, n_a)
        """
        Bhat_u = Bhat[self.q_u_indices_into_x, :]
        cov_u = Bhat_u @ Bhat_u.T + self.params.regularization * np.eye(
            self.q_dynamics.dim_x - self.q_dynamics.dim_u)
        return cov_u, chat[self.q_u_indices_into_x]

    def calc_bundled_dynamics(self, Bhat, chat, du):
        xhat_next = Bhat.dot(du) + chat
        return xhat_next

    def calc_bundled_dynamics_batch(self, Bhat, chat, du_batch):
        xhat_next_batch = (
                Bhat.dot(du_batch.transpose()).transpose() + chat)
        return xhat_next_batch

    def calc_node_metric(self, covinv, mu, q_query):
        return (q_query - mu).T @ covinv @ (q_query - mu)

    def calc_node_metric_batch(self, covinv, mu, q_query_batch):
        batch_error = q_query_batch - mu[None, :]
        intsum = np.einsum('Bj,ij->Bi', batch_error, covinv)
        metric_batch = np.einsum('Bi,Bi->B', intsum, batch_error)
        return metric_batch
