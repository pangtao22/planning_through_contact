from typing import Dict
import copy

import numpy as np
import networkx as nx
from irs_rrt.rrt_params import IrsRrtParams
from pydrake.all import AngleAxis, Quaternion, RotationMatrix
from qsim.simulator import (
    QuasistaticSimulator,
    QuasistaticSimParameters,
    GradientMode,
    ForwardDynamicsMode,
)
from qsim.parser import QuasistaticParser
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc2.irs_mpc_params import (
    kSmoothingMode2ForwardDynamicsModeMap,
    kNoSmoothingModes,
    k0RandomizedSmoothingModes,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
)


class ReachableSet:
    """
    Computation class that computes parameters and metrics of reachable sets.
    """

    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp,
        rrt_params: IrsRrtParams,
        sim_params: QuasistaticSimParameters,
    ):
        self.q_sim = q_sim
        self.plant = q_sim.get_plant()

        self.sim_params = copy.deepcopy(sim_params)

        parser = QuasistaticParser(rrt_params.q_model_path)
        self.q_sim_batch = parser.make_batch_simulator()

        self.q_u_indices_into_x = self.q_sim.get_q_u_indices_into_q()
        self.rrt_params = rrt_params
        self.n_samples = self.rrt_params.n_samples
        self.std_u = self.rrt_params.std_u
        self.regularization = self.rrt_params.regularization

        self.dim_x = self.plant.num_positions()
        self.dim_u = self.q_sim.num_actuated_dofs()
        self.dim_q_u = self.dim_x - self.dim_u

    def calc_exact_Bc(self, q, ubar):
        """
        Compute exact dynamics.
        """
        assert self.rrt_params.smoothing_mode in kNoSmoothingModes
        self.sim_params.gradient_mode = GradientMode.kBOnly
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            self.rrt_params.smoothing_mode
        ]

        x = q[None, :]
        u = ubar[None, :]
        (
            x_next,
            A,
            B,
            is_valid,
        ) = self.q_sim_batch.calc_dynamics_parallel(x, u, self.sim_params)

        c = np.array(x_next).squeeze(0)
        B = np.array(B).squeeze(0)
        return B, c

    def calc_bundled_Bc_randomized(self, q, ubar):
        assert self.rrt_params.smoothing_mode in k1RandomizedSmoothingModes
        self.sim_params.gradient_mode = GradientMode.kBOnly
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            self.rrt_params.smoothing_mode
        ]

        x_batch = np.tile(q[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(
            ubar, self.std_u, (self.rrt_params.n_samples, self.dim_u)
        )

        (
            x_next_batch,
            A_batch,
            B_batch,
            is_valid_batch,
        ) = self.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.sim_params
        )

        if np.sum(is_valid_batch) == 0:
            raise RuntimeError("Cannot compute B and c hat for reachable sets.")

        B_batch = np.array(B_batch)
        x_next_batch = np.array(x_next_batch)

        chat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.mean(B_batch[is_valid_batch], axis=0)
        return Bhat, chat

    def calc_bundled_Bc_randomized_zero_numpy(self, q, ubar):
        assert self.rrt_params.smoothing_mode in k0RandomizedSmoothingModes
        self.sim_params.gradient_mode = GradientMode.kNone
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            self.rrt_params.smoothing_mode
        ]

        x_batch = np.tile(q[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(
            ubar, self.std_u, (self.rrt_params.n_samples, self.dim_u)
        )

        (
            x_next_batch,
            A_batch,
            B_batch,
            is_valid_batch,
        ) = self.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.sim_params
        )

        if np.sum(is_valid_batch) == 0:
            raise RuntimeError("Cannot compute B and c hat for reachable sets.")

        chat = np.mean(x_next_batch[is_valid_batch], axis=0)
        Bhat = np.linalg.lstsq(
            u_batch[is_valid_batch] - ubar,
            x_next_batch[is_valid_batch] - chat,
            rcond=None,
        )[0].transpose()

        return Bhat, chat

    def calc_bundled_Bc_randomized_zero(self, q, ubar):
        raise RuntimeError(
            "this method is buggy and should not be used "
            "without further investigation."
        )
        Bhat, chat = self.q_sim_batch.calc_Bc_lstsq(
            q, ubar, self.sim_params, self.std_u, self.rrt_params.n_samples
        )
        print(Bhat)
        return Bhat, chat

    def calc_bundled_Bc_analytic(self, q, ubar):
        assert self.rrt_params.smoothing_mode in kAnalyticSmoothingModes
        self.sim_params.gradient_mode = GradientMode.kBOnly
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            self.rrt_params.smoothing_mode
        ]
        q_next = self.q_sim.calc_dynamics(
            q=q, u=ubar, sim_params=self.sim_params
        )

        Bhat = self.q_sim.get_Dq_nextDqa_cmd()
        return Bhat, q_next

    def calc_metric_parameters(self, Bhat, chat):
        cov = Bhat @ Bhat.T + self.rrt_params.regularization * np.eye(
            self.dim_x
        )
        mu = chat
        return cov, mu

    def calc_unactuated_metric_parameters(self, Bhat, chat):
        """
        Bhat: (n_a + n_u, n_a)
        """
        Bhat_u = Bhat[self.q_u_indices_into_x, :]
        cov_u = Bhat_u @ Bhat_u.T + self.rrt_params.regularization * np.eye(
            self.dim_x - self.dim_u
        )

        return cov_u, chat[self.q_u_indices_into_x]

    def calc_bundled_dynamics(self, Bhat, chat, du):
        xhat_next = Bhat.dot(du) + chat
        return xhat_next

    def calc_bundled_dynamics_batch(self, Bhat, chat, du_batch):
        xhat_next_batch = Bhat.dot(du_batch.transpose()).transpose() + chat
        return xhat_next_batch

    def calc_node_metric(self, covinv, mu, q_query):
        return (q_query - mu).T @ covinv @ (q_query - mu)

    def calc_node_metric_batch(self, covinv, mu, q_query_batch):
        batch_error = q_query_batch - mu[None, :]
        intsum = np.einsum("Bj,ij->Bi", batch_error, covinv)
        metric_batch = np.einsum("Bi,Bi->B", intsum, batch_error)
        return metric_batch
