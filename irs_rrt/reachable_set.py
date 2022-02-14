from typing import Dict
import numpy as np
import networkx as nx
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from pydrake.all import AngleAxis, Quaternion, RotationMatrix
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

class ReachableSet3D(ReachableSet):
    def __init__(self, q_dynamics: QuasistaticDynamics, params: IrsRrtParams,
                 q_dynamics_p: QuasistaticDynamicsParallel = None):
        super().__init__(q_dynamics, params, q_dynamics_p)
        # Notation: Throughout implementation in their child class, q stands
        # for quaternions and x stands for the state. Position is denoted 
        # using p.

        # Mask such that x[quat_mask] gives that wxyz quaternion 
        # representation. It is assumed that the last 7 elements correspond 
        # to states of the unactuated elements in the order of
        # [qw qx qy qz px py pz].
        self.quat_mask = np.zeros(self.q_dynamics.dim_x, dtype=np.bool)
        self.quat_mask[-7:-3] = True

        # When this tensor is multiplied with quaternion, it returns the E(q)
        # matrix used to convert rates to angular velocities in world frame.
        # omega = 2 * E(q) @ qdot = 2 * (E @ q) @ qdot
        # of shape 3 x 4 x 4, such that (E = 3 x 4 x 4) * (q = 4) = 
        # E(q) = (3 x 4).
        self.E_tensor = -np.array([
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0 ,1]],
            [[-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
            [[0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]],
            [[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]
        ]).transpose(1,0,2)

        # T_tensor used to go from angular velocities to rates.
        # qdot = 0.5 * T(q) * omega.
        self.T_tensor = self.E_tensor.transpose(1,0,2)

    def calc_qdot_from_omega(self, omega, q):
        """
        Given current q of shape (4,) in wxyz order, and an omega (3,)
        convert to qdot of shape (4,).
        """
        Tq = np.einsum('ijk,k->ij', self.T_tensor, q)
        qdot = 0.5 * np.einsum('ij,j->i', Tq, omega)
        return qdot

    def calc_qdot_from_omega_batch(self, omega_batch, q_batch):
        """
        Given current q_batch of shape (B,4) in wxyz order, and a batch of
        omegas (B,3), convert to batch of qdot of shape (B,4).
        """        
        Tq_batch = np.einsum('ijk,Bk->Bij', self.T_tensor, q_batch)
        qdot_batch = 0.5 * np.einsum('Bij,Bj->Bi', Tq_batch, omega_batch)
        return qdot_batch

    def calc_omega_from_qdot_batch(self, qdot, q):
        """
        Given current q of shape (4,) in wyxz order, and a qdot (4,)
        convert to omega of shape (3,)
        """
        Eq = np.einsum('ijk,k->ij', self.E_tensor, q)
        omega = 2.0 * np.einsum('ij,j->i', Eq, qdot)
        return omega

    def calc_omega_from_qdot_batch(self, qdot_batch, q_batch):
        """
        Given current q_batch of shape (B,4) in wxyz order, and a batch of
        qdots (B,4), convert to batch of omegas of shape (B,3).
        """        
        Eq_batch = np.einsum('ijk,Bk->Bij', self.E_tensor, q_batch)
        omega_batch = 2.0 * np.einsum('Bij,Bj->Bi', Eq_batch, qdot_batch)
        return omega_batch

    def calc_chat_given_batch_quat(self, qnext_batch, q):
        """
        Given a batch of quaternions (B,4) and the current quaternion (4,)
        compute the bundled dynamics chat of shape (4,).
        """
        B = qnext_batch.shape[0]
        q_batch = np.tile(q[None,:], (B,1))

        # 1. Compute dq = qdot * h
        dq_batch = qnext_batch - q_batch

        # 2. Convert dq to angular velocity and average.
        omega_batch = self.calc_omega_from_qdot_batch(
            dq_batch, q_batch)
        omega_hat = np.mean(omega_batch, axis=0)

        # 3. Use omega_hat to construct a rotation using the exponential
        # map.
        angle = np.linalg.norm(omega_hat)
        axis = omega_hat / angle
        omega_drake = AngleAxis(angle, axis)
        q_drake = Quaternion(q)

        qnext_drake = Quaternion(RotationMatrix(omega_drake).multiply(
            RotationMatrix(q_drake)).matrix())
        return qnext_drake.wxyz()

    def calc_Bhat_given_batch_quat(self, Bquat_batch, q):
        """
        Given a batch of dqdu (B,4,m) and current quatenrion (4,)
        compute the Bhat matrix (4,m).
        """
        # This function is isolated out so we can change it later.
        # Right now, it makes a locally linear approximation to the 
        # rotation dynamics and ignores the higher-order terms related to
        # taking the derivative of the matrix exponential.
        # We can revisit and resample if this becomes a problem.
        Bhat = np.mean(Bquat_batch, axis=0)
        return Bhat

    def calc_chat_given_batch(self, xnext_batch, x):
        """
        Given a batch of xnext, compute chat, the bundled dynamics.
        """
        quat_next_batch = xnext_batch[:,self.quat_mask]
        quat = x[self.quat_mask]
        chat_quat = self.calc_chat_given_batch_quat(quat_next_batch, quat)

        else_next_batch = xnext_batch[:,np.invert(self.quat_mask)]
        chat_else = np.mean(else_next_batch, axis=0)

        chat = np.zeros(self.q_dynamics.dim_x)
        chat[self.quat_mask] = chat_quat
        chat[np.invert(self.quat_mask)] = chat_else
        return chat

    def calc_Bhat_given_batch(self, B_batch, x):
        """
        Given a batch of xnext, compute chat, the bundled dynamics.
        """
        Bquat_batch = B_batch[:,self.quat_mask,:]
        quat = x[self.quat_mask]
        Bhat_quat = self.calc_Bhat_given_batch_quat(Bquat_batch, quat)

        else_B_batch = B_batch[:,np.invert(self.quat_mask),:]
        Bhat_else = np.mean(else_B_batch, axis=0)

        Bhat = np.zeros((self.q_dynamics.dim_x, self.q_dynamics.dim_u))
        Bhat[self.quat_mask] = Bhat_quat
        Bhat[np.invert(self.quat_mask)] = Bhat_else
        return Bhat

    def calc_bundled_Bc(self, x, ubar):
        """
        Compute bundled dynamics on Bc. 
        """
        x_batch = np.tile(x[None, :], (self.n_samples, 1))
        u_batch = np.random.normal(ubar, self.std_u, (
            self.params.n_samples, self.q_dynamics.dim_u))

        (x_next_batch, B_batch, is_valid_batch
         ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_dynamics.h, GradientMode.kBOnly)

        # print(np.sum(is_valid_batch))

        B_batch = np.array(B_batch)
        x_next_batch = x_next_batch[is_valid_batch]
        B_batch = B_batch[is_valid_batch]

        Bhat = self.calc_Bhat_given_batch(B_batch, x)
        chat = self.calc_chat_given_batch(x_next_batch, x)
        return Bhat, chat
