import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from irs_rrt.reachable_set import ReachableSet
from irs_rrt.rrt_params import IrsRrtParams
from pydrake.all import AngleAxis, Quaternion, RotationMatrix
from qsim_cpp import GradientMode


class ReachableSet3D(ReachableSet):
    def __init__(
        self,
        q_dynamics: QuasistaticDynamics,
        rrt_params: IrsRrtParams,
        q_dynamics_p: QuasistaticDynamicsParallel = None,
    ):
        super().__init__(q_dynamics, rrt_params, q_dynamics_p)
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
        self.E_tensor = -np.array(
            [
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
                [[0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]],
                [[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
            ]
        ).transpose(1, 0, 2)

        # T_tensor used to go from angular velocities to rates.
        # qdot = 0.5 * T(q) * omega.
        self.T_tensor = self.E_tensor.transpose(1, 0, 2)

    def calc_qdot_from_omega(self, omega, q):
        """
        Given current q of shape (4,) in wxyz order, and an omega (3,)
        convert to qdot of shape (4,).
        """
        Tq = np.einsum("ijk,k->ij", self.T_tensor, q)
        qdot = 0.5 * np.einsum("ij,j->i", Tq, omega)
        return qdot

    def calc_qdot_from_omega_batch(self, omega_batch, q_batch):
        """
        Given current q_batch of shape (B,4) in wxyz order, and a batch of
        omegas (B,3), convert to batch of qdot of shape (B,4).
        """
        Tq_batch = np.einsum("ijk,Bk->Bij", self.T_tensor, q_batch)
        qdot_batch = 0.5 * np.einsum("Bij,Bj->Bi", Tq_batch, omega_batch)
        return qdot_batch

    def calc_omega_from_qdot_batch(self, qdot, q):
        """
        Given current q of shape (4,) in wyxz order, and a qdot (4,)
        convert to omega of shape (3,)
        """
        Eq = np.einsum("ijk,k->ij", self.E_tensor, q)
        omega = 2.0 * np.einsum("ij,j->i", Eq, qdot)
        return omega

    def calc_omega_from_qdot_batch(self, qdot_batch, q_batch):
        """
        Given current q_batch of shape (B,4) in wxyz order, and a batch of
        qdots (B,4), convert to batch of omegas of shape (B,3).
        """
        Eq_batch = np.einsum("ijk,Bk->Bij", self.E_tensor, q_batch)
        omega_batch = 2.0 * np.einsum("Bij,Bj->Bi", Eq_batch, qdot_batch)
        return omega_batch

    def calc_chat_given_batch_quat(self, qnext_batch, q):
        """
        Given a batch of quaternions (B,4) and the current quaternion (4,)
        compute the bundled dynamics chat of shape (4,).
        """
        B = qnext_batch.shape[0]
        q_batch = np.tile(q[None, :], (B, 1))

        # 1. Compute dq = qdot * h
        dq_batch = qnext_batch - q_batch

        # 2. Convert dq to angular velocity and average.
        omega_batch = self.calc_omega_from_qdot_batch(dq_batch, q_batch)
        omega_hat = np.mean(omega_batch, axis=0)

        # 3. Use omega_hat to construct a rotation using the exponential
        # map.
        angle = np.linalg.norm(omega_hat)
        axis = omega_hat / angle
        omega_drake = AngleAxis(angle, axis)
        q_drake = Quaternion(q)

        qnext_drake = Quaternion(
            RotationMatrix(omega_drake)
            .multiply(RotationMatrix(q_drake))
            .matrix()
        )
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
        quat_next_batch = xnext_batch[:, self.quat_mask]
        quat = x[self.quat_mask]
        chat_quat = self.calc_chat_given_batch_quat(quat_next_batch, quat)

        else_next_batch = xnext_batch[:, np.invert(self.quat_mask)]
        chat_else = np.mean(else_next_batch, axis=0)

        chat = np.zeros(self.q_dynamics.dim_x)
        chat[self.quat_mask] = chat_quat
        chat[np.invert(self.quat_mask)] = chat_else
        return chat

    def calc_Bhat_given_batch(self, B_batch, x):
        """
        Given a batch of xnext, compute chat, the bundled dynamics.
        """
        Bquat_batch = B_batch[:, self.quat_mask, :]
        quat = x[self.quat_mask]
        Bhat_quat = self.calc_Bhat_given_batch_quat(Bquat_batch, quat)

        else_B_batch = B_batch[:, np.invert(self.quat_mask), :]
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
        u_batch = np.random.normal(
            ubar, self.std_u, (self.rrt_params.n_samples, self.q_dynamics.dim_u)
        )

        (
            x_next_batch,
            B_batch,
            is_valid_batch,
        ) = self.q_dynamics_p.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_dynamics.h, GradientMode.kBOnly
        )

        B_batch = np.array(B_batch)
        x_next_batch = x_next_batch[is_valid_batch]
        B_batch = B_batch[is_valid_batch]

        Bhat = self.calc_Bhat_given_batch(B_batch, x)
        chat = self.calc_chat_given_batch(x_next_batch, x)
        return Bhat, chat
