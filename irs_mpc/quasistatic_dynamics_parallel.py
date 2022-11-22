import copy
import logging
from typing import Union

import numpy as np
import zmq
from qsim.simulator import QuasistaticSimulator, ForwardDynamicsMode
from qsim_cpp import GradientMode
from zmq_parallel_cmp.array_io import send_x_and_u, recv_bundled_AB

from .irs_mpc_params import (
    BundleMode,
    ParallelizationMode,
    IrsMpcQuasistaticParameters,
)
from .quasistatic_dynamics import QuasistaticDynamics
from .zmq_dynamics_worker import kTaskVentSocket, kTaskSinkSocket


class QuasistaticDynamicsParallel:
    """
    This class collects various implementations of ~parallelized~ bundled
    dynamics (A_bar and B_bar) computations, including one based on ZMQ and one
    based on MPI in C++.

    Other methods that benefit from parallelization, such as batch rollout of
    dynamics, are also included in this class.

    This class is constructed in the constructor of IrsMpcQuasistatic to provide
     bundled dynamics computation, but it can also be constructed independently,
     which can be useful for applications where direct access to the bundled
     dynamics is needed.
    """

    def __init__(
        self, q_dynamics: QuasistaticDynamics, use_zmq_workers: bool = False
    ):
        self.q_dynamics = q_dynamics
        self.q_sim_batch = q_dynamics.parser.make_batch_simulator()
        self.dim_x = q_dynamics.dim_x
        self.dim_u = q_dynamics.dim_u
        self.indices_u_into_x = q_dynamics.get_q_a_indices_into_x()
        self.q_sim_params = self.q_dynamics.q_sim_py.get_sim_parmas_copy()

        if use_zmq_workers:
            context = zmq.Context()

            # Socket to send messages on
            self.sender = context.socket(zmq.PUSH)
            self.sender.bind(f"tcp://*:{kTaskVentSocket}")

            # Socket to receive messages on
            self.receiver = context.socket(zmq.PULL)
            self.receiver.bind(f"tcp://*:{kTaskSinkSocket}")

            print(
                "Using ZMQ workers. This will hang if worker processes are "
                "not running"
            )

    def dynamics_batch_serial(self, x_batch: np.ndarray, u_batch: np.ndarray):
        """
        Computes x_next for every (x, u) pair in the input on a single thread.
        **This function should only be used for debugging.**
        -args:
            x_batch (n_batch, dim_x): batched state
            u_batch (n_batch, dim_u): batched input
        -returns
            x_next_batch (n_batch, dim_x): batched next state
        """
        n_batch = x_batch.shape[0]
        x_next_batch = np.zeros((n_batch, self.dim_x))

        for i in range(n_batch):
            x_next_batch[i] = self.q_dynamics.dynamics(x_batch[i], u_batch[i])
        return x_next_batch

    def dynamics_batch(self, x_batch: np.ndarray, u_batch: np.ndarray):
        self.q_sim_params.gradient_mode = GradientMode.kNone
        x_next, _, _ = self.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_sim_params
        )
        return x_next

    def dynamics_rollout_batch(
        self, x0_batch: np.ndarray, u_trj_batch: np.ndarray
    ):
        """
        Computes rollout of trajectories starting from x0_batch to every
        u_trj_batch.
        -args:
            x0_batch (n_batch, dim_x): batch of initial states.
            u_batch (n_batch, T, dim_u): batch of input trajectories.
        -returns:
            x_trj_batch (n_batch, T+1, dim_x): batch of resulting rollout
              trajectories.
        """

        n_batch = u_trj_batch.shape[0]
        T = u_trj_batch.shape[1]
        x_trj_batch = np.zeros((n_batch, T + 1, self.dim_x))
        x_trj_batch[:, 0, :] = x0_batch

        for t in range(T):
            x_trj_batch[:, t + 1, :] = self.dynamics_batch(
                x_trj_batch[:, t, :], u_trj_batch[:, t, :]
            )

        return x_trj_batch

    def dynamics_bundled_from_samples(self, x_nominal, u_batch):
        """
        Compute bundled dynamics given samples of u_batch.
        u_batch must be of shape (n_samples, dim_u).
        """
        n_samples = u_batch.shape[0]
        x_batch = np.tile(x_nominal[:, None], (1, n_samples)).transpose()
        xnext_batch = self.dynamics_batch(x_batch, u_batch)
        return np.mean(xnext_batch, axis=0)

    def dynamics_bundled(
        self,
        x_nominal: np.ndarray,
        u_nominal: np.ndarray,
        n_samples: int,
        std_u: Union[np.ndarray, float],
    ):
        """
        Compute bundled dynamics using dynamics_batch function.
        """
        u_batch = np.random.normal(
            u_nominal, std_u, size=[n_samples, self.dim_u]
        )
        return self.dynamics_bundled_from_samples(x_nominal, u_batch)

    def calc_bundled_ABc(
        self,
        x_trj: np.ndarray,
        u_trj: np.ndarray,
        irs_mpc_params: IrsMpcQuasistaticParameters,
        std_u: Union[np.ndarray, None] = None,
        log_barrier_weight: Union[float, None] = None,
    ):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj ((T + 1), dim_x)
            u_trj (T, dim_u)
            std_u: (dim_u,) or float
        """
        # unpack some parameters from irs_mpc_params.
        parallel_mode = irs_mpc_params.parallel_mode
        bundle_mode = irs_mpc_params.bundle_mode
        n_samples = irs_mpc_params.num_samples

        T = u_trj.shape[0]
        assert x_trj.shape[0] == T + 1

        # dispatch based on parallelization mode.
        if parallel_mode == ParallelizationMode.kNone:
            At, Bt = self.calc_bundled_AB_serial(
                x_trj, u_trj, std_u, n_samples, bundle_mode
            )
        elif parallel_mode == ParallelizationMode.kZmq:
            At, Bt = self.calc_bundled_AB_zmq(
                x_trj, u_trj, std_u, n_samples, bundle_mode
            )
        elif (
            parallel_mode == ParallelizationMode.kCppBundledB
            or parallel_mode == ParallelizationMode.kCppBundledBDirect
        ):
            """
            The CPP API only supports scalar standard deviation and
            computing bundled B from averaging gradients, Zero-order
            methods such as least-squared is not supported.
            """
            if bundle_mode == BundleMode.kFirstRandomized:
                is_direct = (
                    parallel_mode == ParallelizationMode.kCppBundledBDirect
                )
                At, Bt, = self.calc_bundled_AB_cpp(
                    x_trj, u_trj, std_u, n_samples, is_direct=is_direct
                )
            elif bundle_mode == BundleMode.kFirstAnalytic:
                At, Bt = self.calc_bundled_AB_cpp_analytic(
                    x_trj, u_trj, log_barrier_weight
                )
            else:
                raise NotImplementedError

        elif parallel_mode == ParallelizationMode.kCppDebug:
            """
            This mode exists because I'd like to compare the difference
             between sampling in python and sampling in C++. The conclusion
             so far is that there doesn't seem to be much difference.
            """
            assert bundle_mode == BundleMode.kFirstRandomized
            du_samples = np.random.normal(0, std_u, [T, n_samples, self.dim_u])
            At = np.zeros((T, self.dim_x, self.dim_x))
            Bt = self.calc_bundled_B_cpp_debug(x_trj, u_trj, du_samples)
        else:
            raise NotImplementedError(
                f"Parallel mode {parallel_mode} has not been implemented."
            )

        # Post processing.
        if irs_mpc_params.decouple_AB:
            At, Bt = self.decouple_AB(At, Bt)

        # compute ct
        ct = np.zeros((T, self.dim_x))
        for t in range(T):
            x_next_nominal = self.q_dynamics.dynamics(
                x_trj[t],
                u_trj[t],
                forward_mode=self.q_sim_params.forward_mode,
                gradient_mode=GradientMode.kNone,
            )
            ct[t] = x_next_nominal - At[t].dot(x_trj[t]) - Bt[t].dot(u_trj[t])

        return At, Bt, ct

    def decouple_AB(self, At: np.ndarray, Bt: np.ndarray):
        """
        Receives a list containing At and Bt matrices and decouples the
        off-diagonal entries corresponding to 0.0.
        """
        # At[:, self.indices_u_into_x, :] = 0.0
        Bt[:, self.indices_u_into_x, :] = np.eye(self.dim_u)
        At[:] = np.eye(At.shape[1])
        At[:, :, self.indices_u_into_x] = 0.0
        # At[:, :] = 0
        return At, Bt

    def initialize_AB(self, T: int):
        At = np.zeros((T, self.dim_x, self.dim_x))
        Bt = np.zeros((T, self.dim_x, self.dim_u))

        return At, Bt

    def calc_bundled_AB_serial(
        self,
        x_trj: np.ndarray,
        u_trj: np.ndarray,
        std_u: np.ndarray,
        n_samples: int,
        bundle_mode: BundleMode,
    ):
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)

        # Compute ABhat.
        ABhat_list = self.q_dynamics.calc_bundled_AB(
            x_trj[:-1, :],
            u_trj,
            n_samples=n_samples,
            std_u=std_u,
            bundle_mode=bundle_mode,
        )

        for t in range(T):
            At[t] = ABhat_list[t, :, : self.dim_x]
            Bt[t] = ABhat_list[t, :, self.dim_x :]

        return At, Bt

    def calc_bundled_AB_zmq(
        self,
        x_trj: np.ndarray,
        u_trj: np.ndarray,
        std_u: np.ndarray,
        n_samples: int,
        bundle_mode: BundleMode,
    ):
        """
        Get time varying linearized dynamics given a nominal trajectory,
         using worker processes launched separately.
        - args:
            x_trj (np.array, shape (T + 1) x n)
            u_trj (np.array, shape T x m)
        """
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)

        # send tasks.
        for t in range(T):
            x_u = np.zeros(self.dim_x + self.dim_u)
            x_u[: self.dim_x] = x_trj[t]
            x_u[self.dim_x :] = u_trj[t]
            send_x_and_u(
                socket=self.sender,
                x_u=x_u,
                t=t,
                n_samples=n_samples,
                std=std_u.tolist(),
                bundle_mode=bundle_mode,
            )

        # receive tasks.
        for _ in range(T):
            ABhat, t = recv_bundled_AB(self.receiver)
            At[t] = ABhat[:, :, : self.dim_x]
            Bt[t] = ABhat[:, :, self.dim_x :]

        return At, Bt

    def calc_bundled_AB_cpp(
        self,
        x_trj: np.ndarray,
        u_trj: np.ndarray,
        std_u: np.ndarray,
        n_samples: int,
        is_direct: bool,
    ):
        """
        Dispatches to BatchQuasistaticSimulator::CalcBundledBTrj or
            BatchQuasistaticSimulator::CalcBundledBTrjDirect, depending on
            is_direct. Right now the "direct" variant, based on Drake's MC
            simulation implementation, is the slowest.
        """
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)
        if is_direct:
            Bt[:] = self.q_sim_batch.calc_bundled_B_trj_direct(
                x_trj, u_trj, std_u, self.q_sim_params, n_samples, None
            )
        else:
            sim_params = copy.deepcopy(self.q_sim_params)
            sim_params.gradient_mode = GradientMode.kBOnly
            (A_trj, B_trj, c_trj) = self.q_sim_batch.calc_bundled_ABc_trj(
                x_trj, u_trj, std_u, sim_params, n_samples, None
            )
            Bt[:] = B_trj

        return At, Bt

    def calc_bundled_AB_cpp_analytic(
        self, x_trj: np.ndarray, u_trj: np.ndarray, log_barrier_weight: float
    ):
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)
        q_sim_params = QuasistaticSimulator.copy_sim_params(self.q_sim_params)
        q_sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidMp
        q_sim_params.gradient_mode = GradientMode.kBOnly
        q_sim_params.log_barrier_weight = log_barrier_weight
        (
            x_next_batch,
            A_Batch,
            B_batch,
            is_valid,
        ) = self.q_sim_batch.calc_dynamics_parallel(
            x_trj[:T], u_trj, q_sim_params
        )

        if not all(is_valid):
            raise RuntimeError("analytic bundling failed.")
        Bt[:] = B_batch

        return At, Bt

    def calc_bundled_B_serial_debug(
        self, x_trj: np.ndarray, u_trj: np.ndarray, du_samples: np.ndarray
    ):
        """
        du_samples: shape (T, n_samples, dim_u)
        """
        T = len(u_trj)
        Bt = np.zeros((T, self.dim_x, self.dim_u))
        n_samples = du_samples.shape[1]

        # Compute ABhat.
        for t in range(T):
            n_good_samples = 0
            for i in range(n_samples):
                try:
                    self.q_dynamics.dynamics(
                        x_trj[t],
                        u_trj[t] + du_samples[t, i],
                        gradient_mode=GradientMode.kBOnly,
                    )
                    Bt[t] += self.q_dynamics.q_sim.get_Dq_nextDqa_cmd()
                    n_good_samples += 1
                except RuntimeError as err:
                    logging.warning(err.__str__())

            Bt[t] /= n_good_samples

        return Bt

    def calc_bundled_B_cpp_debug(
        self, x_trj: np.ndarray, u_trj: np.ndarray, du_samples: np.ndarray
    ):
        """
        du_samples: shape (T, n_samples, dim_u)
        """
        T = len(u_trj)
        Bt = np.zeros((T, self.dim_x, self.dim_u))
        n_samples = du_samples.shape[1]

        x_batch = np.zeros((T, n_samples, self.dim_x))
        x_batch[...] = x_trj[:T, None, :]
        u_batch = u_trj[:, None, :] + du_samples

        x_batch_m = x_batch.view().reshape([T * n_samples, self.dim_x])
        u_batch_m = u_batch.view().reshape([T * n_samples, self.dim_u])
        sp = self.q_dynamics.q_sim.get_sim_params()
        sp.gradient_mode = GradientMode.kBOnly
        (
            x_next_batch_m,
            B_batch,
            is_valid_batch,
        ) = self.q_sim_batch.calc_dynamics_parallel(x_batch_m, u_batch_m, sp)

        # Shape of B_batch: (T * self.n_samples, self.dim_x, self.dim_u).
        B_batch = np.array(B_batch)
        B_batch.resize((T, n_samples, self.dim_x, self.dim_u))
        is_valid_batch = np.array(is_valid_batch)
        is_valid_batch.resize((T, n_samples))
        for t in range(T):
            n_valid_samples = is_valid_batch[t].sum()
            Bt[t] = B_batch[t, is_valid_batch[t]].sum(axis=0) / n_valid_samples

        return Bt
