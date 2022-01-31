import os
from typing import Union

import numpy as np
import spdlog
import zmq
from qsim_cpp import GradientMode
from zmq_parallel_cmp.array_io import (send_x_and_u, recv_bundled_AB)

from .irs_mpc_params import (BundleMode, ParallelizationMode)
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

    def __init__(self,
                 q_dynamics: QuasistaticDynamics,
                 use_zmq_workers: bool = False):
        self.q_dynamics = q_dynamics
        self.q_sim_batch = q_dynamics.parser.make_batch_simulator()
        self.dim_x = q_dynamics.dim_x
        self.dim_u = q_dynamics.dim_u
        self.indices_u_into_x = q_dynamics.get_u_indices_into_x()

        # logger
        self.logger = self.q_dynamics.logger

        if use_zmq_workers:
            context = zmq.Context()

            # Socket to send messages on
            self.sender = context.socket(zmq.PUSH)
            self.sender.bind(f"tcp://*:{kTaskVentSocket}")

            # Socket to receive messages on
            self.receiver = context.socket(zmq.PULL)
            self.receiver.bind(f"tcp://*:{kTaskSinkSocket}")

            self.logger.info(
                "Using ZMQ workers. This will hang if worker processes are not "
                "running")

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
        x_next, _, _ = self.q_sim_batch.calc_dynamics_parallel(
            x_batch, u_batch, self.q_dynamics.h, GradientMode.kNone)
        return x_next

    def calc_bundled_ABc(self, x_trj: np.ndarray, u_trj: np.ndarray,
                         n_samples: int,
                         std_u: Union[np.ndarray, float], decouple_AB: bool,
                         bundle_mode: BundleMode,
                         parallel_mode: ParallelizationMode):
        """
        Get time varying linearized dynamics given a nominal trajectory.
        - args:
            x_trj ((T + 1), dim_x)
            u_trj (T, dim_u)
            std_u: (dim_u,) or float
        """
        T = u_trj.shape[0]
        assert x_trj.shape[0] == T + 1

        # dispatch based on parallelization mode.
        if parallel_mode == ParallelizationMode.kNone:
            At, Bt = self.calc_bundled_AB_serial(x_trj, u_trj, std_u,
                                                 n_samples, bundle_mode)
        elif parallel_mode == ParallelizationMode.kZmq:
            At, Bt = self.calc_bundled_AB_zmq(x_trj, u_trj, std_u,
                                              n_samples, bundle_mode)
        elif (parallel_mode == ParallelizationMode.kCppBundledB or
              parallel_mode == ParallelizationMode.kCppBundledBDirect):
            '''
            The CPP API only supports scalar standard deviation and
            computing bundled B from averaging gradients, Zero-order
            methods such as least-squared is not supported.
            '''
            assert np.allclose(std_u, std_u[0])
            assert bundle_mode == BundleMode.kFirst
            At, Bt, = self.calc_bundled_AB_cpp(
                x_trj, u_trj, std_u[0], n_samples,
                is_direct=parallel_mode == ParallelizationMode.kCppBundledBDirect)
        elif parallel_mode == ParallelizationMode.kCppDebug:
            '''
            This mode exists because I'd like to compare the difference 
             between sampling in python and sampling in C++. The conclusion 
             so far is that there doesn't seem to be much difference. 
            '''
            assert np.allclose(std_u, std_u[0])
            assert bundle_mode == BundleMode.kFirst
            du_samples = np.random.normal(0, std_u[0],
                                          [T, n_samples, self.dim_u])
            At = np.zeros((T, self.dim_x, self.dim_x))
            Bt = self.calc_bundled_B_cpp_debug(x_trj, u_trj, du_samples)
        else:
            raise NotImplementedError(
                f"Parallel mode {parallel_mode} has not been implemented.")

        # Post processing.
        if decouple_AB:
            At, Bt = self.decouple_AB(At, Bt)

        # compute ct
        ct = np.zeros((T, self.dim_x))
        for t in range(T):
            x_next_nominal = self.q_dynamics.dynamics(x_trj[t], u_trj[t])
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

    def calc_bundled_AB_serial(self, x_trj: np.ndarray, u_trj: np.ndarray,
                               std_u: np.ndarray, n_samples: int,
                               bundle_mode: BundleMode):
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)

        # Compute ABhat.
        ABhat_list = self.q_dynamics.calc_bundled_AB(
            x_trj[:-1, :], u_trj, n_samples=n_samples, std_u=std_u,
            bundle_mode=bundle_mode)

        for t in range(T):
            At[t] = ABhat_list[t, :, :self.dim_x]
            Bt[t] = ABhat_list[t, :, self.dim_x:]

        return At, Bt

    def calc_bundled_AB_zmq(self, x_trj: np.ndarray, u_trj: np.ndarray,
                            std_u: np.ndarray, n_samples: int,
                            bundle_mode: BundleMode):
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
            x_u[:self.dim_x] = x_trj[t]
            x_u[self.dim_x:] = u_trj[t]
            send_x_and_u(
                socket=self.sender,
                x_u=x_u,
                t=t,
                n_samples=n_samples, std=std_u.tolist(),
                bundle_mode=bundle_mode)

        # receive tasks.
        for _ in range(T):
            ABhat, t = recv_bundled_AB(self.receiver)
            At[t] = ABhat[:, :, :self.dim_x]
            Bt[t] = ABhat[:, :, self.dim_x:]

        return At, Bt

    def calc_bundled_AB_cpp(self, x_trj: np.ndarray, u_trj: np.ndarray,
                            std_u: float, n_samples: int,
                            is_direct: bool):
        """
        Dispatches to BatchQuasistaticSimulator::CalcBundledBTrj or
            BatchQuasistaticSimulator::CalcBundledBTrjDirect, depending on
            is_direct. Right now the "direct" variant, based on Drake's MC
            simulation implementation, is the slowest.
        """
        T = len(u_trj)
        At, Bt = self.initialize_AB(T)
        if is_direct:
            Bt = np.array(self.q_sim_batch.calc_bundled_B_trj_direct(
                x_trj, u_trj, self.q_dynamics.h, std_u,
                n_samples, None))
        else:
            Bt = np.array(self.q_sim_batch.calc_bundled_B_trj(
                x_trj, u_trj, self.q_dynamics.h, std_u,
                n_samples, None))

        return At, Bt

    def calc_bundled_B_serial_debug(self, x_trj: np.ndarray, u_trj: np.ndarray,
                                    du_samples: np.ndarray):
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
                        gradient_mode=GradientMode.kBOnly)
                    Bt[t] += self.q_dynamics.q_sim.get_Dq_nextDqa_cmd()
                    n_good_samples += 1
                except RuntimeError as err:
                    self.logger.warn(err.__str__())

            Bt[t] /= n_good_samples

        return Bt

    def calc_bundled_B_cpp_debug(self, x_trj: np.ndarray, u_trj: np.ndarray,
                                 du_samples: np.ndarray):
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
        (x_next_batch_m, B_batch, is_valid_batch
         ) = self.q_sim_batch.calc_dynamics_parallel(
            x_batch_m, u_batch_m, self.q_dynamics.h, GradientMode.kBOnly)

        # Shape of B_batch: (T * self.n_samples, self.dim_x, self.dim_u).
        B_batch = np.array(B_batch)
        B_batch.resize((T, n_samples, self.dim_x, self.dim_u))
        is_valid_batch = np.array(is_valid_batch)
        is_valid_batch.resize((T, n_samples))
        for t in range(T):
            n_valid_samples = is_valid_batch[t].sum()
            Bt[t] = B_batch[t, is_valid_batch[t]].sum(axis=0) / n_valid_samples

        return Bt
