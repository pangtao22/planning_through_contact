from typing import Dict
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from tqdm import tqdm
import time

from irs_rrt.rrt_base import Node, Edge, Tree, TreeParams
from irs_mpc.irs_mpc_params import BundleMode, ParallelizationMode

"""
IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
and the Gaussian parametrized by the bundled dynamics.
"""
class IrsTreeParams(TreeParams):
    def __init__(self):
        super().__init__()
        self.q_dynamics = None # QuasistaticDynamics class.
        self.q_dynamics_p = None # QuasistaticDynamicsParallel class.

        # Options for computing bundled dynamics.
        self.n_samples = None
        self.std_u = None # std_u for input.
        self.decouple_AB = True
        self.bundle_mode = BundleMode.kFirst
        self.parallel_mode = ParallelizationMode.kZmq


class IrsNode(Node):
    def __init__(self, q: np.array, params: IrsTreeParams):
        super().__init__(q)
        # Bundled dynamics parameters.
        self.params = params
        self.q_dynamics = params.q_dynamics # QuasistaticDynamics class.        
        self.q_dynamics_p = params.q_dynamics_p # QuasistaticDynamics class.
        self.std_u = params.std_u # Injected noise for input.

        # NOTE(terry-suh): This should ideally use calc_Bundled_ABc from the
        # quasistatic dynamics class, but we are not doing this here because
        # the ct returned by calc_bundled_ABc works on exact dynamics.

        self.Bhat, self.chat = self.compute_bundled_Bc()

    def compute_bundled_Bc(self):
        # Compute bundled dynamics.
        ubar = self.q[self.q_dynamics.get_u_indices_into_x()]

        ## Sample xnext_batch.
        u_batch = np.random.normal(ubar, self.params.std_u, size=[
            self.params.n_samples, self.q_dynamics_p.dim_u])
        x_batch = np.tile(self.q[:,None], (1, self.params.n_samples)).transpose()

        xnext_batch = self.q_dynamics_p.dynamics_batch(x_batch, u_batch)

        # Do SVD.
        _, sigma, Vh = np.linalg.svd(xnext_batch - np.mean(xnext_batch, axis=0))

        cov = Vh.T @ np.diag(sigma ** 2) @ Vh
        np.set_printoptions(precision=3)
        print("cov")
        print(cov)



        fhat = self.q_dynamics_p.dynamics_bundled(
            self.q, ubar, self.params.n_samples, self.params.std_u)

        # Get the B matrix.
        x_trj = np.zeros((2, self.q_dynamics_p.dim_x))
        u_trj = np.zeros((1, self.q_dynamics_p.dim_u))

        x_trj[0,:] = self.q
        u_trj[0,:] = ubar

        Ahat, Bhat = self.q_dynamics_p.calc_bundled_AB_cpp(
            x_trj, u_trj, self.params.std_u, self.params.n_samples, False)
        chat = fhat

        return Bhat[0,:,:], chat

    def dynamics_Bc(self, du: np.array):
        # Evaluate effect of applying du on the bundle-linearized dynamics.
        # du: shape (dim_u,)
        xhat_next = self.Bhat.dot(du) + self.chat
        return xhat_next

    def dynamics_Bc_batch(self, du_batch: np.array):
        # Evaluate effect of applying du on the bundle-linearized dynamics.
        # du: shape (dim_b, dim_u)
        dim_b = du_batch.shape[0]
        xhat_next_batch = (
            self.Bhat.dot(du_batch.transpose()).transpose() + self.chat)
        return xhat_next_batch

    def eval_gaussian(self, q):
        pass

    def eval_gaussian_volume_scaled(self, q):
        pass

    def is_reachable_within_tol(self, q, tol: float):
        return True
"""
IrsEdge.
"""
class IrsEdge(Edge):
    def __init__(self):
        super().__init__()

class IrsTree(Tree):
    def __init__(self, params: TreeParams):
        super().__init__(params)
        self.q_dynamics = params.q_dynamics

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        raise NotImplementedError("This method is virtual.")

    def sample_node_from_tree(self):
        raise NotImplementedError("This method is virtual.")

    def extend(self, node: Node):
        raise NotImplementedError("This method is virtual.")

    def termination(self, node: Node):
        raise NotImplementedError("This method is virtual.")
