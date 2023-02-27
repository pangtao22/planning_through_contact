from typing import List, Tuple
import copy
import pickle
import os
import warnings

import networkx
import numpy as np
from pydrake.all import Quaternion, AngleAxis

from irs_rrt.reachable_set import ReachableSet
from irs_rrt.rrt_base import Node, Edge, Rrt
from irs_rrt.rrt_params import IrsRrtParams
from irs_mpc2.irs_mpc_params import (
    kNoSmoothingModes,
    k0RandomizedSmoothingModes,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
    kSmoothingMode2ForwardDynamicsModeMap,
)

from qsim.simulator import QuasistaticSimulator, InternalVisualizationType
from qsim_cpp import QuasistaticSimulatorCpp
from qsim.parser import QuasistaticParser
from qsim.model_paths import package_paths_dict


class IrsNode(Node):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """

    def __init__(self, q: np.array):
        super().__init__(q)
        # Bundled dynamics parameters.
        self.ubar = np.nan

        # For q_u and q_a.
        self.Bhat = np.nan
        self.chat = np.nan
        self.cov = np.nan
        self.covinv = np.nan
        self.mu = np.nan

        # For q_u only.
        self.Bhat_u = None
        self.chat_u = None
        self.cov_u = None
        self.covinv_u = None
        self.mu_u = None


class IrsEdge(Edge):
    """
    IrsEdge.
    """

    def __init__(self):
        super().__init__()
        self.du = np.nan
        self.u = np.nan
        self.trj = None


class IrsRrt(Rrt):
    def __init__(
        self,
        rrt_params: IrsRrtParams,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
    ):
        self.q_sim = q_sim
        self.plant = q_sim.get_plant()
        self.q_sim_py = q_sim_py
        # q_sim_py must have an internal MeshcatVisualizer.
        assert q_sim_py.internal_vis

        self.sim_params = copy.deepcopy(q_sim.get_sim_params())
        self.sim_params.calc_contact_forces = False
        self.sim_params.h = rrt_params.h
        self.sim_params.log_barrier_weight = (
            rrt_params.log_barrier_weight_for_bundling
        )
        self.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[
            rrt_params.smoothing_mode
        ]
        self.sim_params.use_free_solvers = rrt_params.use_free_solvers

        # TODO(pang): what does self.load_params() do?
        self.rrt_params = self.load_joint_limits_dict(rrt_params)
        self.reachable_set = ReachableSet(
            q_sim=q_sim, rrt_params=rrt_params, sim_params=self.sim_params
        )
        self.max_size = rrt_params.max_size

        self.dim_x = self.plant.num_positions()
        self.dim_u = self.q_sim.num_actuated_dofs()
        self.dim_q_u = self.dim_x - self.dim_u

        self.q_lb, self.q_ub = self.get_joint_limits()

        # Initialize tensors for batch computation.
        self.Bhat_tensor = np.zeros((self.max_size, self.dim_x, self.dim_u))
        self.covinv_tensor = np.zeros((self.max_size, self.dim_x, self.dim_x))
        self.chat_matrix = np.zeros((self.max_size, self.dim_x))

        self.covinv_u_tensor = np.zeros(
            (self.max_size, self.dim_q_u, self.dim_q_u)
        )

        self.q_u_indices_into_x = self.q_sim.get_q_u_indices_into_q()
        self.q_a_indices_into_x = self.q_sim.get_q_a_indices_into_q()

        self.calc_q_u_diff = self.get_calc_q_u_diff()

        super().__init__(rrt_params)

    def load_joint_limits_dict(self, params: IrsRrtParams) -> IrsRrtParams:
        """
        This function makes sure that joint limits are keyed by model
        instance indices.
        """
        for key in params.joint_limits.keys():
            break
        if isinstance(key, str):
            joint_limits_keyed_by_model_instance_index = {
                self.plant.GetModelInstanceByName(name): value
                for name, value in params.joint_limits.items()
            }
            params.joint_limits = joint_limits_keyed_by_model_instance_index

        return params

    def get_joint_limits(self, padding=0.05):
        """
        This function returns joint limits for the entire system as two
            vectors of length n_q.
        self.params.joint_limits contains joint limits for the un-actauted
            objects, which is used for sampling subgoals for objects.
        The robot joint limits are also used for sampling subgoals, but the
            sample is not used in distance computation.
        The robot joint limits are used when computing actions.
        """
        joint_limit_ub = {}
        joint_limit_lb = {}
        robot_joint_limits = self.q_sim.get_actuated_joint_limits()

        for model in self.q_sim.get_unactuated_models():
            joint_limit_lb[model] = self.rrt_params.joint_limits[model][:, 0]
            joint_limit_ub[model] = self.rrt_params.joint_limits[model][:, 1]

        for model, limits in robot_joint_limits.items():
            lower = limits["lower"]
            upper = limits["upper"]
            mid = (lower + upper) / 2
            range = (upper - lower) * (1 - padding)
            joint_limit_lb[model] = mid - range / 2
            joint_limit_ub[model] = mid + range / 2

        q_lb = self.q_sim.get_q_vec_from_dict(joint_limit_lb)
        q_ub = self.q_sim.get_q_vec_from_dict(joint_limit_ub)
        return q_lb, q_ub

    def populate_node_parameters(self, node: IrsNode):
        """
        Given a node which has a q, this method populates the rest of the
        node parameters using reachable set computations.
        """
        node.ubar = node.q[self.q_sim.get_q_a_indices_into_q()]

        # For q_u and q_a.
        if self.rrt_params.smoothing_mode in kNoSmoothingModes:
            Bhat, chat = self.reachable_set.calc_exact_Bc(node.q, node.ubar)
        elif self.rrt_params.smoothing_mode in k1RandomizedSmoothingModes:
            Bhat, chat = self.reachable_set.calc_bundled_Bc_randomized(
                node.q, node.ubar
            )
        elif self.rrt_params.smoothing_mode in kAnalyticSmoothingModes:
            Bhat, chat = self.reachable_set.calc_bundled_Bc_analytic(
                node.q, node.ubar
            )
        elif self.rrt_params.smoothing_mode == k0RandomizedSmoothingModes:
            Bhat, chat = self.reachable_set.calc_bundled_Bc_randomized_zero(
                node.q, node.ubar
            )
        else:
            raise NotImplementedError(
                f"{self.rrt_params.smoothing_mode} is not supported."
            )

        node.Bhat = Bhat
        node.chat = chat
        node.cov, node.mu = self.reachable_set.calc_metric_parameters(
            node.Bhat, node.chat
        )
        node.covinv = np.linalg.inv(node.cov)

        # For q_u only.
        node.Bhat_u = Bhat[self.q_u_indices_into_x, :]
        node.chat_u = chat[self.q_u_indices_into_x]
        (
            node.cov_u,
            node.mu_u,
        ) = self.reachable_set.calc_unactuated_metric_parameters(Bhat, chat)
        node.covinv_u = np.linalg.inv(node.cov_u)

    def get_Bhat_tensor_up_to(self, n_nodes: int):
        return self.Bhat_tensor[:n_nodes]

    def get_covinv_tensor_up_to(self, n_nodes: int, is_q_u_only: bool):
        if is_q_u_only:
            return self.covinv_u_tensor[:n_nodes]
        return self.covinv_tensor[:n_nodes]

    def get_chat_matrix_up_to(self, n_nodes: int, is_q_u_only: bool):
        if is_q_u_only:
            return self.chat_matrix[:n_nodes, self.q_u_indices_into_x]
        return self.chat_matrix[:n_nodes]

    def add_node(self, node: IrsNode, draw_node: bool = False):
        if draw_node:
            self.q_sim_py.update_mbp_positions_from_vector(node.q)
            self.q_sim_py.draw_current_configuration()
        self.populate_node_parameters(node)  # exception may be thrown here.

        super().add_node(node)
        # In addition to the add_node operation, we'll have to add the
        # B and c matrices of the node into our batch tensor.

        # Note we use self.size-1 here since the parent method increments
        # size by 1.

        self.Bhat_tensor[node.id] = node.Bhat
        self.covinv_tensor[node.id] = node.covinv
        self.chat_matrix[node.id] = node.chat
        self.covinv_u_tensor[node.id] = node.covinv_u

    def sample_subgoal(self):
        """
        Sample a subgoal from the configuration space.
        """
        subgoal = np.random.rand(self.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal
        return subgoal

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node,
        """
        # Compute least-squares solution.
        du = np.linalg.lstsq(
            parent_node.Bhat, q - parent_node.chat, rcond=None
        )[0]

        # Normalize least-squares solution.
        du = du / np.linalg.norm(du)
        ustar = parent_node.ubar + self.rrt_params.stepsize * du
        xnext = self.q_sim.calc_dynamics(parent_node.q, ustar, self.sim_params)
        cost = self.reachable_set.calc_node_metric(
            parent_node.covinv, parent_node.mu, xnext
        )

        child_node = IrsNode(xnext)
        child_node.subgoal = q

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.du = self.rrt_params.stepsize * du
        edge.u = ustar
        edge.cost = cost

        return child_node, edge

    def calc_distance_batch_local(
        self, q_query: np.ndarray, n_nodes: int, is_q_u_only: bool
    ):
        if is_q_u_only:
            q_query = q_query[self.q_u_indices_into_x]
        # B x n
        mu_batch = self.get_chat_matrix_up_to(n_nodes, is_q_u_only)
        # B x n x n
        covinv_tensor = self.get_covinv_tensor_up_to(n_nodes, is_q_u_only)
        error_batch = q_query - mu_batch
        int_batch = np.einsum("Bij,Bi -> Bj", covinv_tensor, error_batch)
        metric_batch = np.einsum("Bi,Bi -> B", int_batch, error_batch)

        return metric_batch

    def calc_pairwise_distance_batch_local(
        self, q_query_batch: np.ndarray, n_nodes: int, is_q_u_only: bool
    ):
        """
        q_query_batch consists is a (N x n) array  where N is the number of
        nodes to be queried for. THe returned array will be a (N x B) array
        where each element is the distance between the two nodes.
        """
        # N x n
        if is_q_u_only:
            q_query_batch = q_query_batch[:, self.q_u_indices_into_x]
        # B x n
        mu_batch = self.get_chat_matrix_up_to(n_nodes, is_q_u_only)
        # B x n x n
        covinv_tensor = self.get_covinv_tensor_up_to(n_nodes, is_q_u_only)

        # N x B x n
        error_batch = q_query_batch[:, None, :] - mu_batch[None, :, :]
        int_batch = np.einsum("Bij,NBi -> NBj", covinv_tensor, error_batch)
        metric_batch = np.einsum("NBi,NBi -> NB", int_batch, error_batch)
        return metric_batch

    def calc_distance_batch_global(
        self, q_query: np.ndarray, n_nodes: int, is_q_u_only: bool
    ):
        q_batch = self.get_q_matrix_up_to(n_nodes)

        if is_q_u_only:
            error_batch = (
                q_query[self.q_u_indices_into_x]
                - q_batch[:, self.q_u_indices_into_x]
            )
            metric_mat = np.diag(
                self.rrt_params.global_metric[self.q_u_indices_into_x]
            )
        else:
            error_batch = q_query - q_batch
            metric_mat = np.diag(self.rrt_params.global_metric)

        intsum = np.einsum("Bi,ij->Bj", error_batch, metric_mat)
        metric_batch = np.einsum("Bi,Bi->B", intsum, error_batch)

        return metric_batch

    def calc_distance_batch(
        self, q_query: np.ndarray, n_nodes=None, distance_metric=None
    ):
        """
        Given q_query, return a np.array of \|q_query - q\|_{\Sigma}^{-1}_q,
        local distances from all the existing nodes in the tree to q_query.

        This function computes the batch metric for the first n_nodes nodes
        in the tree. If n_nodes is None, the batch metric to all nodes in the
        tree is computed.

        Metric computation supports a 3 combinations of modes:
         - full_q OR un_actuated
         - full_tree OR up_to_n_nodes
         - global_metric OR local_metric
        """
        assert len(q_query) == self.dim_x

        if distance_metric is None:
            distance_metric = self.rrt_params.distance_metric

        if n_nodes is None:
            n_nodes = self.size

        if distance_metric == "global":
            return self.calc_distance_batch_global(
                q_query, n_nodes, is_q_u_only=False
            )
        elif distance_metric == "global_u":
            return self.calc_distance_batch_global(
                q_query, n_nodes, is_q_u_only=True
            )
        elif distance_metric == "local":
            return self.calc_distance_batch_local(
                q_query, n_nodes, is_q_u_only=False
            )
        elif distance_metric == "local_u":
            return self.calc_distance_batch_local(
                q_query, n_nodes, is_q_u_only=True
            )
        else:
            raise RuntimeError(
                f"distance metric {distance_metric} is not " f"supported."
            )

    @staticmethod
    def load_q_model_path(tree: networkx.DiGraph):
        if "model_file_package_name" in tree.graph:
            q_model_path = os.path.join(
                package_paths_dict[tree.graph["model_file_package_name"]],
                tree.graph["model_file_path_in_package"],
            )
            return q_model_path

        warnings.warn(
            "q_model_path is an absolute path and therefore "
            "may not work on other machines."
        )
        return tree.graph["irs_rrt_params"].q_model_path

    @staticmethod
    def make_from_pickled_tree(
        tree: networkx.DiGraph, internal_vis: InternalVisualizationType
    ):
        # Factory method for making an IrsRrt object from a pickled tree.
        q_model_path = IrsRrt.load_q_model_path(tree)
        parser = QuasistaticParser(q_model_path)

        rrt_param = tree.graph["irs_rrt_params"]
        rrt_param.q_model_path = q_model_path
        parser.set_sim_params(**tree.graph["q_sim_params"])

        prob_rrt = IrsRrt(
            rrt_params=rrt_param,
            q_sim=parser.make_simulator_cpp(),
            q_sim_py=parser.make_simulator_py(internal_vis=internal_vis),
        )
        prob_rrt.graph = tree
        prob_rrt.size = tree.number_of_nodes()

        for i_node in tree.nodes:
            node = tree.nodes[i_node]["node"]
            prob_rrt.Bhat_tensor[i_node] = node.Bhat
            prob_rrt.covinv_tensor[i_node] = node.covinv
            prob_rrt.chat_matrix[i_node] = node.chat
            prob_rrt.covinv_u_tensor[i_node] = node.covinv_u
            prob_rrt.q_matrix[i_node] = node.q

        return prob_rrt

    def save_tree(self, filename):
        """
        self.params.joint_limits are keyed by ModelInstanceIndex,
         which pickle does not like. Here we create a copy of self.params
         with a joint_limits dictionary keyed by model instance names.
        """
        found_package = False
        for name, path in package_paths_dict.items():
            if self.rrt_params.q_model_path.find(path) == -1:
                continue
            # found path at the beginning of q_model_path
            found_package = True
            self.graph.graph["model_file_package_name"] = name
            self.graph.graph[
                "model_file_path_in_package"
            ] = self.rrt_params.q_model_path[len(path) + 1 :]
            # + 1 to skip the '/' between package and model paths.

        if not found_package:
            raise RuntimeError(
                f"q_model_path {self.sim_params.q_model_path} "
                f"cannot be found in package_paths_dict."
            )

        picklable_params_dict = {
            key: copy.deepcopy(value)
            for key, value in vars(self.rrt_params).items()
            if key != "joint_limits"
        }
        model_name_to_joint_limits_map = {
            self.plant.GetModelInstanceName(model): copy.deepcopy(value)
            for model, value in self.rrt_params.joint_limits.items()
        }
        picklable_params_dict["joint_limits"] = model_name_to_joint_limits_map

        picklable_params = IrsRrtParams(None, None)
        for key, value in picklable_params_dict.items():
            setattr(picklable_params, key, value)

        # QuasistaticSimParams
        pickable_q_sim_params = {}
        for name in self.sim_params.__dir__():
            if name.startswith("_"):
                continue
            pickable_q_sim_params[name] = getattr(self.sim_params, name)

        self.graph.graph["irs_rrt_params"] = picklable_params
        self.graph.graph["goal_node_id"] = self.goal_node_idx
        self.graph.graph["q_sim_params"] = pickable_q_sim_params
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f)

    def get_u_knots_from_node_idx_path(self, node_idx_path: List[int]):
        n = len(node_idx_path)
        u_knots = np.zeros((n - 1, self.dim_u))
        for i in range(n - 1):
            id_node0 = node_idx_path[i]
            id_node1 = node_idx_path[i + 1]
            u_knots[i] = self.graph.edges[id_node0, id_node1]["edge"].u

        return u_knots

    @staticmethod
    def trim_regrasps(u_knots: np.ndarray):
        """
        @param u_knots: (T, dim_u).
        A regrasp in RRT has an associated action u consisitng of nans. When
         there are more than one consecutive nans in u_knots, we trim u_knots so
         that
         1. If there are more than one consecutive nans, only keep the last one.
         2. Remove all trailing nans.
        @return bool array of shape (T + 1,), entry t indicates whether the t-th
         entry in the original state path is kept. Note that there is 1 more
         entry in the state trajectory than in the action trajectory.
        """
        T = len(u_knots)
        node_idx_path_to_keep = np.ones(T + 1, dtype=bool)
        node_idx_path_to_keep[0] = True  # keep root
        for t in range(T):
            is_t_nan = any(np.isnan(u_knots[t]))
            if t == T - 1:
                is_t1_nan = True
            else:
                is_t1_nan = any(np.isnan(u_knots[t + 1]))

            if is_t_nan:
                if is_t1_nan:
                    node_idx_path_to_keep[t + 1] = False
                else:
                    node_idx_path_to_keep[t + 1] = True
            else:
                node_idx_path_to_keep[t + 1] = True

        return node_idx_path_to_keep

    def get_q_and_u_knots_to_goal(self):
        node_id_closest = self.find_node_closest_to_goal().id

        node_idx_path = self.trace_nodes_to_root_from(node_id_closest)
        node_idx_path = np.array(node_idx_path)

        q_knots = self.q_matrix[node_idx_path]
        u_knots = self.get_u_knots_from_node_idx_path(node_idx_path)

        return q_knots, u_knots

    def get_trimmed_q_and_u_knots_to_goal(self):
        """
        This function does three things:
            1. Finds the node closest to the goal according to the local
                distance metric,
            2. Traces a path from the node to the root of the tree.
            3. Trims the path of consecutive re-grasps.

        The returned q_knots_trimmed has shape (T + 1, dim_x),
         and u_knots_trimmed (T, dim_u).
        """
        q_knots, u_knots = self.get_q_and_u_knots_to_goal()

        node_idx_path_to_keep = self.trim_regrasps(u_knots)
        q_knots_trimmed = q_knots[node_idx_path_to_keep]
        u_knots_trimmed = u_knots[node_idx_path_to_keep[1:]]

        return q_knots_trimmed, u_knots_trimmed

    @staticmethod
    def get_regrasp_segments(u_knots_trimmed: np.ndarray):
        """
        @param u_knots_trimmed: (T, dim_u) is a 2D array punctuated by rows
         of nans, which indicate re-grasps.
        Returns a list of 2-tuples. Both integers in a tuple are indices into
         q_knots_trimmed. The first and second integer in each tuple are the
         indices of the first and last element in a segment of q.
        """
        T = len(u_knots_trimmed)
        segments = []
        t_start = 0
        for t in range(T):
            if np.isnan(u_knots_trimmed[t, 0]):
                segments.append((t_start, t))
                t_start = t + 1

            if t == T - 1:
                segments.append((t_start, t + 1))

        if segments[0][0] == 0 and segments[0][1] == 0:
            segments.pop(0)

        return segments

    @staticmethod
    def calc_q_u_diff_SE2(q_u_0, q_u_1):
        """
        q_u := [x, y, theta].
        """
        return abs(q_u_0[2] - q_u_1[2]), np.linalg.norm(q_u_0[:2] - q_u_1[:2])

    @staticmethod
    def calc_q_u_diff_SE3(q_u_0, q_u_1):
        """
        q_u_0 and q_u_1 are 7-vectors. The first 4 elements represent a
         quaternion and the last three a position.
        Returns (angle_diff_in_radians, position_diff_norm)
        """
        Q_U0 = Quaternion(q_u_0[:4])
        Q_U1 = Quaternion(q_u_1[:4])
        aa = AngleAxis(Q_U0.multiply(Q_U1.inverse()))

        return aa.angle(), np.linalg.norm(q_u_0[4:] - q_u_1[4:])

    @staticmethod
    def calc_q_u_diff_euclidean(q_u_0, q_u_1):
        """
        q_u_0 and q_u_1 are 7-vectors. The first 4 elements represent a
         quaternion and the last three a position.
        Returns (angle_diff_in_radians, position_diff_norm)
        """
        error = np.linalg.norm(q_u_0 - q_u_1)
        return error, error

    def get_calc_q_u_diff(self):
        # TODO(terry-suh): this  doesn't seem like the right if statement.
        # What if there are 3 dofs but they're in joint-space instead of
        # SE(2)? ditto for 7.
        if self.dim_q_u == 3:
            return self.calc_q_u_diff_SE2
        elif self.dim_q_u == 7:
            return self.calc_q_u_diff_SE3
        elif self.dim_q_u == 1:
            return lambda q_u0, q_u1: q_u0 - q_u1
        else:
            return self.calc_q_u_diff_euclidean

    def print_segments_displacements(
        self, q_knots_trimmed: np.ndarray, segments: List[Tuple[int, int]]
    ):
        """
        Prints the angular and translational displacements for each segment
        in segments.
        """
        print("======== Displacement for RRT trajectory segments ==========")
        for t_start, t_end in segments:
            q_u_start = q_knots_trimmed[t_start][self.q_u_indices_into_x]
            q_u_end = q_knots_trimmed[t_end][self.q_u_indices_into_x]

            angle_diff, position_diff = self.calc_q_u_diff(q_u_start, q_u_end)
            print("angle diff", angle_diff, "position diff", position_diff)

        print("============================================================")

    def trim_trajectory(
        self,
        q_trj: np.ndarray,
        angle_threshold: float = 1e-3,
        pos_threshold: float = 1e-3,
    ):
        q_u_final = q_trj[-1, self.q_u_indices_into_x]
        for t, q in enumerate(q_trj):
            q_u = q[self.q_u_indices_into_x]
            angle_diff, pos_diff = self.calc_q_u_diff(q_u, q_u_final)
            if angle_diff < angle_threshold and pos_diff < pos_threshold:
                break

        return t

    @staticmethod
    def concatenate_traj_list(q_trj_list: List[np.ndarray]):
        """
        Concatenates a list of trajectories into a single trajectory.
        q_trj_list[i] has shape (T_i, n_q)
        """
        q_trj_sizes = np.array([len(q_trj) for q_trj in q_trj_list])
        dim_q = q_trj_list[0].shape[1]
        q_trj_all = np.zeros((q_trj_sizes.sum(), dim_q))

        t_start = 0
        for q_trj_size, q_trj in zip(q_trj_sizes, q_trj_list):
            q_trj_all[t_start : t_start + q_trj_size] = q_trj
            t_start += q_trj_size

        return q_trj_all
