from calendar import c
import os

import numpy as np
import meshcat
import networkx as nx
from tqdm import tqdm

from pydrake.all import (
    MultibodyPlant,
    RigidTransform,
    RollPitchYaw,
    JacobianWrtVariable,
)
from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode

from control.controller_system import ControllerParams
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar2 import (
    IiwaBimanualPlanarContactSampler,
)

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams


q_parser = QuasistaticParser(q_model_path_planar)


class CollisionFreeRRT(Rrt):
    def __init__(self, irs_rrt, rrt_params, qu):
        self.irs_rrt = irs_rrt
        self.q_sim_py = irs_rrt.q_sim_py
        self.q_sim = irs_rrt.q_sim
        self.plant = self.q_sim.get_plant()
        self.sg = self.q_sim.get_scene_graph()

        self.q_lb = irs_rrt.q_lb
        self.q_ub = irs_rrt.q_ub
        self.qu = qu

        self.ind_q_a = self.q_sim.get_q_a_indices_into_q()
        self.ind_q_u = self.q_sim.get_q_u_indices_into_q()
        self.dim_x = self.plant.num_positions()
        self.dim_u = self.q_sim.num_actuated_dofs()

        self.params = rrt_params
        super().__init__(rrt_params)

    def is_collision(self, x):
        """
        Checks if given configuration vector x is in collision.
        """
        q_a = x[self.ind_q_a]
        if np.linalg.norm(q_a - self.root_node.q) < 1e-6:
            return False
        self.q_sim_py.update_mbp_positions_from_vector(x)
        # self.q_sim_py.draw_current_configuration()

        plant = self.q_sim_py.get_plant()
        sg = self.q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(self.q_sim_py.context_sg)
        collision_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(0.02)
        )
        inspector = query_object.inspector()

        # 1. Compute closest distance pairs and normals.
        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            # left arm collision
            if body_A.model_instance() == 2:
                return True

        return False

    def sample_subgoal(self):
        while True:
            q = np.random.rand(self.dim_x)
            q = (self.q_ub - self.q_lb) * q + self.q_lb

            q_goal = np.zeros(self.dim_x)
            q_goal[self.ind_q_a] = q[self.ind_q_a]
            q_goal[self.ind_q_u] = self.qu

            if not (self.is_collision(q_goal)):
                return q_goal[self.ind_q_a]

    def calc_distance_batch(self, q_query: np.array):
        error_batch = q_query - self.get_q_matrix_up_to(self.size)
        metric_mat = np.diag(np.ones(self.dim_u))

        intsum = np.einsum("Bi,ij->Bj", error_batch, metric_mat)
        metric_batch = np.einsum("Bi,Bi->B", intsum, error_batch)

        return metric_batch

    def map_qa_to_q(self, qa):
        q = np.zeros(self.dim_x)
        q[self.ind_q_a] = qa
        q[self.ind_q_u] = self.qu
        return q

    def extend_towards_q(self, parent_node: Node, q: np.array):
        q_start = parent_node.q

        # Linearly interpolate with step size.
        distance = np.linalg.norm(q - q_start)
        direction = (q - q_start) / distance

        if distance < self.params.stepsize:
            xnext = q
        else:
            xnext = q_start + self.params.stepsize * direction

        collision = True
        if self.segment_has_no_collision(q_start, xnext, 10):
            collision = False

        child_node = Node(xnext)
        child_node.subgoal = q

        edge = Edge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 0.0

        q = np.zeros(self.dim_x)
        q[self.ind_q_u] = self.qu
        q[self.ind_q_a] = xnext
        # self.q_sim_py.update_mbp_positions_from_vector(q)
        # self.q_sim_py.draw_current_configuration()

        return child_node, edge, collision

    def iterate(self):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        while self.size < self.params.max_size:
            pbar.update(1)

            collision = True
            while collision:

                # 1. Sample a subgoal.
                if self.cointoss_for_goal():
                    subgoal = self.params.goal
                else:
                    subgoal = self.sample_subgoal()

                # 2. Sample closest node to subgoal
                parent_node = self.select_closest_node(subgoal)

                # 3. Extend to subgoal.
                child_node, edge, collision = self.extend_towards_q(
                    parent_node, subgoal
                )

            # 4. Attempt to rewire a candidate child node.
            if self.params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node
                )

            # 5. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_close_to_goal():
                print("done")
                self.goal_node_idx = child_node
                break

        pbar.close()

    def get_final_path_qa(self):
        # Find closest to the goal.
        q_final = self.select_closest_node(self.params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id
        )

        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_q))

        for i in range(path_T - 1):
            x_trj[i, :] = self.get_node_from_id(path[i]).q
        x_trj[path_T - 1, :] = self.get_node_from_id(path[path_T - 1]).q

        return x_trj

    def get_final_path_q(self):
        # Find closest to the goal.
        qa_final = self.select_closest_node(self.params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=qa_final.id
        )

        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_x))

        for i in range(path_T - 1):
            x_trj[
                i, self.ind_q_a
            ] = self.get_node_from_id(path[i]).q
            x_trj[i, self.ind_q_u] = self.qu
        x_trj[
            path_T - 1, self.ind_q_a
        ] = self.get_node_from_id(path[path_T - 1]).q
        x_trj[path_T - 1, self.ind_q_u] = self.qu

        return x_trj

    def interpolate_traj(self, q_start, q_end, T):
        return np.linspace(q_start, q_end, T)

    def segment_has_no_collision(self, q_start, q_end, T):
        q_trj = self.interpolate_traj(q_start, q_end, T)
        has_collision = False
        for t in range(q_trj.shape[0]):
            if self.is_collision(self.map_qa_to_q(q_trj[t])):
                has_collision = True
        return not has_collision

    def shortcut_path(self, x_trj, num_tries=100):
        x_trj_shortcut = np.copy(x_trj)
        T = x_trj_shortcut.shape[0]
        for _ in range(num_tries):
            # choose two random points on the path.
            ind_a, ind_b = np.sort(np.random.choice(T, 2, replace=False))

            x_a = x_trj_shortcut[
                ind_a, self.ind_q_a
            ]
            x_b = x_trj_shortcut[
                ind_b, self.ind_q_a
            ]

            if self.segment_has_no_collision(x_a, x_b, 100):
                x_trj_shortcut[
                    ind_a:ind_b, self.ind_q_a
                ] = self.interpolate_traj(x_a, x_b, ind_b - ind_a)

        return x_trj_shortcut

def step_out(q_sim, q_sim_py, x, scale=0.06, num_iters=3):
    """
    Given a near-contact configuration, give a trajectory that steps out.
    """
    q_sim_py.update_mbp_positions(q_sim.get_q_dict_from_vec(x))
    idx_qa = q_sim.get_q_a_indices_into_q()
    idx_qu = q_sim.get_q_u_indices_into_q()

    plant = q_sim_py.get_plant()
    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
        0.2
    )

    inspector = query_object.inspector()

    # 1. Compute closest distance pairs and normals.

    min_dist_left = np.inf
    min_dist_right = np.inf

    min_body_left = None
    min_body_right = None
    min_normal_left = None
    min_normal_right = None

    for collision in collision_pairs:
        f_id = inspector.GetFrameId(collision.id_A)
        body_A = plant.GetBodyFromFrameId(f_id)
        f_id = inspector.GetFrameId(collision.id_B)
        body_B = plant.GetBodyFromFrameId(f_id)

        # left arm collision
        if (body_A.model_instance() == 2) and (body_B.model_instance() == 3):
            # print("left: " + body_B.name())
            if collision.distance < min_dist_left:
                min_dist_left = collision.distance
                min_body_left = body_B
                min_normal_left = -collision.nhat_BA_W

        # right arm collision
        if (body_A.model_instance() == 2) and (body_B.model_instance() == 4):
            # print("right: " + body_B.name())
            if collision.distance < min_dist_right:
                min_dist_right = collision.distance
                min_body_right = body_B
                min_normal_right = -collision.nhat_BA_W

    # 2. Compute Jacobians and qdot.

    left_iiwa = plant.GetModelInstanceByName("iiwa_left")
    left_iiwa_base_frame = plant.GetFrameByName("iiwa_link_0", left_iiwa)

    J_L = plant.CalcJacobianTranslationalVelocity(
        q_sim_py.context_plant,
        JacobianWrtVariable.kV,
        min_body_left.body_frame(),
        np.array([0, 0, 0]),
        left_iiwa_base_frame,
        left_iiwa_base_frame,
    )

    J_La = J_L[:2, 3:6]

    right_iiwa = plant.GetModelInstanceByName("iiwa_right")
    right_iiwa_base_frame = plant.GetFrameByName("iiwa_link_0", right_iiwa)

    J_R = plant.CalcJacobianTranslationalVelocity(
        q_sim_py.context_plant,
        JacobianWrtVariable.kV,
        min_body_right.body_frame(),
        np.array([0, 0, 0]),
        right_iiwa_base_frame,
        right_iiwa_base_frame,
    )

    J_Ra = J_R[:2, 6:9]

    qdot_La = np.linalg.pinv(J_La).dot(scale * min_normal_left[:2])
    qdot_Ra = np.linalg.pinv(J_Ra).dot(scale * min_normal_right[:2])

    qdot = np.zeros(9)
    qdot[0:3] = np.zeros(3)
    qdot[3:6] = qdot_La
    qdot[6:9] = qdot_Ra
    qnext = x + qdot

    return qnext


def find_collision_free_path(irs_rrt, qa_start, qa_end, qu):
    params = RrtParams()
    params.goal = qa_end
    params.root_node = qa_start

    cf_rrt = CollisionFreeRRT(irs_rrt, params, qu)
    cf_rrt.iterate()


def is_collision(q_sim, x):
    """
    Checks if given configuration vector x is in collision.
    """
    q_sim.update_mbp_positions(q_sim.get_q_dict_from_vec(x))
    # self.q_sim_py.draw_current_configuration()

    sg = q_sim.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim.context_sg)
    collision_pairs = query_object.ComputePointPairPenetration()
    return len(collision_pairs) > 0


def generate_random_configuration(q_sim, qu, q_lb, q_ub):
    """
    Generate random configuration of qa.
    """
    ind_q_u = q_sim.get_q_u_indices_into_q()
    ind_q_a = q_sim.get_q_a_indices_into_q()
    while True:
        x = np.random.rand(q_sim.dim_x)
        x = (q_ub - q_lb) * x + q_lb
        q = np.zeros(q_sim.dim_x)
        q[ind_q_u] = qu
        q[ind_q_a] = x[ind_q_a]

        if not is_collision(q_sim, q):
            return q[ind_q_a]

def test_collision():
    h = 0.01

    q_parser = QuasistaticParser(q_model_path_planar)
    q_sim = q_parser.make_simulator_cpp()
    plant = q_sim.get_plant()

    idx_a_l = plant.GetModelInstanceByName(iiwa_l_name)
    idx_a_r = plant.GetModelInstanceByName(iiwa_r_name)
    idx_u = plant.GetModelInstanceByName(object_name)

    dim_x = plant.num_positions()
    dim_u = q_sim.num_actuated_dofs()
    dim_u_l = plant.num_positions(idx_a_l)
    dim_u_r = plant.num_positions(idx_a_r)

    contact_sampler = IiwaBimanualPlanarContactSampler(q_dynamics)

    # initial conditions.
    q_a0_r = [-0.7, -1.4, 0]
    q_a0_l = [0.7, 1.4, 0]
    q_u0 = np.array([0.65, 0, 0])
    q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
    x0 = q_sim.get_q_vec_from_dict(q0_dict)

    # input()

    joint_limits = {
        idx_u: np.array([[0.25, 0.75], [-0.3, 0.3], [-np.pi - 0.1, 0.1]])
    }

    q_u_goal = np.array([0.5, 0, -np.pi])

    params = IrsRrtProjectionParams(q_model_path_planar, joint_limits)
    params.smoothing_mode = BundleMode.kFirstAnalytic
    params.root_node = IrsNode(x0)
    params.max_size = 40000
    params.goal = np.copy(x0)
    params.goal[q_sim.get_q_u_indices_into_q()] = q_u_goal

    params.termination_tolerance = 0.01
    params.goal_as_subgoal_prob = 0.4
    params.global_metric = np.ones(x0.shape) * 0.1
    params.quat_metric = 5
    params.distance_threshold = np.inf
    std_u = 0.2 * np.ones(6)
    params.regularization = 1e-3
    # params.log_barrier_weight_for_bundling = 1000
    params.std_u = std_u
    params.stepsize = 0.02
    params.rewire = False
    params.distance_metric = "local_u"
    params.grasp_prob = 0.3
    params.h = 0.05

    prob_rrt = IrsRrtProjection(
        params,
        contact_sampler,
        q_sim,
    )

    # Choose some random configuration of u.
    qu = np.array([0.55, 0.0, 0.0])

    cf_params = RrtParams()
    root_q = generate_random_configuration(
        q_dynamics, qu, prob_rrt.q_lb, prob_rrt.q_ub
    )

    cf_params.root_node = Node(root_q)
    cf_params.goal = generate_random_configuration(
        q_dynamics, qu, prob_rrt.q_lb, prob_rrt.q_ub
    )
    cf_params.termination_tolerance = 1e-3
    cf_params.stepsize = 0.5
    cf_params.goal_as_subgoal_prob = 0.1
    cf_params.max_size = 10000
    cf_rrt = CollisionFreeRRT(prob_rrt, cf_params, qu)
    cf_rrt.iterate()

    final_traj = cf_rrt.get_final_path_q()

    q_dict_lst = []
    for t in range(final_traj.shape[0]):
        q_dict = q_dynamics.get_q_dict_from_x(final_traj[t, :])
        q_dict_lst.append(q_dict)

    q_dynamics.q_sim_py.animate_system_trajectory(0.01, q_dict_lst)
