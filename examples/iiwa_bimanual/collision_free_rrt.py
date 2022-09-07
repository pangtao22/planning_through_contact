from calendar import c
import os

import numpy as np
import meshcat
import networkx as nx
from tqdm import tqdm

from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir
from qsim_cpp import (ForwardDynamicsMode, GradientMode)

from control.controller_system import ControllerParams

from irs_mpc.irs_mpc_params import BundleMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar2 import (
    IiwaBimanualPlanarContactSampler)

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams


q_parser = QuasistaticParser(q_model_path_planar)

class CollisionFreeRRT(Rrt):
    def __init__(self, irs_rrt, rrt_params, qu):
        self.irs_rrt = irs_rrt
        self.q_dynamics = irs_rrt.q_dynamics
        self.q_sim_py = irs_rrt.q_dynamics.q_sim_py
        self.q_sim = irs_rrt.q_sim

        self.q_lb = irs_rrt.q_lb
        self.q_ub = irs_rrt.q_ub
        self.qu = qu

        self.params = rrt_params
        super().__init__(rrt_params)

    def is_collision(self, x):
        """
        Checks if given configuration vector x is in collision.
        """
        self.q_sim_py.update_mbp_positions_from_vector(x)
        #self.q_sim_py.draw_current_configuration()

        sg = self.q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(
            self.q_sim_py.context_sg)
        collision_pairs = query_object.ComputePointPairPenetration()

        return len(collision_pairs) > 0

    def sample_subgoal(self):
        while(True):
            q = np.random.rand(self.q_dynamics.dim_x)
            q = (self.q_ub - self.q_lb) * q + self.q_lb

            q_goal = np.zeros(self.q_dynamics.dim_x)
            q_goal[self.q_dynamics.get_q_a_indices_into_x()] = q[
                self.q_dynamics.get_q_a_indices_into_x()]
            q_goal[self.q_dynamics.get_q_u_indices_into_x()] = self.qu
            
            if not (self.is_collision(q_goal)):
                return q_goal[self.q_dynamics.get_q_a_indices_into_x()]

    def calc_distance_batch(self, q_query: np.array):
        error_batch = q_query - self.get_q_matrix_up_to(self.size)
        metric_mat = np.diag(np.ones(self.q_dynamics.dim_u))

        intsum = np.einsum('Bi,ij->Bj', error_batch, metric_mat)
        metric_batch = np.einsum('Bi,Bi->B', intsum, error_batch)

        return metric_batch

    def map_qa_to_q(self, qa):
        q = np.zeros(self.q_dynamics.dim_x)
        q[self.q_dynamics.get_q_a_indices_into_x()] = qa
        q[self.q_dynamics.get_q_u_indices_into_x()] = self.qu
        return q

    def extend_towards_q(self, parent_node: Node, q: np.array):
        q_start = parent_node.q

        # Linearly interpolate with step size.
        distance = np.linalg.norm(q - q_start)
        direction = (q - q_start) / distance

        if (distance < self.params.stepsize):
            xnext = q
        else:
            xnext = q_start + self.params.stepsize * direction

        collision = True
        if not self.is_collision(self.map_qa_to_q(xnext)):
            collision = False

        child_node = Node(xnext)
        child_node.subgoal = q

        edge = Edge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 0.0

        q = np.zeros(self.q_dynamics.dim_x)
        q[self.q_dynamics.get_q_u_indices_into_x()] = self.qu
        q[self.q_dynamics.get_q_a_indices_into_x()] = xnext
        #self.q_sim_py.update_mbp_positions_from_vector(q)
        #self.q_sim_py.draw_current_configuration()        

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
                    parent_node, subgoal)

            # 4. Attempt to rewire a candidate child node.
            if self.params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node)

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
            self.graph, source=self.root_node.id, target=q_final.id)

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
            self.graph, source=self.root_node.id, target=qa_final.id)

        path_T = len(path)

        x_trj = np.zeros((path_T, self.q_dynamics.dim_x))

        for i in range(path_T - 1):
            x_trj[i, self.q_dynamics.get_q_a_indices_into_x()] = (
                self.get_node_from_id(path[i]).q)
            x_trj[i, self.q_dynamics.get_q_u_indices_into_x()] = self.qu
        x_trj[path_T - 1, self.q_dynamics.get_q_a_indices_into_x()] = (
            self.get_node_from_id(path[path_T - 1]).q)
        x_trj[path_T - 1, self.q_dynamics.get_q_u_indices_into_x()] = self.qu

        return x_trj        

def find_collision_free_path(irs_rrt, qa_start, qa_end, qu):
    params = RrtParams()
    params.goal = qa_end
    params.root_node = qa_start

    cf_rrt = CollisionFreeRRT(irs_rrt, params, qu)
    cf_rrt.iterate()

def is_collision(q_dynamics, x):
    """
    Checks if given configuration vector x is in collision.
    """
    q_sim_py = q_dynamics.q_sim_py
    q_sim_py.update_mbp_positions_from_vector(x)
    #self.q_sim_py.draw_current_configuration()

    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputePointPairPenetration()
    return len(collision_pairs) > 0    

def generate_random_configuration(q_dynamics, qu, q_lb, q_ub):
    """
    Generate random configuration of qa.
    """
    while(True):
        x = np.random.rand(q_dynamics.dim_x)
        x = (q_ub - q_lb) * x + q_lb
        q = np.zeros(q_dynamics.dim_x)
        q[q_dynamics.get_q_u_indices_into_x()] = qu
        q[q_dynamics.get_q_a_indices_into_x()] = x[
            q_dynamics.get_q_a_indices_into_x()
        ]

        if not is_collision(q_dynamics, q):
            q_dynamics.q_sim_py.update_mbp_positions_from_vector(q)
            q_dynamics.q_sim_py.draw_current_configuration()
            input()
            return q[q_dynamics.get_q_a_indices_into_x()]

def test_collision():
    h = 0.01

    q_parser = QuasistaticParser(q_model_path_planar)
    q_dynamics = QuasistaticDynamics(h=h,
                                    q_model_path=q_model_path_planar,
                                    internal_viz=True)
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
        idx_u: np.array([
            [0.25, 0.75],
            [-0.3, 0.3],
            [-np.pi - 0.1, 0.1]])}

    q_u_goal = np.array([0.5, 0, -np.pi])

    params = IrsRrtProjectionParams(q_model_path_planar, joint_limits)
    params.bundle_mode = BundleMode.kFirstAnalytic
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
    params.distance_metric = 'local_u'
    params.grasp_prob = 0.3
    params.h = 0.05

    prob_rrt = IrsRrtProjection(params, contact_sampler)    

    # Choose some random configuration of u.
    qu = np.array([0.55, 0.0, 0.0])

    cf_params = RrtParams()
    root_q = generate_random_configuration(
        q_dynamics, qu, prob_rrt.q_lb, prob_rrt.q_ub)

    cf_params.root_node = Node(root_q)
    cf_params.goal = generate_random_configuration(
        q_dynamics, qu, prob_rrt.q_lb, prob_rrt.q_ub)        
    cf_params.termination_tolerance = 1e-3
    cf_params.stepsize = 0.5
    cf_params.goal_as_subgoal_prob = 0.1
    cf_params.max_size = 10000
    cf_rrt = CollisionFreeRRT(prob_rrt, cf_params, qu)
    cf_rrt.iterate()

    final_traj = cf_rrt.get_final_path_q()

    q_dict_lst = []
    for t in range(final_traj.shape[0]):
        q_dict = q_dynamics.get_q_dict_from_x(final_traj[t,:])
        q_dict_lst.append(q_dict)

    q_dynamics.q_sim_py.animate_system_trajectory(
        0.01, q_dict_lst)
