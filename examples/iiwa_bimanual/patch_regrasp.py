from calendar import c
import os

import numpy as np
import meshcat
import networkx as nx
import pickle
from tqdm import tqdm

from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir
from qsim_cpp import (ForwardDynamicsMode, GradientMode)

from control.controller_system import ControllerParams

from irs_mpc.irs_mpc_params import BundleMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.irs_rrt import IrsNode, IrsRrt
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar2 import (
    IiwaBimanualPlanarContactSampler)

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams
from collision_free_rrt import find_collision_free_path, CollisionFreeRRT

pickled_tree_path = "./bimanual_planar.pkl"
qu_trj_path = "./hand_optimized_q_and_u_trj.pkl"

with open(pickled_tree_path, 'rb') as f:
    tree = pickle.load(f)

with open(qu_trj_path, "rb") as f:
    trj_dict = pickle.load(f)    

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = prob_rrt.q_dynamics

q_knots_ref_list = trj_dict['q_trj_list']
u_knots_ref_list = trj_dict['u_trj_list']

q_knots_patched_list = []
u_knots_patched_list = []

q_knots_total = np.zeros((0, q_dynamics.dim_x))
u_knots_total = np.zeros((0, q_dynamics.dim_u))

n_segment = len(q_knots_ref_list)

for n in range(n_segment - 1):
    qa_start = q_knots_ref_list[n][-1, q_dynamics.get_q_a_indices_into_x()]
    qa_end = q_knots_ref_list[n + 1][0, q_dynamics.get_q_a_indices_into_x()]

    qu_start = q_knots_ref_list[n][-1, q_dynamics.get_q_u_indices_into_x()]
    qu_end = q_knots_ref_list[n + 1][0, q_dynamics.get_q_u_indices_into_x()]

    q_dynamics.q_sim_py.update_mbp_positions_from_vector(q_knots_ref_list[n][-1])
    q_dynamics.q_sim_py.draw_current_configuration()
    input()
    q_dynamics.q_sim_py.update_mbp_positions_from_vector(q_knots_ref_list[n+1][0])
    q_dynamics.q_sim_py.draw_current_configuration()
    input()

    cf_params = RrtParams()
    cf_params.goal = qa_end
    cf_params.root_node = Node(qa_start)
    cf_params.termination_tolerance = 1e-3
    cf_params.goal_as_subgoal_prob = 0.1
    cf_params.stepsize = 0.1
    cf_params.max_size = 20000

    cf_rrt = CollisionFreeRRT(prob_rrt, cf_params, qu_start)
    cf_rrt.q_lb = np.zeros(q_dynamics.dim_x)
    cf_rrt.q_lb[q_dynamics.get_q_a_indices_into_x()] = np.minimum(
        qa_start, qa_end) - 0.3
    cf_rrt.q_lb[q_dynamics.get_q_a_indices_into_x()] = np.maximum(
        cf_rrt.q_lb[q_dynamics.get_q_a_indices_into_x()],
        prob_rrt.q_lb[q_dynamics.get_q_a_indices_into_x()])

    cf_rrt.q_ub = np.zeros(q_dynamics.dim_x)
    cf_rrt.q_ub[q_dynamics.get_q_a_indices_into_x()] = np.maximum(
        qa_start, qa_end) + 0.3
    cf_rrt.q_ub[q_dynamics.get_q_a_indices_into_x()] = np.minimum(
        cf_rrt.q_ub[q_dynamics.get_q_a_indices_into_x()],
        prob_rrt.q_ub[q_dynamics.get_q_a_indices_into_x()])

    cf_rrt.iterate()
    patch_trj = cf_rrt.shortcut_path(cf_rrt.get_final_path_q())

    q_dict_lst = []
    for t in range(patch_trj.shape[0]):
        q_dict_lst.append(
            q_dynamics.get_q_dict_from_x(patch_trj[t]))
    q_dynamics.q_sim_py.animate_system_trajectory(0.1, q_dict_lst)
    input()

    q_knots_total = np.vstack((q_knots_total, q_knots_ref_list[n]))
    q_knots_total = np.vstack((q_knots_total, patch_trj))

    q_knots_patched_list.append(q_knots_ref_list[n])
    q_knots_patched_list.append(patch_trj[n])

    u_knots_patched_list.append(u_knots_ref_list[n])
    u_knots_patched_list.append(patch_trj[1:,
                                q_dynamics.get_q_a_indices_into_x()])

q_knots_total = np.vstack((q_knots_total, q_knots_ref_list[n_segment-1]))

q_dict_lst = []
for t in range(q_knots_total.shape[0]):
    q_dict_lst.append(
        q_dynamics.get_q_dict_from_x(q_knots_total[t])
    )

q_dynamics.q_sim_py.animate_system_trajectory(0.1, q_dict_lst)

#%%
with open("bimanual_patched_q_and_u_trj.pkl", 'wb') as f:
    pickle.dump({"q_trj_list": q_knots_patched_list,
                 "u_trj_list": u_knots_patched_list}, f)
