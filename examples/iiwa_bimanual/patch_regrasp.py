import os
import pathlib
from calendar import c
import os

import numpy as np
import meshcat
import networkx as nx
import pickle
from tqdm import tqdm

from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw

from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode

from control.controller_system import ControllerParams

from irs_rrt.irs_rrt import IrsNode, IrsRrt
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar2 import (
    IiwaBimanualPlanarContactSampler,
)

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams
from collision_free_rrt import (
    find_collision_free_path,
    CollisionFreeRRT,
    step_out,
)

folder_path = str(pathlib.Path(__file__).parent.resolve())
pickled_tree_path = os.path.join(folder_path, "bimanual_planar.pkl")
qu_trj_path = os.path.join(folder_path, "bimanual_optimized_q_and_u_trj.pkl")

with open(pickled_tree_path, "rb") as f:
    tree = pickle.load(f)

with open(qu_trj_path, "rb") as f:
    trj_dict = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree, internal_vis=True)
q_sim = prob_rrt.q_sim
q_sim_py = prob_rrt.q_sim_py
q_vis = QuasistaticVisualizer(q_sim, q_sim_py)
dim_x = prob_rrt.dim_x
dim_u = prob_rrt.dim_u
idx_q_a = q_sim.get_q_a_indices_into_q()
idx_q_u = q_sim.get_q_u_indices_into_q()

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

q_knots_patched_list = []
u_knots_patched_list = []

q_knots_total = np.zeros((0, dim_x))
u_knots_total = np.zeros((0, dim_u))

n_segment = len(q_knots_ref_list)

for n in range(n_segment - 1):
    qa_start = q_knots_ref_list[n][-1, idx_q_a]
    qa_end = q_knots_ref_list[n + 1][0, idx_q_a]

    qu_start = q_knots_ref_list[n][-1, idx_q_u]
    qu_end = q_knots_ref_list[n + 1][0, idx_q_u]

    # q_dynamics.q_sim_py.update_mbp_positions_from_vector(q_knots_ref_list[n][-1])
    # q_dynamics.q_sim_py.draw_current_configuration()
    # input()
    # q_dynamics.q_sim_py.update_mbp_positions_from_vector(q_knots_ref_list[n+1][0])
    # q_dynamics.q_sim_py.draw_current_configuration()
    # input()

    ##
    qin = step_out(q_sim, q_sim_py, q_knots_ref_list[n][-1])
    qout = step_out(q_sim, q_sim_py, q_knots_ref_list[n + 1][0])

    qa_in = qin[idx_q_a]
    qa_out = qout[idx_q_a]
    ##

    cf_params = RrtParams()
    cf_params.goal = qa_out
    cf_params.root_node = Node(qa_in)
    cf_params.termination_tolerance = 1e-3
    cf_params.goal_as_subgoal_prob = 0.1
    cf_params.stepsize = 0.1
    cf_params.max_size = 20000

    cf_rrt = CollisionFreeRRT(prob_rrt, cf_params, qu_start)
    cf_rrt.q_lb = np.zeros(dim_x)
    cf_rrt.q_lb[idx_q_a] = (
        np.minimum(qa_start, qa_end) - 0.3
    )
    cf_rrt.q_lb[idx_q_a] = np.maximum(
        cf_rrt.q_lb[idx_q_a],
        prob_rrt.q_lb[idx_q_a],
    )

    cf_rrt.q_ub = np.zeros(dim_x)
    cf_rrt.q_ub[idx_q_a] = (
        np.maximum(qa_start, qa_end) + 0.3
    )
    cf_rrt.q_ub[idx_q_a] = np.minimum(
        cf_rrt.q_ub[idx_q_a],
        prob_rrt.q_ub[idx_q_a],
    )

    cf_rrt.iterate()
    cf_rrt_trj = cf_rrt.shortcut_path(cf_rrt.get_final_path_q())
    patch_trj = np.zeros((0, 9))

    patch_trj = np.vstack(
        (patch_trj, np.linspace(q_knots_ref_list[n][-1], qin, 20))
    )
    patch_trj = np.vstack((patch_trj, cf_rrt_trj))
    patch_trj = np.vstack(
        (patch_trj, np.linspace(qout, q_knots_ref_list[n + 1][0], 20))
    )

    print(patch_trj.shape)
    q_vis.publish_trajectory(patch_trj, 0.1)

    q_knots_total = np.vstack((q_knots_total, q_knots_ref_list[n]))
    q_knots_total = np.vstack((q_knots_total, patch_trj))

    q_knots_patched_list.append(q_knots_ref_list[n])
    q_knots_patched_list.append(patch_trj)

    u_knots_patched_list.append(u_knots_ref_list[n])
    u_knots_patched_list.append(
        patch_trj[1:, idx_q_a]
    )

q_knots_total = np.vstack((q_knots_total, q_knots_ref_list[-1]))
q_knots_patched_list.append(q_knots_ref_list[-1])
u_knots_patched_list.append(u_knots_ref_list[-1])
#%%
q_vis.publish_trajectory(q_knots_total, 0.1)

#%%
with open("bimanual_patched_q_and_u_trj.pkl", "wb") as f:
    pickle.dump(
        {
            "q_trj_list": q_knots_patched_list,
            "u_trj_list": u_knots_patched_list,
        },
        f,
    )
