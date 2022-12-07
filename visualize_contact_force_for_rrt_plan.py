import copy
from typing import List
import os
import pickle
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import ContactResults, RigidTransform, Quaternion

from qsim_cpp import ForwardDynamicsMode, GradientMode

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from pydrake.systems.meshcat_visualizer import AddTriad

# %%
# load RRT tree.
pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__),
    "..",
    "examples",
    "allegro_hand",
    "tree_1000_analytic_0.pkl",
)

with open(pickled_tree_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = prob_rrt.q_dynamics
q_sim = q_dynamics.q_sim

q_vis = QuasistaticVisualizer(q_sim=q_sim, q_sim_py=q_dynamics.q_sim_py)

qu_goal = prob_rrt.rrt_params.goal[q_dynamics.get_q_u_indices_into_x()]
Q_WB_d = Quaternion(qu_goal[:4])
p_WB_d = qu_goal[4:]
dim_q = prob_rrt.dim_q
dim_u = dim_q - prob_rrt.dim_q_u


#%%
# load optimized trajectories
# TODO: consistency of simulation parameters between optimized and RRT
#  trajectories are not checked.
pickled_optimized_trajectories_path = os.path.join(
    os.path.dirname(irs_rrt.__file__),
    "..",
    "examples",
    "allegro_hand",
    "hand_optimized_q_and_u_trj.pkl",
)

with open(pickled_optimized_trajectories_path, "rb") as f:
    q_and_u_trj_optimized_dict = pickle.load(f)

q_trj_optimized_list = q_and_u_trj_optimized_dict["q_trj_list"]
u_trj_optimized_list = q_and_u_trj_optimized_dict["u_trj_list"]


#%% Allegro-specific
# visualize goal.
AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name="frame",
    prefix="drake/plant/sphere/sphere",
    length=0.1,
    radius=0.001,
    opacity=1,
)

AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name="frame",
    prefix="goal",
    length=0.1,
    radius=0.005,
    opacity=0.7,
)

q_dynamics.q_sim_py.viz.vis["goal"].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4()
)


# %% Compute q_knots again to get ContactResults, which are not saved in the
# tree (because they are hard to pickle...).
sim_params = copy.deepcopy(q_dynamics.q_sim_params_default)
sim_params.h = q_and_u_trj_optimized_dict["h_small"]
sim_params.gradient_mode = GradientMode.kNone

indices_q_a_into_q = q_sim.get_q_a_indices_into_q()
indices_q_u_into_q = q_sim.get_q_u_indices_into_q()

q_knots_computed_list = []
contact_results_list = []
for q_trj, u_trj in zip(q_trj_optimized_list, u_trj_optimized_list):
    q_knots_computed = np.zeros_like(q_trj)
    q_knots_computed[0] = q_trj[0]

    # calc contact results for the first state in q_trj.
    q_sim.calc_dynamics(q_trj[0], q_trj[0, indices_q_a_into_q], sim_params)
    contact_results_list.append(q_sim.get_contact_results_copy())

    T = len(u_trj)
    for t in range(T):
        q_knots_computed[t + 1] = q_dynamics.q_sim.calc_dynamics(
            q_knots_computed[t], u_trj[t], sim_params
        )
        contact_results_list.append(q_sim.get_contact_results_copy())

    print(
        "q_knots_norm_diff trimmed vs computed",
        np.linalg.norm(q_knots_computed - q_trj),
    )
    assert np.allclose(q_knots_computed, q_trj, atol=1e-4)


T = len(contact_results_list)
q_knots_all = np.zeros((T, dim_q))
t_start = 0
for q_trj in q_trj_optimized_list:
    n_knots = len(q_trj)
    q_knots_all[t_start : t_start + n_knots] = q_trj
    t_start += n_knots

#%%
cf_knots_map = q_vis.calc_contact_forces_knots_map(contact_results_list)
f_W_knot_norms = []
for cf_knots in cf_knots_map.values():
    f_W_knot_norms.extend(np.linalg.norm(cf_knots, axis=1))
f_W_knot_norms = np.array(f_W_knot_norms)
plt.hist(f_W_knot_norms[f_W_knot_norms > 0], bins=50)
plt.show()

#%%
x = np.linspace(0, 200, 100)
y = 1 - np.exp(-x / (np.percentile(f_W_knot_norms, 95) / 2.3))
plt.plot(x, y)
plt.show()

# %% video rendering
# assert False
# time.sleep(1 + T * h)
# frames_path_prefix = "/Users/pangtao/PycharmProjects/contact_videos"
frames_path_prefix = "/home/amazon/PycharmProjects/contact_videos"

folder_path_normal_color = os.path.join(
    frames_path_prefix, "allegro_rgba_0_normal_color"
)
q_vis.render_trajectory(
    x_traj_knots=q_knots_all,
    h=prob_rrt.rrt_params.h,
    folder_path=folder_path_normal_color,
    fps=120,
)

#
# folder_path = os.path.join(frames_path_prefix, "allegro_rgba_0")
# q_vis.render_trajectory(x_traj_knots=q_knots_all,
#                         h=0.1,
#                         folder_path=folder_path,
#                         fps=120,
#                         contact_results_list=contact_results_list,
#                         stride=1)


#%%
for i in range(len(q_trj_optimized_list) - 1):
    qu_last = q_trj_optimized_list[i][-1][indices_q_u_into_q]
    qu_first = q_trj_optimized_list[i + 1][0][indices_q_u_into_q]
    print(i, np.linalg.norm(qu_last - qu_first))
