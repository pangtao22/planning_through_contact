import copy
from typing import List
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import ContactResults, RigidTransform, Quaternion

from qsim_cpp import ForwardDynamicsMode, GradientMode

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from pydrake.systems.meshcat_visualizer import AddTriad

# %%
# pickled_tree_path = os.path.join(
#     os.path.dirname(irs_rrt.__file__), '..',
#     'examples', 'planar_hand', "tree_2000_0.pkl")

pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__), '..',
    'examples', 'allegro_hand', "tree_1000_analytic_0.pkl")

# pickled_tree_path = os.path.join(
#     os.path.dirname(irs_rrt.__file__), '..',
#     'examples', 'allegro_hand', "tree_2000_randomized_0.pkl")

with open(pickled_tree_path, 'rb') as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = prob_rrt.q_dynamics
q_dynamics.update_default_sim_params(forward_mode=ForwardDynamicsMode.kSocpMp)

q_vis = QuasistaticVisualizer(q_sim=q_dynamics.q_sim,
                              q_sim_py=q_dynamics.q_sim_py)

# %%
qu_goal = prob_rrt.params.goal[q_dynamics.get_q_u_indices_into_x()]
Q_WB_d = Quaternion(qu_goal[:4])
p_WB_d = qu_goal[4:]
dim_q = prob_rrt.dim_q
dim_u = dim_q - prob_rrt.dim_q_u

#%% Allegro-specific
# visualize goal.
AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name='frame',
    prefix='drake/plant/sphere/sphere',
    length=0.1,
    radius=0.001,
    opacity=1)

AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name='frame',
    prefix='goal',
    length=0.1,
    radius=0.005,
    opacity=0.7)

q_dynamics.q_sim_py.viz.vis['goal'].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4())


# %%
# find closet point to goal.
node_id_closest = prob_rrt.find_node_closest_to_goal().id

node_idx_path = prob_rrt.trace_nodes_to_root_from(node_id_closest)
node_idx_path = np.array(node_idx_path)

q_knots = prob_rrt.q_matrix[node_idx_path]
u_knots = prob_rrt.get_u_knots_from_node_idx_path(node_idx_path)

node_idx_path_to_keep = prob_rrt.trim_regrasps(u_knots)
node_idx_path_trimmed = node_idx_path[node_idx_path_to_keep]
q_knots_trimmed = prob_rrt.q_matrix[node_idx_path_trimmed]
u_knots_trimmed = u_knots[node_idx_path_to_keep[1:]]

# Compute q_knots again to get ContactResults, which are not saved in the tree.
h = q_dynamics.q_sim_params_default.h
q_knots_computed = np.zeros_like(q_knots_trimmed)
q_knots_computed[0] = q_knots_trimmed[0]
contact_results_list = [ContactResults()]
t_knots_contact_results = [0]
T = len(u_knots_trimmed)

# TODO: this is hard coded, bad!
n_steps = 1
h_small = h / n_steps
sim_params = copy.deepcopy(q_dynamics.q_sim_params_default)
sim_params.h /= n_steps
sim_params.gradient_mode = GradientMode.kNone
sim_params.unactuated_mass_scale = 5

for t in range(T):
    u_t = u_knots_trimmed[t]

    if any(np.isnan(u_t)):
        q_knots_computed[t + 1] = q_knots_trimmed[t + 1]
        for i in range(n_steps):
            contact_results_list.append(ContactResults())
            t_knots_contact_results.append(
                t_knots_contact_results[-1] + h_small)
    else:
        q = q_knots_trimmed[t]
        for i in range(n_steps):
            q = q_dynamics.q_sim.calc_dynamics(q, u_t, sim_params)
            contact_results_list.append(
                q_dynamics.q_sim.get_contact_results_copy())
            t_knots_contact_results.append(
                t_knots_contact_results[-1] + h_small)
        q_knots_computed[t + 1] = q


print("q_knots_norm_diff trimmed vs computed",
      np.linalg.norm(q_knots_trimmed - q_knots_computed))
assert np.allclose(q_knots_trimmed, q_knots_computed, atol=1e-4)
q_vis.publish_trajectory(q_knots_trimmed, prob_rrt.params.h)



#%%
cf_knots_map = q_vis.calc_contact_forces_knots_map(
    contact_results_list)
f_W_knot_norms = []
for cf_knots in cf_knots_map.values():
    f_W_knot_norms.extend(np.linalg.norm(cf_knots, axis=1))
f_W_knot_norms = np.array(f_W_knot_norms)
plt.hist(f_W_knot_norms[f_W_knot_norms > 0], bins=50)
plt.show()

#%%
x = np.linspace(0, 500, 100)
y = 1 - np.exp(-x / (np.percentile(f_W_knot_norms, 95) / 2.3))
plt.plot(x, y)
plt.show()

# %% video rendering
# assert False
time.sleep(1 + T * h)
frames_path_prefix = "/Users/pangtao/PycharmProjects/contact_videos"
# frames_path_prefix = "/home/amazon/PycharmProjects/contact_videos"

# folder_path_normal_color = os.path.join(frames_path_prefix,
#                                         "allegro_rgba_0_normal_color")
# q_vis.render_trajectory(x_traj_knots=q_knots_computed,
#                         h=prob_rrt.params.h,
#                         folder_path=folder_path_normal_color,
#                         fps=120)


folder_path = os.path.join(frames_path_prefix, "allegro_rgba_6")
q_vis.render_trajectory(x_traj_knots=q_knots_computed,
                        h=prob_rrt.params.h,
                        folder_path=folder_path,
                        fps=120,
                        contact_results_list=contact_results_list,
                        stride=n_steps)




#%%
idx_u_not_nan = np.invert(np.isnan(u_knots_trimmed[:, 0]))
idx_q = np.hstack((idx_u_not_nan, [False]))
print(np.linalg.norm(
    q_knots_trimmed[idx_q][:, q_dynamics.get_q_a_indices_into_x()]
    - u_knots_trimmed[idx_u_not_nan]) / T)

