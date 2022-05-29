from typing import List
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import ContactResults, RigidTransform, Quaternion

from qsim_cpp import ForwardDynamicsMode

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from dash_vis.dash_common import trace_nodes_to_root_from
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from pydrake.systems.meshcat_visualizer import AddTriad

# %%
# pickled_tree_path = os.path.join(
#     os.path.dirname(irs_rrt.__file__), '..',
#     'examples', 'planar_hand', "tree_2000_0.pkl")

pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__), '..',
    'examples', 'allegro_hand', "tree_2000_4.pkl")

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


def get_u_knots_from_node_idx_path(node_idx_path: List[int]):
    n = len(node_idx_path)
    u_knots = np.zeros((n - 1, dim_u))
    for i in range(n - 1):
        id_node0 = node_idx_path[i]
        id_node1 = node_idx_path[i + 1]
        u_knots[i] = tree.edges[id_node0, id_node1]['edge'].u

    return u_knots


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


# %%
# find closet point to goal.
d_batch = prob_rrt.calc_distance_batch(prob_rrt.params.goal)
node_id_closest = np.argmin(d_batch)
print("closest distance to goal", d_batch[node_id_closest])

node_idx_path = trace_nodes_to_root_from(node_id_closest, tree)
node_idx_path = np.array(node_idx_path)

q_knots = prob_rrt.q_matrix[node_idx_path]
u_knots = get_u_knots_from_node_idx_path(node_idx_path)

node_idx_path_to_keep = trim_regrasps(u_knots)
node_idx_path_trimmed = node_idx_path[node_idx_path_to_keep]
q_knots_trimmed = prob_rrt.q_matrix[node_idx_path_trimmed]
u_knots_trimmed = u_knots[node_idx_path_to_keep[1:]]

# Compute q_knots again to get ContactResults, which are not saved in the tree.
q_knots_computed = np.zeros_like(q_knots_trimmed)
q_knots_computed[0] = q_knots_trimmed[0]
contact_results_list = [ContactResults()]
T = len(u_knots_trimmed)

for t in range(T):
    u_t = u_knots_trimmed[t]
    if any(np.isnan(u_t)):
        q_knots_computed[t + 1] = q_knots_trimmed[t + 1]
        contact_results_list.append(ContactResults())
    else:
        q_knots_computed[t + 1] = q_dynamics.dynamics(
            q_knots_trimmed[t], u_t)
        contact_results_list.append(q_dynamics.q_sim.get_contact_results_copy())

print("q_knots_norm_diff trimmed vs computed",
      np.linalg.norm(q_knots_trimmed - q_knots_computed))
assert np.allclose(q_knots_trimmed, q_knots_computed, atol=1e-6)
q_vis.publish_trajectory(q_knots_computed, prob_rrt.params.h)

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

#%%
cf_knots_map = q_vis.calc_contact_forces_knots_map(
    contact_results_list)
f_W_knot_norms = []
for cf_knots in cf_knots_map.values():
    f_W_knot_norms.extend(np.linalg.norm(cf_knots, axis=1))
f_W_knot_norms = np.array(f_W_knot_norms)
plt.hist(f_W_knot_norms[f_W_knot_norms > 0], bins=50)
plt.show()

# %% video rendering
time.sleep(5.0)
frames_path_prefix = "/Users/pangtao/PycharmProjects/contact_videos"

folder_path_normal_color = os.path.join(frames_path_prefix,
                                        "allegro_rgba_0_normal_color")
q_vis.render_trajectory(x_traj_knots=q_knots_computed,
                        h=prob_rrt.params.h,
                        folder_path=folder_path_normal_color,
                        fps=120)


folder_path = os.path.join(frames_path_prefix, "allegro_rgba_0")
q_vis.render_trajectory(x_traj_knots=q_knots_computed,
                        h=prob_rrt.params.h,
                        folder_path=folder_path,
                        fps=120,
                        contact_results_list=contact_results_list)



#%%
x = np.linspace(0, 500, 100)
y = 1 - np.exp(-x / (np.percentile(f_W_knot_norms, 95) / 2.3))
plt.plot(x, y)
plt.show()
