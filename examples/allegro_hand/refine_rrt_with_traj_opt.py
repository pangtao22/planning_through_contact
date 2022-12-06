import copy
from typing import List
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    RigidTransform,
    Quaternion,
    RollPitchYaw,
    AngleAxis,
    PiecewisePolynomial,
)

from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.parser import QuasistaticParser

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from pydrake.systems.meshcat_visualizer import AddTriad

from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from allegro_hand_setup import robot_name, object_name

#%%
pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__),
    "..",
    "examples",
    "allegro_hand",
    "tree_1000_0.pkl",
)

# pickled_tree_path = "ptc_data/allegro_hand/analytic/tree_1000_0.pkl"

with open(pickled_tree_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = prob_rrt.q_dynamics
q_dynamics.update_default_sim_params(forward_mode=ForwardDynamicsMode.kSocpMp)

q_vis = QuasistaticVisualizer(
    q_sim=q_dynamics.q_sim, q_sim_py=q_dynamics.q_sim_py
)

# get goal and some problem data from RRT parameters.
q_u_goal = prob_rrt.rrt_params.goal[q_dynamics.get_q_u_indices_into_x()]
Q_WB_d = Quaternion(q_u_goal[:4])
p_WB_d = q_u_goal[4:]
dim_q = prob_rrt.dim_q
dim_u = dim_q - prob_rrt.dim_q_u

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
q_sim = q_dynamics.q_sim


#%%
# get trimmed path to goal.
q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# split trajectory into segments according to re-grasps.
segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

#%% see the segments.
prob_rrt.print_segments_displacements(q_knots_trimmed, segments)
q_vis.publish_trajectory(q_knots_trimmed, q_dynamics.q_sim_params_default.h)


#%% determining h_small and n_steps_per_h from simulating a segment.
h_small = 0.01

#%% IrsMpc
q_parser = q_dynamics.parser
plant = q_sim.get_plant()
indices_q_u_into_x = q_sim.get_q_u_indices_into_q()
indices_q_a_into_x = q_sim.get_q_a_indices_into_q()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# traj-opt parameters
params = IrsMpcQuasistaticParameters()
params.h = h_small
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 50, 50, 50.0]),
    idx_a: np.ones(dim_u) * 1e-3,
}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 200

params.R_dict = {idx_a: 10 * np.ones(dim_u)}

u_size = 5.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * params.h, np.ones(dim_u) * u_size * params.h]
)

params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.ones(dim_u) * 0.3
params.num_samples = 100
# analytic bundling
params.log_barrier_weight_initial = 100
log_barrier_weight_final = 6000
max_iterations = 10

base = (
    np.log(log_barrier_weight_final / params.log_barrier_weight_initial)
    / max_iterations
)
base = np.exp(base)
params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

params.use_A = False
params.rollout_forward_dynamics_mode = (
    q_dynamics.q_sim_params_default.forward_mode
)

prob_mpc = IrsMpcQuasistatic(
    q_sim=q_sim, parser=q_parser, params=params, q_vis=q_vis
)


#%% traj-opt for segment
sim_params_projection = copy.deepcopy(q_dynamics.q_sim_params_default)
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q: np.ndarray):
    return q_sim.calc_dynamics(q, q[indices_q_a_into_x], sim_params_projection)


q_trj_optimized_list = []
u_trj_optimized_list = []

sub_segments = segments[0:]
input("Starting refinement...")
for i_s, (t_start, t_end) in enumerate(sub_segments):
    u_trj = u_knots_trimmed[t_start:t_end]
    q_trj = q_knots_trimmed[t_start : t_end + 1]
    prob_mpc.q_vis.publish_trajectory(q_trj, prob_rrt.rrt_params.h)

    q0 = np.array(q_trj[0])
    if len(q_trj_optimized_list) > 0:
        q0[indices_q_u_into_x] = q_trj_optimized_list[-1][
            -1, indices_q_u_into_x
        ]
        print("qu0 before projection", q0[indices_q_u_into_x])
        q0 = project_to_non_penetration(q0)
        print("qu0 after projection", q0[indices_q_u_into_x])

    input("Original trajectory segment shown. Press any key to optimize...")

    q_final = np.array(q_trj[-1])
    if i_s == len(sub_segments) - 1:
        q_final[indices_q_u_into_x] = q_u_goal

    n_steps_per_h = max(2, int(np.ceil(10 / len(u_trj))))

    (
        q_trj_optimized,
        u_trj_optimized,
        idx_best,
    ) = prob_mpc.run_traj_opt_on_rrt_segment(
        n_steps_per_h=n_steps_per_h,
        h_small=h_small,
        q0=q0,
        q_final=q_final,
        u_trj=u_trj,
        max_iterations=max_iterations,
    )

    q_trj_optimized_list.append(q_trj_optimized)
    u_trj_optimized_list.append(u_trj_optimized)

    prob_mpc.plot_costs()
    prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)
    print(f"Best trajectory iteration index: {idx_best}")
    input("Optimized trajectory shown. Press any key to go to the next segment")

#%%
q_trj_optimized_all = prob_rrt.concatenate_traj_list(q_trj_optimized_list)
prob_mpc.q_vis.publish_trajectory(q_trj_optimized_all, 0.1)


#%% see differences between RRT and optimized trajectories.
for t, q_trj_optimized in enumerate(q_trj_optimized_list):
    # prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)

    t_end = segments[t][1]
    q_u_d = q_knots_trimmed[t_end, q_sim.get_q_u_indices_into_q()]
    q_u_final = q_trj_optimized[-1][indices_q_u_into_x]
    angle_diff, position_diff = prob_rrt.calc_q_u_diff(q_u_final, q_u_d)
    print("angle diff", angle_diff, "position diff", position_diff)


#%%
print("Trimming optimized trajectory segments")
q_trj_optimized_trimmed_list = []
u_trj_optimized_trimmed_list = []
for q_trj, u_trj in zip(q_trj_optimized_list, u_trj_optimized_list):
    t = prob_rrt.trim_trajectory(q_trj)
    print(f"{t} / {len(q_trj)}")
    q_trj_optimized_trimmed_list.append(q_trj[: t + 1])
    u_trj_optimized_trimmed_list.append(u_trj[:t])


q_trj_optimized_trimmed_all = prob_rrt.concatenate_traj_list(
    q_trj_optimized_trimmed_list
)
prob_mpc.q_vis.publish_trajectory(q_trj_optimized_trimmed_all, 0.1)

#%%
with open("hand_optimized_q_and_u_trj.pkl", "wb") as f:
    pickle.dump(
        {
            "q_trj_list": q_trj_optimized_trimmed_list,
            "u_trj_list": u_trj_optimized_trimmed_list,
            "h_small": h_small,
        },
        f,
    )


#%%
prob_mpc.q_vis.meshcat_vis["/Cameras/default"].set_transform(
    RigidTransform(RollPitchYaw(0, 0, 0), [-0.25, 0.2, 0.25]).GetAsMatrix4()
)
prob_mpc.q_vis.meshcat_vis["/Cameras/default/rotated/<object>"].set_property(
    "position", [-0.05, 0, 0.05]
)
prob_mpc.q_vis.meshcat_vis["/Grid"].delete()
prob_mpc.q_vis.meshcat_vis["/Axes"].delete()


#%%
res = prob_mpc.q_vis.meshcat_vis.static_html()
# save to a file
with open("allegro_hand.html", "w") as f:
    f.write(res)
