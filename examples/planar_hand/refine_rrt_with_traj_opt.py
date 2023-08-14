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
from qsim.parser import QuasistaticParser, QsimVisualizationType

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from qsim.visualizer import QuasistaticVisualizer

from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from planar_hand_setup import robot_r_name, robot_l_name, object_name

# %%
pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__),
    "..",
    "examples",
    "planar_hand",
    "planar_hand_tree_1000_0.pkl",
)

with open(pickled_tree_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_parser = QuasistaticParser(prob_rrt.rrt_params.q_model_path)

q_vis = q_parser.make_visualizer(QsimVisualizationType.Cpp)
q_sim = q_vis.q_sim

# get goal and some problem data from RRT parameters.
q_u_goal = prob_rrt.rrt_params.goal[q_sim.get_q_u_indices_into_q()]
rpy_WB_d = RollPitchYaw(np.array([q_u_goal[2], 0, 0]))
p_WB_d = np.array([0, q_u_goal[0], q_u_goal[1]])
dim_q = prob_rrt.dim_q
dim_u = q_sim.num_actuated_dofs()
dim_u_l = 2
dim_u_r = 2

# visualize goal.
q_vis.draw_object_triad(
    length=0.1,
    radius=0.001,
    opacity=1,
    path="sphere/sphere",
)
q_vis.draw_goal_triad(
    length=0.1, radius=0.005, opacity=0.7, X_WG=RigidTransform(rpy_WB_d, p_WB_d)
)

# get trimmed path to goal.
q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# split trajectory into segments according to re-grasps.
segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

# %% see the segments.
prob_rrt.print_segments_displacements(q_knots_trimmed, segments)
q_vis.publish_trajectory(prob_rrt.rrt_params.h, q_knots_trimmed)

# %% determining h_small and n_steps_per_h from simulating a segment.
h_small = 0.02

# %% IrsMpc
plant = q_sim.get_plant()
indices_q_u_into_x = q_sim.get_q_u_indices_into_q()
indices_q_a_into_x = q_sim.get_q_a_indices_into_q()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

# traj-opt parameters
impc_params = IrsMpcQuasistaticParameters()
impc_params.enforce_joint_limits = (
    prob_rrt.rrt_params.enforce_robot_joint_limits
)

impc_params.h = h_small
impc_params.Q_dict = {
    idx_u: np.array([20, 50, 10]),
    idx_a_l: np.ones(dim_u_l) * 1e-3,
    idx_a_r: np.ones(dim_u_r) * 1e-3,
}

impc_params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    impc_params.Qd_dict[model] = impc_params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    impc_params.Qd_dict[model] = impc_params.Q_dict[model] * 100

impc_params.R_dict = {
    idx_a_l: 10 * np.ones(dim_u_l),
    idx_a_r: 10 * np.ones(dim_u_r),
}

u_size = 0.5
impc_params.u_bounds_abs = np.array(
    [
        -np.ones(dim_u) * u_size,
        np.ones(dim_u) * u_size,
    ]
)

impc_params.smoothing_mode = SmoothingMode.k1AnalyticPyramid
# sampling-based bundling
impc_params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
impc_params.std_u_initial = np.ones(dim_u) * 0.1
impc_params.num_samples = 100
# analytic bundling
impc_params.log_barrier_weight_initial = 100
log_barrier_weight_final = 200
max_iterations = 20

base = (
    np.log(log_barrier_weight_final / impc_params.log_barrier_weight_initial)
    / max_iterations
)
base = np.exp(base)
impc_params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

impc_params.use_A = False
impc_params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp
prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=impc_params)

# %% traj-opt for segment
sim_params_projection = copy.deepcopy(prob_rrt.sim_params)
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q: np.ndarray):
    return q_sim.calc_dynamics(q, q[indices_q_a_into_x], sim_params_projection)


q_trj_optimized_list = []
u_trj_optimized_list = []

sub_segments = segments[0:]
# input("Starting refinement...")
for i_s, (t_start, t_end) in enumerate(sub_segments):
    u_trj = u_knots_trimmed[t_start:t_end]
    q_trj = q_knots_trimmed[t_start : t_end + 1]
    q_vis.publish_trajectory(prob_rrt.rrt_params.h, q_trj)

    q0 = np.array(q_trj[0])
    if len(q_trj_optimized_list) > 0:
        q0[indices_q_u_into_x] = q_trj_optimized_list[-1][
            -1, indices_q_u_into_x
        ]
        print("qu0 before projection", q0[indices_q_u_into_x])
        q0 = project_to_non_penetration(q0)
        print("qu0 after projection", q0[indices_q_u_into_x])

    # input("Original trajectory segment shown. Press any key to optimize...")

    q_final = np.array(q_trj[-1])
    if i_s == len(sub_segments) - 1:
        q_final[indices_q_u_into_x] = q_u_goal

    n_steps_per_h = max(5, int(np.ceil(5 / len(u_trj))))
    # print(int(np.ceil(10 / len(u_trj))))

    # n_steps_per_h = 10

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

    print(u_trj)
    print(u_trj_optimized)

    # prob_mpc.plot_costs()
    q_vis.publish_trajectory(h_small, q_trj_optimized)
    print(f"Best trajectory iteration index: {idx_best}")
    # input("Optimized trajectory shown. Press any key to go to the next segment")

# %%
q_trj_optimized_all = prob_rrt.concatenate_traj_list(q_trj_optimized_list)
q_vis.publish_trajectory(0.1, q_trj_optimized_all)

# %% see differences between RRT and optimized trajectories.
for t, q_trj_optimized in enumerate(q_trj_optimized_list):
    # prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)

    t_end = segments[t][1]
    q_u_d = q_knots_trimmed[t_end, q_sim.get_q_u_indices_into_q()]
    q_u_final = q_trj_optimized[-1][indices_q_u_into_x]
    angle_diff, position_diff = prob_rrt.calc_q_u_diff(q_u_final, q_u_d)
    print("angle diff", angle_diff, "position diff", position_diff)

# %%
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
q_vis.publish_trajectory(0.1, q_trj_optimized_trimmed_all)

# %%
with open("hand_optimized_q_and_u_trj.pkl", "wb") as f:
    pickle.dump(
        {
            "q_trj_list": q_trj_optimized_trimmed_list,
            "u_trj_list": u_trj_optimized_trimmed_list,
            "h_small": h_small,
        },
        f,
    )
