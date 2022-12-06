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
    "allegro_hand_door",
    "tree_1000_analytic_1.pkl",
)

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
dim_q = prob_rrt.dim_q
dim_u = dim_q - prob_rrt.dim_q_u

q_sim = q_dynamics.q_sim


#%%
# get trimmed path to goal.
q_knots, u_knots = prob_rrt.get_q_and_u_knots_to_goal()
q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# split trajectory into segments according to re-grasps.
segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

if segments[0][0] == 0 and segments[0][1] == 0:
    segments.pop(0)

q_vis.publish_trajectory(q_knots_trimmed, q_dynamics.q_sim_params_default.h)


#%% see the segments.
indices_q_u_into_x = q_sim.get_q_u_indices_into_q()
indices_q_a_into_x = q_sim.get_q_a_indices_into_q()


def calc_q_u_diff(q_u_0, q_u_1):
    """
    q_u_0 and q_u_1 are 7-vectors. The first 4 elements represent a
     quaternion and the last three a position.
    Returns (angle_diff_in_radians, position_diff_norm)
    """

    return np.linalg.norm(q_u_0 - q_u_1)


for t_start, t_end in segments:
    q_u_start = q_knots_trimmed[t_start][indices_q_u_into_x]
    q_u_end = q_knots_trimmed[t_end][indices_q_u_into_x]

    q_u_diff = calc_q_u_diff(q_u_start, q_u_end)

    print("q_u diff", q_u_diff)


#%% IrsMpc
h_small = 0.01
q_parser = q_dynamics.parser
plant = q_sim.get_plant()

idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# traj-opt parameters
params = IrsMpcQuasistaticParameters()
params.h = h_small

params.R_dict = {idx_a: 200 * np.ones(dim_u)}
params.R_dict[idx_a][:3] *= 10

params.Q_dict = {
    idx_u: np.array([100, 100.0]),
    idx_a: params.R_dict[idx_a] * 0.1,
}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 100


u_size = 10.0
u_bounds = np.ones(dim_u) * u_size * params.h
# u_bounds[:3] *= 0.2
params.u_bounds_abs = np.array([-u_bounds, u_bounds])

params.smoothing_mode = SmoothingMode.k1RandomizedPyramid
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i)
params.std_u_initial = np.ones(dim_u) * 0.05
params.std_u_initial[:3] *= 0.1
params.num_samples = 50
# analytic bundling
params.log_barrier_weight_initial = 100
log_barrier_weight_final = 1000
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
def run_traj_opt_on_rrt_segment(
    n_steps_per_h: int, q0: np.ndarray, q_final: np.ndarray, u_trj: np.ndarray
):
    q_a0 = q0[indices_q_a_into_x]

    # q_goal (end of segment)
    q_u_d = q_final[indices_q_u_into_x]
    q_d_dict = {idx_u: q_u_d, idx_a: q_a0}
    q_d = q_sim.get_q_vec_from_dict(q_d_dict)

    T = len(u_trj) * n_steps_per_h
    q_trj_d = np.tile(q_d, (T + 1, 1))

    u_trj_small = IrsMpcQuasistatic.calc_u_trj_small(
        u_trj, h_small, n_steps_per_h
    )

    prob_mpc.initialize_problem(x0=q0, x_trj_d=q_trj_d, u_trj_0=u_trj_small)
    prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=0)

    return np.array(prob_mpc.x_trj_best), np.array(prob_mpc.u_trj_best)


sim_params_projection = copy.deepcopy(q_dynamics.q_sim_params_default)
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q: np.ndarray):
    return q_sim.calc_dynamics(q, q[indices_q_a_into_x], sim_params_projection)


q_trj_optimized_list = []
u_trj_optimized_list = []

n_steps_per_h = 3
sub_segments = segments[0:]
time_start = time.time()
for i_s, (t_start, t_end) in enumerate(sub_segments):
    u_trj = u_knots_trimmed[t_start:t_end]
    q_trj = q_knots_trimmed[t_start : t_end + 1]
    prob_mpc.q_vis.publish_trajectory(q_trj, prob_rrt.rrt_params.h)

    q0 = np.array(q_trj[0])
    if len(q_trj_optimized_list) > 0:
        q0[indices_q_u_into_x] = q_trj_optimized_list[-1][
            -1, indices_q_u_into_x
        ]
        q_vis.draw_configuration(q0)
        print("q0 before projection", q0)
        q0 = project_to_non_penetration(q0)
        print("q0 after projection", q0)

    input("Original trajectory segment shown. Press any key to optimize...")

    q_final = np.array(q_trj[-1])
    if i_s == len(sub_segments) - 1:
        q_final[indices_q_u_into_x] = q_u_goal
    try:
        q_trj_optimized, u_trj_optimized = run_traj_opt_on_rrt_segment(
            n_steps_per_h=n_steps_per_h, q0=q0, q_final=q_final, u_trj=u_trj
        )
    except RuntimeError or ValueError as e:
        u_trj = np.zeros_like(u_trj)
        u_trj[:] = q0[indices_q_a_into_x]
        q_trj_optimized, u_trj_optimized = run_traj_opt_on_rrt_segment(
            n_steps_per_h=n_steps_per_h, q0=q0, q_final=q_final, u_trj=u_trj
        )

    q_trj_optimized_list.append(q_trj_optimized)
    u_trj_optimized_list.append(u_trj_optimized)

    prob_mpc.plot_costs()
    prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)
    input("Optimized trajectory shown. Press any key to go to the next segment")

time_end = time.time()
print(f"Refinement took {time_end - time_start} seconds.")

#%%
def concatenate_traj_list(q_trj_list: List[np.ndarray]):
    """
    Concatenates a list of trajectories into a single trajectory.
    """
    q_trj_sizes = np.array([len(q_trj) for q_trj in q_trj_list])

    q_trj_all = np.zeros((q_trj_sizes.sum(), dim_q))

    t_start = 0
    for q_trj_size, q_trj in zip(q_trj_sizes, q_trj_list):
        q_trj_all[t_start : t_start + q_trj_size] = q_trj
        t_start += q_trj_size

    return q_trj_all


q_trj_optimized_all = concatenate_traj_list(q_trj_optimized_list)
prob_mpc.q_vis.publish_trajectory(q_trj_optimized_all, 0.1)


#%% see differences between RRT and optimized trajectories.
for t, q_trj_optimized in enumerate(q_trj_optimized_list):
    # prob_mpc.q_vis.publish_trajectory(q_trj_optimized, h_small)

    t_end = segments[t][1]
    q_u_d = q_knots_trimmed[t_end, q_sim.get_q_u_indices_into_q()]
    q_u_final = q_trj_optimized[-1][indices_q_u_into_x]
    if t == len(q_trj_optimized_list) - 1:
        q_u_d = q_u_goal
    angle_diff = calc_q_u_diff(q_u_final, q_u_d)
    print("angle diff", angle_diff)


#%%
def trim_trajectory(q_trj: np.ndarray, angle_threshold: float = 1e-3):
    q_u_final = q_trj[-1, indices_q_u_into_x]
    for t, q in enumerate(q_trj):
        q_u = q[indices_q_u_into_x]
        angle_diff = calc_q_u_diff(q_u, q_u_final)
        if angle_diff < angle_threshold:
            break

    return t


print("Trimming optimized trajectory segments")
q_trj_optimized_trimmed_list = []
for q_trj in q_trj_optimized_list:
    t = trim_trajectory(q_trj)
    print(f"{t} / {len(q_trj)}")
    q_trj_optimized_trimmed_list.append(q_trj[: t + 1])


q_trj_optimized_trimmed_all = concatenate_traj_list(
    q_trj_optimized_trimmed_list
)
prob_mpc.q_vis.publish_trajectory(q_trj_optimized_trimmed_all, 0.1)
