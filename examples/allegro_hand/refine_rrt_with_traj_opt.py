import copy
from typing import List
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (RigidTransform, Quaternion,
                         RollPitchYaw, AngleAxis, PiecewisePolynomial)

from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.parser import QuasistaticParser

import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from pydrake.systems.meshcat_visualizer import AddTriad

from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import (SmoothingMode, IrsMpcQuasistaticParameters)
from allegro_hand_setup import robot_name, object_name

#%%
pickled_tree_path = os.path.join(
    os.path.dirname(irs_rrt.__file__), '..',
    'examples', 'allegro_hand', "tree_1000_analytic_0.pkl")

with open(pickled_tree_path, 'rb') as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = prob_rrt.q_dynamics
q_dynamics.update_default_sim_params(forward_mode=ForwardDynamicsMode.kSocpMp)

q_vis = QuasistaticVisualizer(q_sim=q_dynamics.q_sim,
                              q_sim_py=q_dynamics.q_sim_py)

# get goal and some problem data from RRT parameters.
qu_goal = prob_rrt.params.goal[q_dynamics.get_q_u_indices_into_x()]
Q_WB_d = Quaternion(qu_goal[:4])
p_WB_d = qu_goal[4:]
dim_q = prob_rrt.dim_q
dim_u = dim_q - prob_rrt.dim_q_u

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
# get trimmed path to goal.
q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# split trajectory into segments according to re-grasps.
segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

q_vis.publish_trajectory(
    q_knots_trimmed, q_dynamics.q_sim_params_default.h)

#%%
# see the segments.
indices_q_u_into_x = q_dynamics.get_q_u_indices_into_x()


def calc_q_u_diff(q_u_0, q_u_1):
    """
    q_u_0 and q_u_1 are 7-vectors. The first 4 elements represent a
     quaternion and the last three a position.
    Returns (angle_diff_in_radians, position_diff_norm)
    """
    Q_U0 = Quaternion(q_u_0[:4])
    Q_U1 = Quaternion(q_u_1[:4])
    aa = AngleAxis(Q_U0.multiply(Q_U1.inverse()))

    return aa.angle(), np.linalg.norm(q_u_start[4:] - q_u_end[4:])


for t_start, t_end in segments:
    q_u_start = q_knots_trimmed[t_start][indices_q_u_into_x]
    q_u_end = q_knots_trimmed[t_end][indices_q_u_into_x]

    angle_diff, position_diff = calc_q_u_diff(q_u_start, q_u_end)

    print("angle diff", angle_diff,
          "position diff", position_diff)

    q_vis.publish_trajectory(
        q_knots_trimmed[t_start: t_end + 1], prob_rrt.params.h)
    input()

#%% determining h_small and n_steps_per_h from simulating a segment.
h_small = 0.01
n_steps_per_h = 5

t_start = segments[0][0]
t_end = segments[0][1]
u_trj = u_knots_trimmed[t_start: t_end]
q_trj = q_knots_trimmed[t_start: t_end + 1]

sim_params_small = copy.deepcopy(q_dynamics.q_sim_params_default)
sim_params_small.h = h_small

q_trj_small, u_trj_small = IrsMpcQuasistatic.rollout_smaller_steps(
    x0=q_trj[0], u_trj=u_trj, h_small=h_small, n_steps_per_h=n_steps_per_h,
    q_sim=q_dynamics.q_sim, sim_params=sim_params_small)

q_vis.publish_trajectory(
    q_trj_small[::n_steps_per_h], q_dynamics.q_sim_params_default.h)

# A staircase pattern in this plot is a good indication that n_steps_per_h is
# large enough.
plt.plot(q_trj_small[:, 0])
plt.show()


#%% IrsMpc
q_parser = q_dynamics.parser
q_sim = q_dynamics.q_sim
plant = q_sim.get_plant()

idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# traj-opt parameters
params = IrsMpcQuasistaticParameters()
params.h = h_small
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1.]),
    idx_a: np.ones(dim_u) * 1e-3}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a: 10 * np.ones(dim_u)}

u_size = 5.0
params.u_bounds_abs = np.array([
    -np.ones(dim_u) * u_size * params.h, np.ones(dim_u) * u_size * params.h])

params.smoothing_mode = SmoothingMode.kFirstAnalyticIcecream
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3
params.num_samples = 100
# analytic bundling
params.log_barrier_weight_initial = 100
log_barrier_weight_final = 6000
max_iterations = 15

base = np.log(
    log_barrier_weight_final / params.log_barrier_weight_initial) \
       / max_iterations
base = np.exp(base)
params.calc_log_barrier_weight = (
    lambda kappa0, i: kappa0 * (base ** i))

params.use_A = False
params.rollout_forward_dynamics_mode = \
    q_dynamics.q_sim_params_default.forward_mode

prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params,
                             q_vis=q_vis)


#%% traj-opt for segment
# q0
n_steps_per_h = 1
q0_dict = q_sim.get_q_dict_from_vec(q_trj[0])
q_u0 = q0_dict[idx_u]
q_a0 = q0_dict[idx_a]

# q_goal (end of segment)
q_u_d = q_trj[-1, q_sim.get_q_u_indices_into_q()]
q_d_dict = {idx_u: q_u_d, idx_a: q_a0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(q_d_dict)
T = len(u_trj) * n_steps_per_h
x_trj_d = np.tile(xd, (T + 1, 1))

u_trj_small = IrsMpcQuasistatic.calc_u_trj_small(u_trj, h_small, n_steps_per_h)
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_small)

prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=0)
prob_mpc.plot_costs()

q_u_final = prob_mpc.x_trj_best[-1][indices_q_u_into_x]
angle_diff, position_diff = calc_q_u_diff(q_u_final, q_u_d)
print("angle diff", angle_diff,
      "position diff", position_diff)

#%%
prob_mpc.q_vis.publish_trajectory(prob_mpc.x_trj_best, h_small)

#%%
angle_diffs = np.zeros(T + 1)
pos_diffs = np.zeros_like(angle_diffs)
for t, q in enumerate(prob_mpc.x_trj_best):
    angle_diffs[t], pos_diffs[t] = calc_q_u_diff(q[indices_q_u_into_x], q_u_d)

