#!/usr/bin/env python3
import os
import time

import numpy as np
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from pydrake.all import AngleAxis, Quaternion, RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad
from qsim.model_paths import models_dir
from qsim.parser import QuasistaticParser
from qsim_cpp import ForwardDynamicsMode

q_model_path = os.path.join(
    models_dir, "q_sys", "allegro_hand_tilted_and_sphere.yml"
)
robot_name = "allegro_hand_right"
object_name = "sphere"

# %% sim setup
h = 0.1
T = 20  # num of time steps to simulate forward.
duration = T * h

# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(gravity=[0, 0, -10])

q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# initial conditions.
q_a0 = np.array(
    [
        -0.03513728,
        0.73406172,
        0.64357553,
        0.74325654,
        0.58083794,
        0.96998129,
        0.6349077,
        0.8323073,
        -0.1095671,
        0.70771197,
        0.64165158,
        0.71923356,
        -0.04130878,
        0.80228386,
        0.83890058,
        0.90658696,
    ]
)
q_u0 = np.array(
    [
        0.99605745,
        0.02259868,
        0.08572997,
        -0.00303672,
        -0.09897396,
        0.00716867,
        0.04708814,
    ]
)
q0_dict = {idx_a: q_a0, idx_u: q_u0}

# %%
params = IrsMpcQuasistaticParameters()
params.h = h
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 10, 10, 10.0]),
    idx_a: np.ones(dim_u) * 1e-2,
}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a: 10 * np.ones(dim_u)}

u_size = 1.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * h, np.ones(dim_u) * u_size * h]
)

params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.ones(dim_u) * 0.2
params.num_samples = 100
# analytic bundling
params.log_barrier_weight_initial = 100
log_barrier_weight_final = 2000
base = np.log(log_barrier_weight_final / params.log_barrier_weight_initial) / T
base = np.exp(base)
params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

params.use_A = False
params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)

# %%
Q_WB_d = RollPitchYaw(0, 0, np.pi / 4).ToQuaternion()
p_WB_d = q_u0[4:] + np.array([0, -0.01, 0], dtype=float)
q_d_dict = {idx_u: np.hstack([Q_WB_d.wxyz(), p_WB_d]), idx_a: q_a0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

# %%
t0 = time.time()
prob_mpc.iterate(max_iterations=10, cost_Qu_f_threshold=1)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

# %% visualize goal.
q_sim_py = prob_mpc.q_vis.q_sim_py
AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="drake/plant/sphere/sphere",
    length=0.1,
    radius=0.001,
    opacity=1,
)

AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="goal",
    length=0.1,
    radius=0.005,
    opacity=0.5,
)

q_sim_py.viz.vis["goal"].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4()
)

# %% results visualization.
# Rollout trajectory according to the real physics.
x_trj_to_publish = prob_mpc.rollout(
    x0=x0, u_trj=prob_mpc.u_trj_best, forward_mode=ForwardDynamicsMode.kSocpMp
)

prob_mpc.q_vis.publish_trajectory(x_trj_to_publish, h)
q_dict_final = q_sim.get_q_dict_from_vec(x_trj_to_publish[-1])
q_u_final = q_dict_final[idx_u]
p_WB_f = q_u_final[4:]
Q_WB_f = Quaternion(q_u_final[:4] / np.linalg.norm(q_u_final[:4]))
print("position error:", p_WB_f - p_WB_d)
print(
    "orientation error:", AngleAxis(Q_WB_f.multiply(Q_WB_d.inverse())).angle()
)
print()

# plot different components of the cost for all iterations.
prob_mpc.plot_costs()

# %% save visualization.
# res = q_dynamics.q_sim_py.viz.vis.static_html()
# with open("allegro_hand_irs_lqr_60_degrees_rotation.html", "w") as f:
#     f.write(res)

#%%
q_viz = prob_mpc.q_vis
q_viz.render_trajectory(
    x_traj_knots=x_trj_to_publish,
    h=h,
    folder_path="/home/amazon/PycharmProjects/video_images",
)
