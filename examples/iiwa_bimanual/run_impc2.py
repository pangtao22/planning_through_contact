#!/usr/bin/env python3
import copy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    PiecewisePolynomial,
    RotationMatrix,
    AngleAxis,
    Quaternion,
    RigidTransform,
)
from pydrake.math import RollPitchYaw

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp, GradientMode, ForwardDynamicsMode

from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters

from iiwa_bimanual_setup import *

# %% sim setup
h = 0.01
T = 25  # num of time steps to simulate forward.
duration = T * h
max_iterations = 40

# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)
# q_parser.set_sim_params(gravity=[0, 0, -10])

q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()

idx_a_l = plant.GetModelInstanceByName(iiwa_l_name)
idx_a_r = plant.GetModelInstanceByName(iiwa_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
dim_u_l = plant.num_positions(idx_a_l)
dim_u_r = plant.num_positions(idx_a_r)

# initial conditions.
q_a0_r = [0.11, 1.57, 0, 0, 0, 0, 0]
q_a0_l = [-0.09, 1.03, 0.04, -0.61, -0.15, -0.06, 0]
q_u0 = np.array([1, 0, 0, 0, 0.55, 0, 0.315])

q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}

# %%
params = IrsMpcQuasistaticParameters()
params.h = h
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 10, 10, 10]),
    idx_a_l: np.ones(dim_u_l) * 5e-2,
    idx_a_r: np.ones(dim_u_r) * 5e-2,
}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a_l: 50 * np.ones(dim_u_l), idx_a_r: 50 * np.ones(dim_u_r)}

u_size = 2.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * h, np.ones(dim_u) * u_size * h]
)


params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.ones(dim_u) * 0.2
params.num_samples = 100
# analytic bundling
params.log_barrier_weight_initial = 200
log_barrier_weight_final = 6000
base = (
    np.log(log_barrier_weight_final / params.log_barrier_weight_initial)
    / max_iterations
)
base = np.exp(base)
params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

params.use_A = False
params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)
q_sim_py = prob_mpc.q_vis.q_sim_py
# %%
q_sim_py.update_mbp_positions(q0_dict)
q_sim_py.draw_current_configuration()

# %%
Q_WB_d = RollPitchYaw(np.pi / 4, 0, 0).ToQuaternion()
p_WB_d = q_u0[4:] + np.array([0, 0, 0], dtype=float)
q_d_dict = {
    idx_u: np.hstack([Q_WB_d.wxyz(), p_WB_d]),
    idx_a_l: q_a0_l,
    idx_a_r: q_a0_r,
}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

# %%
t0 = time.time()
prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=1)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

# %% visualize goal.
AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="drake/plant/box/box",
    length=0.4,
    radius=0.01,
    opacity=1,
)

AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="goal",
    length=0.4,
    radius=0.03,
    opacity=0.5,
)

q_sim_py.viz.vis["goal"].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4()
)


# %% Rollout trajectory according to the real physics.
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

# %% plot different components of the cost for all iterations.
prob_mpc.plot_costs()
prob_mpc.q_vis.publish_trajectory(prob_mpc.x_trj_best, h)


# %% save trajectories
things_to_save = {"x_trj": prob_mpc.x_trj_best, "u_trj": prob_mpc.u_trj_best}
with open("box_flipping_trj.pkl", "wb") as f:
    pickle.dump(things_to_save, f)
