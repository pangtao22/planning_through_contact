#!/usr/bin/env python3
import time
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
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from allegro_hand_setup import *

# %% sim setup
T = int(round(1 / h))  # num of time steps to simulate forward.
duration = T * h

# quasistatic dynamical system
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)

dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# initial conditions.
q_a0 = np.array(
    [
        -0.14775985,
        -0.07837441,
        -0.08875541,
        0.03732591,
        0.74914169,
        0.74059597,
        0.83309505,
        0.62379958,
        1.02520157,
        0.63739027,
        0.82612123,
        -0.14798914,
        0.73583272,
        0.61479455,
        0.7005708,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.90415436,
    ]
)
q_u0 = np.array([0, 0.0])
q0_dict = {idx_a: q_a0, idx_u: q_u0}

# %%
params = IrsMpcQuasistaticParameters()

idx_a_cost = np.ones(dim_u) * 1e-1
idx_a_cost[0:3] = 1

params.Q_dict = {idx_u: np.array([10, 10]), idx_a: idx_a_cost}

params.Qd_dict = {}
for model in q_dynamics.models_actuated:
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_dynamics.models_unactuated:
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a: 10 * idx_a_cost}
params.T = T

params.u_bounds_abs = np.array([-np.ones(dim_u) * h, np.ones(dim_u) * h])

params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.hstack([np.ones(3) * 0.01, np.ones(16) * 0.1])

params.decouple_AB = decouple_AB
params.num_samples = num_samples
params.bundle_mode = bundle_mode
params.parallel_mode = parallel_mode

irs_mpc = IrsMpcQuasistatic(q_dynamics=q_dynamics, params=params)

# %%
q_d_dict = {idx_u: np.array([0, np.pi / 2]), idx_a: q_a0}
x0 = q_dynamics.get_x_from_q_dict(q0_dict)

xd = q_dynamics.get_x_from_q_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(q_a0, (T, 1))
u_trj_0[:, 0] = np.linspace(-0.3, 0.1, T)
irs_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

# %%
irs_mpc.iterate(max_iterations=10, cost_Qu_f_threshold=1)
irs_mpc.plot_costs()

# %%
x_traj_to_publish = irs_mpc.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
q_dict_final = q_dynamics.get_q_dict_from_x(x_traj_to_publish[-1])

# %% plot different components of the cost for all iterations.


# %% visualize goal.
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
    opacity=0.5,
)


#%%
from pydrake.all import JointIndex

for i in range(plant.num_joints()):
    print(plant.get_joint(JointIndex(i)))
