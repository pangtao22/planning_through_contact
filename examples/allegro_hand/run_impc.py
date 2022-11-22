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

#%% sim setup
h = 0.02
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
        0.03501504,
        0.75276565,
        0.74146232,
        0.83261002,
        -0.1438725,
        0.74696812,
        0.61908827,
        0.70064279,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.90415436,
        0.63256269,
        1.02378254,
        0.64089555,
        0.82444782,
    ]
)
q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
q0_dict = {idx_a: q_a0, idx_u: q_u0}

#%%
params = IrsMpcQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1]),
    idx_a: np.ones(dim_u) * 1e-3,
}

params.Qd_dict = {}
for model in q_dynamics.models_actuated:
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_dynamics.models_unactuated:
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a: 10 * np.ones(dim_u)}
params.T = T

u_size = 5.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * h, np.ones(dim_u) * u_size * h]
)

params.decouple_AB = decouple_AB
params.parallel_mode = parallel_mode

# sampling-based bundling
# params.bundle_mode = BundleMode.kFirstRandomized
params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.ones(dim_u) * 0.3
params.num_samples = num_samples

# analytic bundling
params.bundle_mode = BundleMode.kFirstAnalytic
params.log_barrier_weight_initial = 100
params.log_barrier_weight_multiplier = 1.5

irs_mpc = IrsMpcQuasistatic(q_dynamics=q_dynamics, params=params)


#%%
Q_WB_d = RollPitchYaw(0, 0, np.pi / 6).ToQuaternion()
p_WB_d = q_u0[4:] + np.array([0, 0, 0], dtype=float)
q_d_dict = {idx_u: np.hstack([Q_WB_d.wxyz(), p_WB_d]), idx_a: q_a0}
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)
xd = q_dynamics.get_x_from_q_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
irs_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

#%%
# irs_lqr_q.q_dynamics_parallel.q_sim_batch.set_num_max_parallel_executions(10)
t0 = time.time()
irs_mpc.iterate(num_iters, cost_Qu_f_threshold=1)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% visualize goal.
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

q_dynamics.q_sim_py.viz.vis["goal"].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4()
)


#%%
x_traj_to_publish = irs_mpc.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
q_dict_final = q_dynamics.get_q_dict_from_x(x_traj_to_publish[-1])
q_u_final = q_dict_final[idx_u]
p_WB_f = q_u_final[4:]
Q_WB_f = Quaternion(q_u_final[:4])
print("position error:", p_WB_f - p_WB_d)
print(
    "orientation error:", AngleAxis(Q_WB_f.multiply(Q_WB_d.inverse())).angle()
)
print()

#%% plot different components of the cost for all iterations.
irs_mpc.plot_costs()

#%%
irs_mpc
