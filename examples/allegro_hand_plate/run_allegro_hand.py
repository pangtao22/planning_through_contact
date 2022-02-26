#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt
import numpy as np
import meshcat

from pydrake.all import (PiecewisePolynomial, RotationMatrix, AngleAxis,
                         Quaternion, RigidTransform)
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_quasistatic import (
    IrsMpcQuasistatic)
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from allegro_hand_setup import *

#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)

dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# initial conditions.
q_a0 = np.array([0.0, 0.3, 0.15,
                 0.03501504, 0.75276565, 0.74146232, 0.83261002, 
                 0.03256269, 1.02378254, 0.64089555, 0.82444782, 
                 -0.0438725, 0.74696812, 0.61908827, 0.70064279,
                 -0.06922541, 0.78533142, 0.82942863, 0.90415436])
q_u0 = np.array([1, 0, 0, 0, 0.0, 0.001, 0.05])
q0_dict = {idx_a: q_a0, idx_u: q_u0}

#%%
params = IrsMpcQuasistaticParameters()

idx_a_cost = np.ones(dim_u) * 1e-4
idx_a_cost[0:3] = 1e-1

params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1]),
    idx_a: np.ones(dim_u) * 1e-3}

params.Qd_dict = {}
for model in q_dynamics.models_actuated:
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_dynamics.models_unactuated:
    params.Qd_dict[model] = params.Q_dict[model]

params.R_dict = {idx_a: 10 * idx_a_cost}
params.T = T

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * h, np.ones(dim_u) * h])

params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = 0.01 * np.ones(dim_u)

params.decouple_AB = decouple_AB
params.num_samples = num_samples
params.bundle_mode = bundle_mode
params.parallel_mode = parallel_mode

irs_mpc = IrsMpcQuasistatic(q_dynamics=q_dynamics, params=params)


#%%
Q_WB_d = RollPitchYaw(0, 0, np.pi / 8).ToQuaternion()
p_WB_d = q_u0[4:] + np.array([0, 0.2, 0.2], dtype=float)
q_d_dict = {idx_u: np.hstack([Q_WB_d.wxyz(), p_WB_d]),
            idx_a: q_a0}
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)
xd = q_dynamics.get_x_from_q_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
u_trj_0[:,1] = np.linspace(0.2, -0.3, T)
u_trj_0[-5:,2] = np.linspace(0.15, 0.25, 5)
irs_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

#%%
# irs_lqr_q.q_dynamics_parallel.q_sim_batch.set_num_max_parallel_executions(10)
try:
    t0 = time.time()
    irs_mpc.iterate(num_iters, cost_Qu_f_threshold=0)
    t1 = time.time()
except:
    pass

#%% visualize goal.
AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name='frame',
    prefix='drake/plant/pen/pen',
    length=0.1,
    radius=0.001,
    opacity=1)

AddTriad(
    vis=q_dynamics.q_sim_py.viz.vis,
    name='frame',
    prefix='goal',
    length=0.1,
    radius=0.005,
    opacity=0.5)

q_dynamics.q_sim_py.viz.vis['goal'].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4())

#%%
x_traj_to_publish = irs_mpc.x_trj_list[0]
#x_traj_to_publish = irs_mpc.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
q_dict_final = q_dynamics.get_q_dict_from_x(x_traj_to_publish[-1])
q_u_final = q_dict_final[idx_u]
p_WB_f = q_u_final[4:]

#%% plot different components of the cost for all iterations.
irs_mpc.plot_costs()

#%% save visualization.
# res = q_dynamics.q_sim_py.viz.vis.static_html()
# with open("allegro_hand_irs_lqr_60_degrees_rotation.html", "w") as f:
#     f.write(res)



