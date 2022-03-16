import os.path
import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import PiecewisePolynomial

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import (
    QuasistaticDynamicsParallel)
from irs_mpc.irs_mpc_quasistatic import (
    IrsMpcQuasistatic)
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from planar_hand_setup import *

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
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)


# trajectory and initial conditions.
nq_a = 2
q_u0 = np.array([0.0, 0.35, 0])
q_a_l0 = np.array([-np.pi / 4, -np.pi / 4])
q_a_r0 = np.array([np.pi / 4, np.pi / 4])
q_cmd_dict = {idx_a_l: q_a_l0,
              idx_a_r: q_a_r0}
q0_dict = {idx_u: q_u0,
           idx_a_l: q_a_l0,
           idx_a_r: q_a_r0}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
u_traj_0 = np.tile(u0, (T, 1))


#%%
params = IrsMpcQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([20, 20, 20]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a_l: 10 * np.array([1, 1]),
    idx_a_r: 10 * np.array([1, 1])}

params.T = T

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.num_samples = num_samples
params.bundle_mode = bundle_mode
params.parallel_mode = parallel_mode

irs_mpc = IrsMpcQuasistatic(q_dynamics=q_dynamics, params=params)

#%%
xd_dict = {idx_u: q_u0 + np.array([-0.3, 0, 0.3]),
           idx_a_l: q_a_l0,
           idx_a_r: q_a_r0}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
irs_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_traj_0)


#%%
t0 = time.time()
irs_mpc.iterate(num_iters, cost_Qu_f_threshold=5)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% plot different components of the cost for all iterations.
irs_mpc.plot_costs()


#%%
x_traj_to_publish = irs_mpc.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
print('x_goal:', xd)
print('x_final:', x_traj_to_publish[-1])
