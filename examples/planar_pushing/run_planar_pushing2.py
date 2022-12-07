import os.path
import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import PiecewisePolynomial

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp, GradientMode, ForwardDynamicsMode


from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters

from planar_pushing_setup import *

#%% sim setup

T = 20
duration = T * h

#%% sim setup
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(gravity=[0, 0, 0], h=h)
q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()

# quasistatic dynamical system
dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# trajectory and initial conditions.
nq_a = 2
q_u0 = np.array([0.0, 0.5, 0])
q_a0 = np.array([0.0, -0.1])
q0_dict = {idx_u: q_u0, idx_a: q_a0}

#%%
params = IrsMpcQuasistaticParameters()
params.Q_dict = {idx_u: np.array([20, 20, 30]), idx_a: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {idx_a: 10 * np.array([1, 1])}

u_size = 2.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * h, np.ones(dim_u) * u_size * h]
)
# params.u_bounds_rel = np.array([
#    -np.ones(dim_u) * 0.2, np.ones(dim_u) * 0.2])

params.smoothing_mode = SmoothingMode.k1RandomizedPyramid
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

params.use_A = True
params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)

#%%

qd_dict = {idx_u: q_u0 + np.array([0.5, 0.5, -np.pi / 4]), idx_a: [0.0, 0.0]}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(qd_dict)

q_sim_py = q_parser.make_simulator_py(internal_vis=True)
q_sim_py.update_mbp_positions_from_vector(x0)
q_sim_py.draw_current_configuration()

x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

#%%
t0 = time.time()
prob_mpc.iterate(max_iterations=10, cost_Qu_f_threshold=0.0)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% plot different components of the cost for all iterations.
prob_mpc.plot_costs()


#%%
x_traj_to_publish = prob_mpc.x_trj_best
prob_mpc.q_vis.publish_trajectory(x_traj_to_publish)
print("x_goal:", xd)
print("x_final:", x_traj_to_publish[-1])
