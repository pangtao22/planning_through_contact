import time

import numpy as np
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from planar_hand_setup import *
from qsim.parser import QuasistaticParser
from qsim_cpp import ForwardDynamicsMode

from contact_sampler import PlanarHandContactSampler

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

"""
Modification of run_planar_hand that uses the new irs_mpc2 interface.
"""

T = 20
duration = T * h

# %% sim setup
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(h=h, gravity=[0, 0, 0])

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

contact_sampler = PlanarHandContactSampler(
    q_sim=q_sim, q_sim_py=q_sim_py, pinch_prob=0.5
)

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

print(idx_a_l)
print(idx_a_r)

# initial conditions.
nq_a = 2
q_u0 = np.array([0.0, 0.35, 0])
q_a_l0 = np.array([-np.pi / 4, -np.pi / 4])
q_a_r0 = np.array([np.pi / 4, np.pi / 4])

q0_dict = contact_sampler.calc_enveloping_grasp(q_u0)

#
# q0_dict = {idx_u: q_u0,
#            idx_a_l: q_a_l0,
#            idx_a_r: q_a_r0}

# %% Set up IrsMpcParameters
params = IrsMpcQuasistaticParameters()
params.h = h
params.Q_dict = {
    idx_u: np.array([10, 10, 10]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3]),
}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {idx_a_l: 10 * np.array([1, 1]), idx_a_r: 10 * np.array([1, 1])}

u_size = 2.0
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * u_size * h, np.ones(dim_u) * u_size * h]
)

params.smoothing_mode = SmoothingMode.k0Pyramid
# sampling-based bundling
params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
params.std_u_initial = np.ones(dim_u) * 0.2
params.num_samples = 100
# analytic bundling
params.log_barrier_weight_initial = 500
log_barrier_weight_final = 1000
base = np.log(log_barrier_weight_final / params.log_barrier_weight_initial) / T
base = np.exp(base)
params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base**i)

params.use_A = False
params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)

# %%
qd_dict = {
    idx_u: q_u0 + np.array([-0.3, 0, 0.5]),
    idx_a_l: q_a_l0,
    idx_a_r: q_a_r0,
}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(qd_dict)

q_vis.draw_configuration(x0)

x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

# %%
n_runs = 1
costs = np.zeros((n_runs, 21))
for i in range(n_runs):
    prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)
    prob_mpc.iterate(max_iterations=20, cost_Qu_f_threshold=0.0)
    prob_mpc.plot_costs()
    costs[i] = prob_mpc.cost_all_list

q_vis.publish_trajectory(prob_mpc.x_trj_best, h)
