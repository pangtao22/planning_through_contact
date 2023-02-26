#!/usr/bin/env python3
import time

from pydrake.all import (
    AngleAxis,
    Quaternion,
    RigidTransform,
)
from pydrake.math import RollPitchYaw
from qsim.parser import QuasistaticParser

from allegro_hand_setup import *
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %% sim setup
h = 0.01
T = 20  # num of time steps to simulate forward.
duration = T * h
max_iterations = 15

# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path_hardware)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py

plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# initial conditions.
q_a0 = np.array(
    [
        0.03501504,
        0.75276565,
        0.74146232,
        0.6,
        -0.1438725,
        0.74696812,
        0.61908827,
        0.5,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.6,
        0.63256269,
        1.02378254,
        0.7,
        0.5,
    ]
)

q_u0 = np.array([1, 0, 0, 0, -0.08, 0.001, 0.076])

q0_dict = {idx_a: q_a0, idx_u: q_u0}

# %%
params = IrsMpcQuasistaticParameters()
params.h = h
params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1.0]),
    idx_a: np.ones(dim_u) * 5e-2,
}

params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 100

params.R_dict = {idx_a: 20 * np.ones(dim_u)}

u_size = 3.0
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


# %%
Q_WB_d = RollPitchYaw(0, 0, np.pi / 6).ToQuaternion()
p_WB_d = q_u0[4:] + np.array([0, 0, 0], dtype=float)
q_d_dict = {idx_u: np.hstack([Q_WB_d.wxyz(), p_WB_d]), idx_a: q_a0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)
u0 = q_sim.get_q_a_cmd_vec_from_dict(q0_dict)
xd = q_sim.get_q_vec_from_dict(q_d_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
u_trj_0 = np.tile(u0, (T, 1))
prob_mpc.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_trj_0)

# %% visualize goal.
q_vis.draw_configuration(q_sim.get_q_vec_from_dict(q0_dict))
q_vis.draw_object_triad(
    length=0.1, radius=0.001, opacity=1, path="sphere/sphere"
)

q_vis.draw_goal_triad(
    length=0.1,
    radius=0.005,
    opacity=0.5,
    X_WG=RigidTransform(Q_WB_d, p_WB_d),
)

# %%
t0 = time.time()
prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=1)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

# %% Rollout trajectory according to the real physics.
x_trj_to_publish = prob_mpc.rollout(
    x0=x0, u_trj=prob_mpc.u_trj_best, forward_mode=ForwardDynamicsMode.kSocpMp
)

q_vis.publish_trajectory(x_trj_to_publish, h)
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
q_vis.publish_trajectory(prob_mpc.x_trj_best, h)

assert False
# %% save trajectories
things_to_save = {"x_trj": prob_mpc.x_trj_best, "u_trj": prob_mpc.u_trj_best}
with open("hand_trj.pkl", "wb") as f:
    pickle.dump(things_to_save, f)

# %%
u_trj_best = prob_mpc.u_trj_best
q_trj_best = prob_mpc.x_trj_best

q_trj_computed = np.zeros_like(q_trj_best)
q_trj_computed[0] = q_trj_best[0]

for t in range(T):
    q_trj_computed[t + 1] = prob_mpc.q_sim.calc_dynamics(
        q_trj_computed[t], u_trj_best[t], prob_mpc.sim_params_rollout
    )

print(q_trj_computed - q_trj_best)

# %% reduce hydro-planing with smaller time steps.
from pydrake.all import PiecewisePolynomial

t_trj = np.arange(T) * h
u_trj_poly = PiecewisePolynomial.ZeroOrderHold(t_trj, u_trj_best.T)

h_small = 0.01
N = int(h / h_small)

q_trj_small = np.zeros((T * N + 1, prob_mpc.dim_x))
q_trj_small[0] = q_trj_best[0]

sim_params_small = copy.deepcopy(prob_mpc.sim_params_rollout)
sim_params_small.h = h_small
sim_params_small.unactuated_mass_scale = np.nan

for t in range(N * T):
    q_trj_small[t + 1] = prob_mpc.q_sim.calc_dynamics(
        q_trj_small[t],
        u_trj_poly.value(h_small * t).squeeze(),
        sim_params_small,
    )

# %%
prob_mpc.q_vis.publish_trajectory(q_trj_small[::N], h)

# %%
np.linalg.norm(q_trj_small[:N] - q_trj_small[N], axis=1)
