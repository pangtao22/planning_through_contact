import os.path
import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import PiecewisePolynomial

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import (
    IrsLqrQuasistatic)
from irs_lqr.irs_lqr_params import IrsLqrQuasistaticParameters

from planar_hand_setup import *

#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h


# quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 quasistatic_model_path=q_model_path,
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
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]

q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0.0, 0.35, 0])

q0_dict = {idx_u: q_u0,
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}


#%% try running the dynamics.
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u_traj_0 = np.zeros((T, dim_u))

x = np.copy(x0)

q_dict_traj = [q0_dict]
for i in range(T):
    print('--------------------------------')
    t = h * i
    q_cmd_dict = {idx_a_l: q_robot_l_traj.value(t + h).ravel(),
                  idx_a_r: q_robot_r_traj.value(t + h).ravel()}
    u = q_dynamics.get_u_from_q_cmd_dict(q_cmd_dict)
    x_next = q_dynamics.dynamics_py(x, u, mode='qp_mp', gradient_mode=GradientMode.kBOnly)
    Dq_nextDq = q_dynamics.q_sim_py.get_Dq_nextDq()
    Dq_nextDqa_cmd = q_dynamics.q_sim_py.get_Dq_nextDqa_cmd()

    q_dynamics.dynamics(x, u, gradient_mode=GradientMode.kBOnly)
    Dq_nextDq_cpp = q_dynamics.q_sim.get_Dq_nextDq()
    Dq_nextDqa_cmd_cpp = q_dynamics.q_sim.get_Dq_nextDqa_cmd()

    print('t={},'.format(t), 'x:', x, 'u:', u)
    print('Dq_nextDq error cpp vs python',
          np.linalg.norm(Dq_nextDq - Dq_nextDq_cpp))
    print('Dq_nextDqa_cmd error cpp vs python',
          np.linalg.norm(Dq_nextDqa_cmd - Dq_nextDqa_cmd_cpp))
    u_traj_0[i] = u
    x = x_next
    q_dict_traj.append(q_dynamics.get_q_dict_from_x(x))


q_sim_py.animate_system_trajectory(h, q_dict_traj)


#%%
params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    idx_u: np.array([10, 10, 10]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a_l: 10 * np.array([1, 1]),
    idx_a_r: 10 * np.array([1, 1])}

params.T = T

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

params.sampling = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.use_zmq_workers = use_workers
params.gradient_mode = gradient_mode
params.num_samples = num_samples

irs_lqr_q = IrsLqrQuasistatic(q_dynamics=q_dynamics, params=params)

#%%
xd_dict = {idx_u: q_u0 + np.array([-0.3, 0, 0.3]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}
xd = q_dynamics.get_x_from_q_dict(xd_dict)
x_trj_d = np.tile(xd, (T + 1, 1))
irs_lqr_q.initialize_problem(x0=x0, x_trj_d=x_trj_d, u_trj_0=u_traj_0)


#%% compare zero-order and first-order gradient estimation.
std_dict = {idx_u: np.ones(3) * 1e-3,
            idx_a_r: np.ones(2) * 0.1,
            idx_a_l: np.ones(2) * 0.1}
std_x = q_dynamics.get_x_from_q_dict(std_dict)
std_u = q_dynamics.get_u_from_q_cmd_dict(std_dict)
ABhat1 = q_dynamics.calc_AB_first_order(x, u, 100, std_u)
ABhat0 = q_dynamics.calc_B_zero_order(x, u, 100, std_u=std_u)


#%%
t0 = time.time()
irs_lqr_q.iterate(num_iters)
t1 = time.time()

print(f"iterate took {t1 - t0} seconds.")

#%% profile iterate
# cProfile.runctx('irs_lqr_q.iterate(10)',
#                 globals=globals(), locals=locals(),
#                 filename='contact_first_order_stats_multiprocessing')

# np.savetxt("examples/planar_hand/analysis/planar_hand_zero_order_B.csv",
#     irs_lqr_q.cost_all_list, delimiter=",")


#%%
x_traj_to_publish = irs_lqr_q.x_trj_best
q_dynamics.publish_trajectory(x_traj_to_publish)
print('x_goal:', xd)
print('x_final:', x_traj_to_publish[-1])


#%% plot different components of the cost for all iterations.
plt.figure()
plt.plot(irs_lqr_q.cost_all_list, label='all')
plt.plot(irs_lqr_q.cost_Qa_list, label='Qa')
plt.plot(irs_lqr_q.cost_Qu_list, label='Qu')
plt.plot(irs_lqr_q.cost_Qa_final_list, label='Qa_f')
plt.plot(irs_lqr_q.cost_Qu_final_list, label='Qu_f')
plt.plot(irs_lqr_q.cost_R_list, label='R')

plt.title('Trajectory cost')
plt.xlabel('Iterations')
# plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
