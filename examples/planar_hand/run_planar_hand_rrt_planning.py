from typing import Dict
import time
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import PiecewisePolynomial, ModelInstanceIndex

from qsim.simulator import QuasistaticSimulator, QuasistaticSimParameters
from qsim.system import cpp_params_from_py_params
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import IrsLqrQuasistatic

from planar_hand_setup import *


from rrt.planner import RRT, ConfigurationSpace, TreeNode

#%% sim setup
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# construct C++ backend.
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

plant = q_sim_cpp.get_plant()
q_sim_py.get_robot_name_to_model_instance_dict()
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

#%%
q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u

#%%
params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    model_u: np.array([10, 10, 10]),
    model_a_l: np.array([1e-3, 1e-3]),
    model_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    model_a_l: 5 * np.array([1, 1]),
    model_a_r: 5 * np.array([1, 1])}

params.sampling = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride
params.num_samples = num_samples
params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])
params.publish_every_iteration = False

T = int(round(2 / h))  # num of time steps to simulate forward.
params.T = T
duration = T * h

irs_lqr_q = IrsLqrQuasistatic(q_dynamics=q_dynamics, params=params)

#%%
# initial conditions (root).
qa_l_0 = [-np.pi / 4, -np.pi / 4]
qa_r_0 = [np.pi / 4, np.pi / 4]
q_u0 = np.array([0.0, 0.35, 0])
q0_dict = {model_u: q_u0,
           model_a_l: qa_l_0,
           model_a_r: qa_r_0}

cspace = ConfigurationSpace(model_u=model_u, model_a_l=model_a_l, model_a_r=model_a_r,
                            q_sim=q_sim_py)
rrt = RRT(root=TreeNode(q0_dict, parent=None), cspace=cspace)

q_current = q0_dict

while True:
    q_goal = cspace.sample_near(q_current, 0.2)
    node_nearest = rrt.nearest(q_goal)
    q_start = node_nearest.q

    xd = q_dynamics.get_x_from_q_dict(q_goal)
    u0 = q_dynamics.get_u_from_q_cmd_dict(q_start)

    irs_lqr_q.initialize_problem(
        x0=q_dynamics.get_x_from_q_dict(q_start),
        x_trj_d=np.tile(xd, (T + 1, 1)),
        u_trj_0=np.tile(u0, (T, 1)))

    irs_lqr_q.iterate(num_iters)

    q_reached = q_dynamics.get_q_dict_from_x(irs_lqr_q.x_trj_best[-1])
    rrt.add_node(parent_node=node_nearest,
                 q_child=q_reached)
    q_current = q_reached

    # plot different components of the cost for all iterations.
    dq = q_goal[model_u] - q_start[model_u]
    q_error = q_goal[model_u] - q_reached[model_u]
    print('trans error / cmd',
          np.linalg.norm(q_error[:2]) / np.linalg.norm(dq[:2]))
    print('rot error / cmd', np.abs(q_error[2]) / np.abs(dq[2]))
    q_dynamics.publish_trajectory(irs_lqr_q.x_trj_best)

    plt.figure()
    plt.plot(irs_lqr_q.cost_all_list, label='all')
    plt.plot(irs_lqr_q.cost_Qa_list, label='Qa')
    plt.plot(irs_lqr_q.cost_Qu_list, label='Qu')
    plt.plot(irs_lqr_q.cost_Qa_final_list, label='Qa_f')
    plt.plot(irs_lqr_q.cost_Qu_final_list, label='Qu_f')
    plt.plot(irs_lqr_q.cost_R_list, label='R')

    plt.title('Trajectory cost')
    plt.xlabel('Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    if rrt.size > 20:
        break


