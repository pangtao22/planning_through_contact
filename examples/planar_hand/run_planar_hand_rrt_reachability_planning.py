from typing import Dict
import time
import matplotlib.pyplot as plt
import numpy as np

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import IrsLqrQuasistatic
from irs_lqr.irs_lqr_params import IrsLqrQuasistaticParameters

from planar_hand_setup import *

from dash_app_common import (add_goal_meshcat, calc_X_WG)

from rrt.planner import RRT, ConfigurationSpace, TreeNode
from rrt.utils import save_rrt

# %% sim setup
q_dynamics = QuasistaticDynamics(h=h,
                                 quasistatic_model_path=quasistatic_model_path,
                                 internal_viz=True)
q_sim_py = q_dynamics.q_sim_py

plant = q_sim_py.get_plant()
q_sim_py.get_robot_name_to_model_instance_dict()
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

# %%
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u

# %%
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

# %%
# initial conditions (root).
qa_l_0 = [-np.pi / 4, -np.pi / 4]
qa_r_0 = [np.pi / 4, np.pi / 4]
q_u0 = np.array([0.0, 0.35, 0])
q0_dict = {model_u: q_u0,
           model_a_l: qa_l_0,
           model_a_r: qa_r_0}

cspace = ConfigurationSpace(model_u=model_u, model_a_l=model_a_l,
                            model_a_r=model_a_r,
                            q_sim=q_sim_py)
rrt = RRT(
    root=TreeNode(q0_dict, q_dynamics=q_dynamics, cspace=cspace),
    cspace=cspace, q_dynamics=q_dynamics)

current_node = rrt.root

vis = q_dynamics.q_sim_py.viz.vis
add_goal_meshcat(vis)

while True:
    q_goal_candidates = cspace.sample_reachable_near(current_node, rrt,
                                           method="explore", n=2)

    for q_goal in q_goal_candidates:
        # Set goal visualization
        q_u = q_goal[model_u]
        X_WG = calc_X_WG(y=q_u[0], z=q_u[1], theta=q_u[2])
        vis['goal'].set_transform(X_WG)

        q_start = current_node.q

        xd = q_dynamics.get_x_from_q_dict(q_goal)
        u0 = q_dynamics.get_u_from_q_cmd_dict(q_goal)

        irs_lqr_q.initialize_problem(
            x0=q_dynamics.get_x_from_q_dict(q_start),
            x_trj_d=np.tile(xd, (T + 1, 1)),
            u_trj_0=np.tile(u0, (T, 1)))

        irs_lqr_q.iterate(num_iters)

        q_reached = q_dynamics.get_q_dict_from_x(irs_lqr_q.x_trj_best[-1])
        new_node = rrt.add_tree_node(current_node, q_reached, irs_lqr_q.cost_best,
                                q_goal, irs_lqr_q.x_trj_best)
        q_dynamics.publish_trajectory(irs_lqr_q.x_trj_best)
        rrt.rewire(new_node, irs_lqr_q, T, num_iters, k=5)

        if cspace.close_to_joint_limits(q_reached) or not new_node.in_contact:
            q_regrasp, cost, x_trj = cspace.regrasp(q_reached, q_dynamics)
            rrt.add_tree_node(new_node, q_regrasp, cost, q_regrasp, x_trj)

        print("Tree size: ", rrt.size)

        if rrt.size == 50:
            rrt.visualize_meshcat()

    current_node = rrt.sample_node(mode="explore")

    if rrt.size > 1500:
        break

rrt.visualize_meshcat(groupby="object")

# Planning
# Sample random configuration
u_limit = cspace.joint_limits[cspace.model_u]
qu_goal = np.random.rand(3) * (u_limit[:, 1] - u_limit[:, 0]) + u_limit[:, 0]
q_goal = cspace.sample_enveloping_grasp(qu_goal)

node_g = TreeNode(q_goal, parent=None, calc_reachable=False,
                  q_dynamics=q_dynamics, cspace=cspace)
last_node = rrt.get_nearest_node(node_g.q[model_u])

# Set goal visualization
X_WG = calc_X_WG(y=qu_goal[0], z=qu_goal[1], theta=qu_goal[2])
vis['goal'].set_transform(X_WG)

planned_traj = []

irs_lqr_q.solve(last_node.q, q_goal, T, num_iters)

planned_traj.append(irs_lqr_q.x_trj_best)
cost = irs_lqr_q.cost_best

# Backtrack node parent to obtain the trajectory
while True:
    parent_node = list(rrt.predecessors(last_node))[0]
    planned_traj.append(last_node.x_waypoints)
    last_node = parent_node

    if len(list(rrt.predecessors(last_node))) == 0:
        break

# Visualize entire trajectory
trajectory = []
for traj in planned_traj[::-1]:
    for x in traj[:-1]:
        trajectory.append(x)
trajectory.append(planned_traj[0][-1])
trajectory = np.array(trajectory)
q_dynamics.publish_trajectory(trajectory)

cost += last_node.value
print("Cost: ", cost)

# Compare with directly solving traj opt
irs_lqr_q.solve(rrt.root.q, q_goal, irs_lqr_q.T,
              num_iters)
q_dynamics.publish_trajectory(irs_lqr_q.x_trj_best)
print("Direct Trajopt cost:", irs_lqr_q.cost_best)

# Smooth out trajectory
irs_lqr_q.T = trajectory.shape[0] - 1
irs_lqr_q.solve(rrt.root.q, q_goal, irs_lqr_q.T,
              num_iters, x_trj_d=trajectory)
q_dynamics.publish_trajectory(irs_lqr_q.x_trj_best)
print("Smoothed cost:", irs_lqr_q.cost_best)

save_rrt(rrt)