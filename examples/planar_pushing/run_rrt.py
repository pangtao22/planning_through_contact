import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pydrake.all import RigidTransform, RollPitchYaw
from qsim.parser import QuasistaticParser

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams, SmoothingMode

from planar_pushing_setup import *
from contact_sampler import PlanarPushingContactSampler

# %% quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = q_sim.num_dofs()
dim_u = q_sim.num_actuated_dofs()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarPushingContactSampler(q_sim=q_sim, q_sim_py=q_sim_py)

q_u0 = np.array([0.0, 0.5, 0])
x0 = contact_sampler.sample_contact(q_u0)

joint_limits = {
    idx_u: np.array([[-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi]]),
    idx_a: np.array([[-2.0, 2.0], [-2.0, 2.0]]),
}

# %% RRT Testing
rrt_params = IrsRrtProjectionParams()
rrt_params.q_model_path = q_model_path
rrt_params.joint_limits = joint_limits
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticPyramid
rrt_params.log_barrier_weight_for_bundling = 100
rrt_params.root_node = IrsNode(x0)
rrt_params.max_size = 5000
rrt_params.goal = np.copy(x0)
rrt_params.goal[1] = -0.5
rrt_params.goal[2] = -3 * np.pi / 4
rrt_params.termination_tolerance = 0.1
rrt_params.goal_as_subgoal_prob = 0.1
rrt_params.grasp_prob = 0.2
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.regularization = 1e-2
# rrt_params.distance_metric = 'global'  # If using global metric
rrt_params.distance_threshold = 50

rrt_params.global_metric = np.array([20, 20, 20, 1e-3, 1e-3])

q_vis.draw_object_triad(length=0.7, radius=0.01, opacity=1, path="box/box")
Q_WB_d = RollPitchYaw(rrt_params.goal[2], 0, 0)
p_WB_d = np.array([0, rrt_params.goal[0], rrt_params.goal[1]])
q_vis.draw_goal_triad(
    length=1.0,
    radius=0.02,
    opacity=0.7,
    X_WG=RigidTransform(Q_WB_d, p_WB_d),
)


prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
prob_rrt.iterate()


# %%
d_batch = prob_rrt.calc_distance_batch(rrt_params.goal)
print("minimum distance: ", d_batch.min())
(
    q_knots_trimmed,
    u_knots_trimmed,
) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)
