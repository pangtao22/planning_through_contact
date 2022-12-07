import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

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
contact_sampler = PlanarPushingContactSampler(
    q_sim=q_sim, q_sim_py=q_sim_py)

q_u0 = np.array([0.0, 0.5, 0])
x0 = contact_sampler.sample_contact(q_u0)

joint_limits = {
    idx_u: np.array([[-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi]]),
    idx_a: np.array([[-2.0, 2.0], [-2.0, 2.0]]),
}

# %% RRT Testing
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticPyramid
rrt_params.log_barrier_weight_for_bundling = 100
rrt_params.root_node = IrsNode(x0)
rrt_params.max_size = 500
rrt_params.goal = np.copy(x0)
rrt_params.goal[1] = -0.5
rrt_params.termination_tolerance = 0.1
rrt_params.goal_as_subgoal_prob = 0.1
rrt_params.grasp_prob = 0.2
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.regularization = 1e-2
# rrt_params.distance_metric = 'global'  # If using global metric
rrt_params.distance_threshold = 50

rrt_params.global_metric = np.array([20, 20, 20, 1e-3, 1e-3])

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
prob_rrt.iterate()

d_batch = prob_rrt.calc_distance_batch(rrt_params.goal)
print("minimum distance: ", d_batch.min())

# %%
node_id_closest = np.argmin(d_batch)

# %%
prob_rrt.save_tree(f"tree_{rrt_params.max_size}_{0}.pkl")
# prob_rrt.save_tree(os.path.join(
#     data_folder,
#     "randomized",
#     f"tree_{params.max_size}_{0}.pkl"))
