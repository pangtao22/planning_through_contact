import numpy as np
from contact_sampler import BoxBallContactSampler

from qsim.parser import QuasistaticParser

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams, SmoothingMode
from box_ball_setup import *

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %%
q_parser = QuasistaticParser(q_model_path)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = q_sim.num_dofs()
dim_u = q_sim.num_actuated_dofs()

idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = BoxBallContactSampler(q_sim=q_sim, q_sim_py=q_sim_py)

q_u0 = np.array([0.0])
q_a0 = np.array([0, 0.0])
q0_dict = {idx_a: q_a0, idx_u: q_u0}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

q_vis.draw_configuration(contact_sampler.sample_contact(q0))

# %%
joint_limits = {idx_u: np.array([[-1.0, 1.0]])}
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits=joint_limits)
rrt_params.h = 0.1
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticPyramid
rrt_params.log_barrier_weight_for_bundling = 100
rrt_params.root_node = IrsNode(q0)
rrt_params.max_size = 5000
rrt_params.goal = np.copy(q0)
rrt_params.goal[q_sim.get_q_u_indices_into_q()] = [0.66]
rrt_params.termination_tolerance = 0.01
rrt_params.goal_as_subgoal_prob = 0.2
rrt_params.regularization = 1e-4
rrt_params.grasp_prob = 0.0
rrt_params.distance_threshold = np.inf
rrt_params.distance_metric = "local_u"

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
prob_rrt.iterate()

q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)


prob_rrt.save_tree(f"tree_{rrt_params.max_size}_no_regrasp.pkl")
