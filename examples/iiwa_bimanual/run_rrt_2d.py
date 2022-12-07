#!/usr/bin/env python3
from qsim.parser import QuasistaticParser

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams, SmoothingMode

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar import ContactSamplerBimanualPlanar
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %% sim setup
q_parser = QuasistaticParser(q_model_path_planar)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a_l = plant.GetModelInstanceByName(iiwa_l_name)
idx_a_r = plant.GetModelInstanceByName(iiwa_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

dim_u_l = plant.num_positions(idx_a_l)
dim_u_r = plant.num_positions(idx_a_r)

contact_sampler = ContactSamplerBimanualPlanar()

# initial conditions.
q_a0_r = [-0.7, -1.4, 0]
q_a0_l = [0.7, -1.4, 0]
q_u0 = np.array([0.65, 0, 0])
q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[0.25, 0.75], [-0.3, 0.3], [-np.pi - 0.1, 0.1]])
}

q_u_goal = np.array([0.5, 0, -np.pi])

rrt_params = IrsRrtProjectionParams(q_model_path_planar, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(q0)
rrt_params.max_size = 3000
rrt_params.goal = np.copy(q0)
rrt_params.goal[q_sim.get_q_u_indices_into_q()] = q_u_goal

rrt_params.termination_tolerance = 0.01
rrt_params.goal_as_subgoal_prob = 0.2
rrt_params.global_metric = np.ones(q0.shape) * 0.1
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
std_u = 0.2 * np.ones(6)
rrt_params.regularization = 1e-3
# params.log_barrier_weight_for_bundling = 1000
rrt_params.std_u = std_u
rrt_params.stepsize = 0.2
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.3
rrt_params.h = 0.05

rrt_params.enforce_robot_joint_limits = True

#%%
draw_goal_and_object_triads_2d(q_vis, plant, q_u_goal)

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
prob_rrt.iterate()

q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)

prob_rrt.save_tree("bimanual_planar.pkl")


#%%
