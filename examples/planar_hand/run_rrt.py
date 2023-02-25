import numpy as np
from contact_sampler import PlanarHandContactSampler

from qsim.parser import QuasistaticParser

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams, SmoothingMode
from planar_hand_setup import *

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %% quasistatic system
q_parser = QuasistaticParser(q_model_path)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = q_sim.num_dofs()
dim_u = q_sim.num_actuated_dofs()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarHandContactSampler(
    q_sim=q_sim, q_sim_py=q_sim_py, pinch_prob=0.1
)

q_u0 = np.array([0.0, 0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(q_u0)
q0 = q_sim.get_q_vec_from_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-0.3, 0.3], [0.3, 0.5], [-0.01, np.pi + 0.01]])
}

# %% RRT testing
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.h = h
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticPyramid
rrt_params.log_barrier_weight_for_bundling = 100
rrt_params.root_node = IrsNode(q0)
rrt_params.max_size = 1000
rrt_params.goal = np.copy(q0)
rrt_params.goal[2] = np.pi
rrt_params.termination_tolerance = 0.01
rrt_params.goal_as_subgoal_prob = 0.3
rrt_params.regularization = 1e-4
rrt_params.rewire = False
rrt_params.grasp_prob = 0.
rrt_params.distance_threshold = np.inf
rrt_params.stepsize = 0.35
rrt_params.distance_metric = "local_u"

# params.distance_metric = 'global'  # If using global metric
rrt_params.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 1.0])

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
prob_rrt.iterate()

d_batch = prob_rrt.calc_distance_batch(rrt_params.goal)
print("minimum distance: ", d_batch.min())

# %%
node_id_closest = np.argmin(d_batch)

# %%
prob_rrt.save_tree(f"planar_hand_tree_{rrt_params.max_size}_{0}.pkl")
# prob_rrt.save_tree(os.path.join(
#     data_folder,
#     "randomized",
#     f"tree_{params.max_size}_{0}.pkl"))
