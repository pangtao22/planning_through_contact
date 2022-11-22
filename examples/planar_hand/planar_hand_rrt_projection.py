import numpy as np
from contact_sampler import PlanarHandContactSampler
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams
from planar_hand_setup import *

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarHandContactSampler(q_dynamics, pinch_prob=0.5)

q_u0 = np.array([0.0, 0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {idx_u: np.array([[-0.3, 0.3], [0.3, 0.5], [-0.01, np.pi]])}

# %% RRT testing
params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.bundle_mode = BundleMode.kFirstAnalytic
params.log_barrier_weight_for_bundling = 100
params.root_node = IrsNode(x0)
params.max_size = 2000
params.goal = np.copy(x0)
params.goal[2] = np.pi
params.termination_tolerance = 0.01
params.goal_as_subgoal_prob = 0.1
params.regularization = 1e-4
params.rewire = False
params.grasp_prob = 0.2
params.distance_threshold = np.inf
params.distance_metric = "local_u"

# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 1.0])


prob_rrt = IrsRrtProjection(params, contact_sampler)
prob_rrt.iterate()

d_batch = prob_rrt.calc_distance_batch(params.goal)
print("minimum distance: ", d_batch.min())

#%%
node_id_closest = np.argmin(d_batch)

# %%
prob_rrt.save_tree(f"tree_{params.max_size}_{0}.pkl")
# prob_rrt.save_tree(os.path.join(
#     data_folder,
#     "randomized",
#     f"tree_{params.max_size}_{0}.pkl"))
