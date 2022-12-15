import numpy as np

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from qsim_cpp import ForwardDynamicsMode
from qsim.parser import QuasistaticParser
from contact_sampler_allegro_door import AllegroHandDoorContactSampler
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode
from allegro_hand_setup import *

from pydrake.multibody.tree import JointIndex
from pydrake.math import RollPitchYaw, RigidTransform

np.set_printoptions(precision=3, suppress=True)

q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandDoorContactSampler(q_sim, q_sim_py)

# trajectory and initial conditions.
q_a0 = np.array(
    [
        -0.14775985,
        -0.07837441,
        -0.08875541,
        0.03732591,
        0.74914169,
        0.74059597,
        0.83309505,
        0.62379958,
        1.02520157,
        0.63739027,
        0.82612123,
        -0.14798914,
        0.73583272,
        0.61479455,
        0.7005708,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.90415436,
    ]
)
q_u0 = np.array([0, 0.0])
x0 = contact_sampler.sample_contact(q_u0)

door_angle_goal = -np.pi / 12 * 5
joint_limits = {
    idx_u: np.array([[door_angle_goal, 0], [np.pi / 4, np.pi / 2]]),
    idx_a: np.zeros([dim_u, 2]),
}


#%% RRT testing
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(x0)
rrt_params.max_size = 1000
rrt_params.goal = np.copy(x0)
rrt_params.goal[q_sim.get_q_u_indices_into_q()] = [door_angle_goal, np.pi / 2]
rrt_params.termination_tolerance = 0
rrt_params.goal_as_subgoal_prob = 0.1
rrt_params.global_metric = np.ones(x0.shape) * 0.1
rrt_params.global_metric[q_sim.get_q_u_indices_into_q()] = [1, 1]
std_u = 0.2 * np.ones(19)
std_u[0:3] = 0.02
# params.regularization = 1e-3
rrt_params.std_u = std_u
rrt_params.stepsize = 0.25
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.1

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)

q_vis.draw_object_triad(length=0.1, radius=0.001, opacity=1, path="sphere/sphere")
# q_vis.draw_goal_triad(length=0.4, radius=0.01, opacity=0.7,
#    X_WG=RigidTransform(Q_WB_d, np.array([0.15, 0.15, 0.15])))
prob_rrt.iterate()

d_batch = prob_rrt.calc_distance_batch(prob_rrt.rrt_params.goal)
node_id_closest = np.argmin(d_batch)
print("closest distance to goal", d_batch[node_id_closest])

# %%
prob_rrt.save_tree(f"allegro_door_tree_{rrt_params.max_size}.pkl")
