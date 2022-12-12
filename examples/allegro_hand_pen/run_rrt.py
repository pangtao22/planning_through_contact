import numpy as np

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.rrt_params import IrsRrtProjectionParams

from qsim_cpp import ForwardDynamicsMode
from qsim.parser import QuasistaticParser
from contact_sampler_allegro_pen import AllegroHandPenContactSampler
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode
from allegro_hand_setup import *

from pydrake.multibody.tree import JointIndex
from pydrake.math import RollPitchYaw, RigidTransform


#%% quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandPenContactSampler(q_sim, q_sim_py)

q_a0 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.03501504,
        0.75276565,
        0.74146232,
        0.83261002,
        0.63256269,
        1.02378254,
        0.64089555,
        0.82444782,
        -0.1438725,
        0.74696812,
        0.61908827,
        0.70064279,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.90415436,
    ]
)

q_u0 = np.array([1, 0, 0, 0, 0.0, 0.0, 0.05])
q0 = q_sim.get_q_vec_from_dict({idx_u:q_u0, idx_a: q_a0})
q_vis.draw_configuration(q0)

num_joints = 19  # The last joint is weldjoint (welded to the world)
joint_limits = {
    idx_u: np.array(
        [
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.2],
            [0, 0],
            [0.0, 0.2],
            [0.0, 0.2],
            [0.0, 0.2],
        ]
    ),
    idx_a: np.zeros([num_joints, 2]),
}

for i in range(num_joints):
    joint = plant.get_joint(JointIndex(i))
    if i >= 0:
        low = joint.position_lower_limits()
        upp = joint.position_upper_limits()
    joint_limits[idx_a][i, :] = [low[0], upp[0]]

joint_limits[idx_a][0, :] = joint_limits[idx_u][4, :]
joint_limits[idx_a][1, :] = joint_limits[idx_u][5, :]
joint_limits[idx_a][2, :] = joint_limits[idx_u][6, :]

#%% RRT testing
# IrsRrt rrt_params
rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(q0)
rrt_params.max_size = 2000
rrt_params.goal = np.copy(q0)
Q_WB_d = RollPitchYaw(0.3, 0.2, 1.0).ToQuaternion()
rrt_params.goal[q_sim.get_q_u_indices_into_q()[:4]] = Q_WB_d.wxyz()
rrt_params.goal[q_sim.get_q_u_indices_into_q()[4]] = 0.15
rrt_params.goal[q_sim.get_q_u_indices_into_q()[5]] = 0.15
rrt_params.goal[q_sim.get_q_u_indices_into_q()[6]] = 0.15
rrt_params.termination_tolerance = 0.01
rrt_params.goal_as_subgoal_prob = 0.3
rrt_params.rewire = False
rrt_params.regularization = 1e-4
rrt_params.distance_metric = "local_u"
# rrt_params.distance_metric = 'global'  # If using global metric
rrt_params.global_metric = np.ones(q0.shape) * 0.1
rrt_params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
rrt_params.stepsize = 0.2
std_u = 0.1 * np.ones(19)
std_u[0:3] = 0.03
rrt_params.std_u = std_u
rrt_params.grasp_prob = 0.1

prob_rrt = IrsRrtProjection3D(rrt_params, contact_sampler, q_sim, q_sim_py)
q_vis.draw_object_triad(
    length=0.1, radius=0.001, opacity=1, path="pen/pen"
)
q_vis.draw_goal_triad(length=0.4, radius=0.01, opacity=0.7,
    X_WG=RigidTransform(Q_WB_d, np.array([0.15, 0.15, 0.15])))
prob_rrt.iterate()

d_batch = prob_rrt.calc_distance_batch(prob_rrt.rrt_params.goal)
node_id_closest = np.argmin(d_batch)
print("closest distance to goal", d_batch[node_id_closest])

# %%
prob_rrt.save_tree(f"tree_{rrt_params.max_size}.pkl")
