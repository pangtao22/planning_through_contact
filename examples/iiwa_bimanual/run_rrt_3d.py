#!/usr/bin/env python3
from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw
from qsim.parser import QuasistaticParser

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.rrt_params import IrsRrtProjectionParams

from contact_sampler_iiwa_bimanual import IiwaBimanualContactSampler
from iiwa_bimanual_setup import *

# %% sim setup
h = 0.1

# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)
# q_parser.set_sim_params(gravity=[0, 0, -10])
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
q_dynamics.update_default_sim_params(
    forward_mode=ForwardDynamicsMode.kSocpMp, log_barrier_weight=100
)

q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()

idx_a_l = plant.GetModelInstanceByName(iiwa_l_name)
idx_a_r = plant.GetModelInstanceByName(iiwa_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
dim_u_l = plant.num_positions(idx_a_l)
dim_u_r = plant.num_positions(idx_a_r)

contact_sampler = IiwaBimanualContactSampler(q_dynamics=q_dynamics)

# initial conditions.
# q_a0_r = [-0.7, 1.8, np.pi/2, -1.4, 0, 0, 0]
# q_a0_l = [0.7, 1.8, -np.pi/2, -1.4, 0, 0, 0]

# q_a0_r = [0.0, 1.8, 0.0, 0.0, 0, 0, 0]
# q_a0_l = [0.0, 1.8, 0.0, 0.0, 0, 0, 0]

q_a0_r = [-0.7, 1.8, np.pi / 2, -1.4, 0, 0, 0]
q_a0_l = [0.7, 1.8, -np.pi / 2, -1.4, 0, 0, 0]

q_u0 = np.array([1, 0, 0, 0, 0.5, 0, 0.21])

q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)

q_dynamics.q_sim_py.update_mbp_positions_from_vector(x0)
q_dynamics.q_sim_py.draw_current_configuration()
# input()

num_joints = 7
joint_limits = {
    idx_u: np.array(
        [
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, np.pi + 0.1],
            [0, 0],
            [0.5 - 0.2, 0.5 + 0.2],
            [0.00 - 0.2, 0.00 + 0.2],
            [0.21 - 0.0, 0.21 + 0.0],
        ]
    )
}


params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.smoothing_mode = BundleMode.kFirstAnalytic
params.root_node = IrsNode(x0)
params.max_size = 5000
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(0.0, 0.0, np.pi).ToQuaternion()
p_WB_d = np.array([0.5, 0.0, 0.21])
params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
params.goal[q_dynamics.get_q_u_indices_into_x()[4:]] = p_WB_d
params.termination_tolerance = 0
params.goal_as_subgoal_prob = 0.2
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[q_dynamics.get_q_u_indices_into_x()[:4]] = 0
params.quat_metric = 5
params.distance_threshold = np.inf
std_u = 0.2 * np.ones(14)
params.regularization = 1e-3
# params.log_barrier_weight_for_bundling = 1000
params.std_u = std_u
params.stepsize = 0.3
params.rewire = False
params.distance_metric = "local_u"
params.grasp_prob = 0.1
params.h = 0.1

prob_rrt = IrsRrtProjection3D(
    params,
    contact_sampler,
    q_sim,
)
q_sim_py = prob_rrt.q_dynamics.q_sim_py
AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="drake/plant/box/box",
    length=0.4,
    radius=0.01,
    opacity=1,
)

AddTriad(
    vis=q_sim_py.viz.vis,
    name="frame",
    prefix="goal",
    length=0.4,
    radius=0.03,
    opacity=0.5,
)

q_sim_py.viz.vis["goal"].set_transform(
    RigidTransform(Q_WB_d, p_WB_d).GetAsMatrix4()
)
#
prob_rrt.iterate()
prob_rrt.save_tree("bimanual.pkl")
