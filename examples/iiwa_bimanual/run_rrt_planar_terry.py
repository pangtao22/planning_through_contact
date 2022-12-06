#!/usr/bin/env python3
from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad
from qsim.parser import QuasistaticParser

from irs_mpc.irs_mpc_params import BundleMode
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from iiwa_bimanual_setup import *
from contact_sampler_iiwa_bimanual_planar2 import (
    IiwaBimanualPlanarContactSampler,
)

# %% sim setup

h = 0.01

q_parser = QuasistaticParser(q_model_path_planar)
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path_planar, internal_viz=True
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

contact_sampler = IiwaBimanualPlanarContactSampler(q_dynamics)

# initial conditions.
q_a0_r = [-0.7, -1.4, 0]
q_a0_l = [0.7, 1.4, 0]
q_u0 = np.array([0.65, 0, 0])
q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)

# input()

joint_limits = {
    idx_u: np.array([[0.25, 0.75], [-0.3, 0.3], [-np.pi - 0.1, 0.1]])
}

q_u_goal = np.array([0.5, 0, -np.pi])

params = IrsRrtProjectionParams(q_model_path_planar, joint_limits)
params.smoothing_mode = BundleMode.kFirstAnalytic
params.root_node = IrsNode(x0)
params.max_size = 40000
params.goal = np.copy(x0)
params.goal[q_sim.get_q_u_indices_into_q()] = q_u_goal

params.termination_tolerance = 0.01
params.goal_as_subgoal_prob = 0.4
params.global_metric = np.ones(x0.shape) * 0.1
params.quat_metric = 5
params.distance_threshold = np.inf
std_u = 0.2 * np.ones(6)
params.regularization = 1e-3
# params.log_barrier_weight_for_bundling = 1000
params.std_u = std_u
params.stepsize = 0.14
params.rewire = False
params.distance_metric = "local_u"
params.grasp_prob = 0.3
params.h = 0.05

prob_rrt = IrsRrtProjection(
    params,
    contact_sampler,
    q_sim,
)
q_sim_py = prob_rrt.q_dynamics.q_sim_py

draw_goal_and_object_triads_2d(
    vis=q_sim_py.viz.vis, plant=q_sim.get_plant(), q_u_goal=q_u_goal
)
#
prob_rrt.iterate()
prob_rrt.save_tree("bimanual_planar.pkl")
