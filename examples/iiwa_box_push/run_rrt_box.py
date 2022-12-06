#!/usr/bin/env python3
import numpy as np

from pydrake.math import RollPitchYaw
from qsim.parser import QuasistaticParser

from pydrake.all import RigidTransform

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode

from contact_sampler_iiwa_ik import ContactSamplerBoxIK
from iiwa_box_setup import *

# %% sim setup
# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path_no_ground)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = ContactSamplerBoxIK()

# initial iiwa pose
q_u0 = np.array([0.75, 0.0, 0.0])
q_a0 = np.array([0.0, 1.157, 0.0, -1.819, 0.0, -0.976, 0.0])

q0_dict = {idx_a: q_a0, idx_u: q_u0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)

num_joints = 7
joint_limits = {
    idx_u: np.array(
        [
            [q_u0[0] - 0.1, q_u0[0] + 0.1],
            [q_u0[1] - 0.1, q_u0[1] + 0.1],
            [-0.1, np.pi + 0.1],
        ]
    ),
    idx_a: np.zeros([num_joints, 2]),
}

rrt_params = IrsRrtProjectionParams(q_model_path_no_ground, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(x0)
rrt_params.max_size = 2000

# Goal
rrt_params.goal = np.copy(x0)
rrt_params.goal[q_sim.get_q_u_indices_into_q()] = [q_u0[0], q_u0[1], np.pi / 4]

# Set weights.
rrt_params.global_metric = np.ones(x0.shape) * 0.1

# RRT parameters.
rrt_params.termination_tolerance = 0.01
rrt_params.goal_as_subgoal_prob = 0.2
rrt_params.stepsize = 0.1
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.2
rrt_params.h = 0.02
rrt_params.regularization = 1e-3

prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)

q_vis.draw_object_triad(length=0.4, radius=0.01, opacity=1, path="box/box")

q_vis.draw_goal_triad(
    length=0.4,
    radius=0.02,
    opacity=0.5,
    X_WG=RigidTransform(
        RollPitchYaw(0, 0, rrt_params.goal[2]),
        [rrt_params.goal[0], rrt_params.goal[1], 0],
    ),
)

prob_rrt.iterate()
prob_rrt.save_tree("tree_box_push.pkl")
