#!/usr/bin/env python3
import copy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from pydrake.all import (PiecewisePolynomial, RotationMatrix, AngleAxis,
                         Quaternion, RigidTransform)
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp, GradientMode, ForwardDynamicsMode

from irs_mpc.irs_mpc_params import BundleMode
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from contact_sampler_iiwa_bimanual import IiwaBimanualContactSampler

from iiwa_bimanual_setup import *

#%% sim setup
h = 0.01
T = 25  # num of time steps to simulate forward.
duration = T * h
max_iterations = 40

# quasistatic dynamical system
q_parser = QuasistaticParser(q_model_path)
# q_parser.set_sim_params(gravity=[0, 0, -10])
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
q_dynamics.update_default_sim_params(
    forward_mode=ForwardDynamicsMode.kSocpMp,
    log_barrier_weight=100)

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
q_a0_r = [0.11, 1.57, np.pi/2, 0, 0, 0, 0]
q_a0_l = [-0.09, 1.03, -np.pi/2, -0.61, -0.15, -0.06, 0]

q_u0 = np.array([1, 0, 0, 0,  0.55, 0, 0.315])

q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
x0 = q_sim.get_q_vec_from_dict(q0_dict)

num_joints = 7
joint_limits = {
    idx_u: np.array([
        [-0.1, np.pi + 0.1],[-0.1, 0.1], [-0.1, 0.1], [0, 0],
        [0.55 -0.2, 0.55 + 0.2],
        [0.00 - 0.2, 0.00 + 0.2],
        [0.315 - 0.2, 0.315 + 0.3]]),
    idx_a_l: np.zeros([num_joints, 2]),
    idx_a_r: np.zeros([num_joints, 2])
}

joint_range = 1.0
for i in range(num_joints):
    joint_limits[idx_a_l][i,0] = q_a0_l[i] - joint_range
    joint_limits[idx_a_l][i,1] = q_a0_l[i] + joint_range
    joint_limits[idx_a_r][i,0] = q_a0_r[i] - joint_range
    joint_limits[idx_a_r][i,1] = q_a0_r[i] + joint_range


params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.bundle_mode = BundleMode.kFirstAnalytic
params.root_node = IrsNode(x0)
params.max_size = 3000
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(np.pi, 0.0, 0.0).ToQuaternion()
params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
params.goal[q_dynamics.get_q_u_indices_into_x()[4:]] = np.array(
    [0.55, 0.0, 0.4])
params.termination_tolerance = 0
params.goal_as_subgoal_prob = 0.2
params.global_metric = np.ones(x0.shape) * 0.1
std_u = 0.3 * np.ones(14)
# params.regularization = 1e-3
params.std_u = std_u
params.stepsize = 0.15
params.rewire = False
params.distance_metric = 'local_u'
params.grasp_prob = 0.02

prob_rrt = IrsRrtProjection(params, contact_sampler)
prob_rrt.iterate()
prob_rrt.save_tree("bimanual.pkl")
