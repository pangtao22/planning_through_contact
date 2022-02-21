import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

from qsim.simulator import GradientMode

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsRrt, IrsNode
from irs_rrt.irs_rrt_traj_opt import IrsRrtTrajOpt
from irs_rrt.rrt_params import IrsRrtParams

from planar_hand_setup import *
from contact_sampler import PlanarHandContactSampler

#%% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarHandContactSampler(q_dynamics, 0.5)

q_u0 = np.array([0.0, 0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-0.3, 0.3], [0.3, 0.5], [-0.01, np.pi]]),
    idx_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
    idx_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])}

#%% RRT testing
# IrsMpc params
mpc_params = IrsMpcQuasistaticParameters()
mpc_params.Q_dict = {
    idx_u: np.array([20, 20, 10]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3])}
mpc_params.Qd_dict = {
    model: Q_i * 100 for model, Q_i in mpc_params.Q_dict.items()}
mpc_params.R_dict = {
    idx_a_l: 10 * np.array([1, 1]),
    idx_a_r: 10 * np.array([1, 1])}
mpc_params.T = 20

mpc_params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

mpc_params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
mpc_params.std_u_initial = np.ones(dim_u) * 0.3

mpc_params.decouple_AB = True
mpc_params.num_samples = 100
mpc_params.bundle_mode = BundleMode.kFirst
mpc_params.parallel_mode = ParallelizationMode.kCppBundledB

# IrsRrt params
params = IrsRrtParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 100
params.goal = np.copy(x0)
params.goal[6] = np.pi
params.termination_tolerance = 1e-2
params.goal_as_subgoal_prob = 0.5
params.rewire = False
params.distance_metric = 'local_u'
# params.distance_metric = 'global'  # If using global metric
params.global_metric = q_dynamics.get_x_from_q_dict(mpc_params.Q_dict)


irs_rrt = IrsRrt(params)
irs_rrt.iterate()

#%%
irs_rrt.save_tree(f"tree_{params.max_size}_planar_hand.pkl")

#%%
# cProfile.runctx('tree.iterate()',
#                  globals=globals(), locals=locals(),
#                  filename='irs_rrt_profile.stat')
