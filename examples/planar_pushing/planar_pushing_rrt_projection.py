import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

from qsim.simulator import GradientMode

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from planar_pushing_setup import *
from contact_sampler import PlanarPushingContactSampler

#%% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = PlanarPushingContactSampler(q_dynamics)

q_u0 = np.array([0.0, 0.5, 0])
x0 = contact_sampler.sample_contact(q_u0)

joint_limits = {
    idx_u: np.array([[-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi]]),
    idx_a: np.array([[-2.0, 2.0], [-2.0, 2.0]]),
}

#%% RRT testing
# IrsMpc params
mpc_params = IrsMpcQuasistaticParameters()
mpc_params.Q_dict = {
    idx_u: np.array([20, 20, 10]),
    idx_a: np.array([1e-3, 1e-3])}
mpc_params.Qd_dict = {
    model: Q_i * 100 for model, Q_i in mpc_params.Q_dict.items()}
mpc_params.R_dict = {
    idx_a: 10 * np.array([1, 1])}

# IrsRrt params
params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 500
params.goal = np.copy(x0)
params.goal[1] = -0.5
params.termination_tolerance = 0.1  # used in irs_rrt.iterate() as cost threshold.
params.goal_as_subgoal_prob = 0.1
params.rewire = False
params.distance_metric = 'local_u'
params.regularization = 1e-2
# params.distance_metric = 'global'  # If using global metric
params.global_metric = q_dynamics.get_x_from_q_dict(mpc_params.Q_dict)
params.distance_threshold = 50


irs_rrt = IrsRrtProjection(params=params,
                           contact_sampler=contact_sampler)
irs_rrt.iterate()

#%%
irs_rrt.save_tree(f"data/tree_{params.max_size}_planar_pushing.pkl")
