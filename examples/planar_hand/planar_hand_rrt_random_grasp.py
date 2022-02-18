import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

from pydrake.all import PiecewisePolynomial

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import (
    QuasistaticDynamicsParallel)
from irs_mpc.irs_mpc_quasistatic import (
    IrsMpcQuasistatic)
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsRrt, IrsNode
from irs_rrt.rrt_params import IrsRrtRandomGraspParams
from irs_rrt.irs_rrt_random_grasp import IrsRrtRandomGrasp

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
contact_sampler = PlanarHandContactSampler(q_dynamics, pinch_prob=0.5)

q_u0 = np.array([0.0, 0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-0.3, 0.3], [0.3, 0.5], [-0.01, np.pi]]),
    idx_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
    idx_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])
}

#%% RRT testing
params = IrsRrtRandomGraspParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 2000
params.goal = np.copy(x0)
params.goal[6] = np.pi
params.termination_tolerance = 1e-2
params.goal_as_subgoal_prob = 0.1
params.rewire = False
params.grasp_prob = 0.2
params.distance_threshold = 50
params.distance_metric = 'local_u'

# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 1.0])

irs_rrt = IrsRrtRandomGrasp(params, contact_sampler)
irs_rrt.iterate()

#%%
irs_rrt.save_tree(f"tree_{params.max_size}_planar_hand_random_grasp.pkl")


#%%
#
# cProfile.runctx('tree.iterate()',
#                  globals=globals(), locals=locals(),
#                  filename='irs_rrt_profile.stat')
