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
from irs_rrt.rrt_params import IrsRrtParams
from irs_rrt.irs_rrt_random_grasp import IrsRrtRandomGrasp

from planar_hand_setup import *
from contact_sampler import PlanarHandContactSampler2D

#%% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
q_dynamics_2d = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path_2d,
                                 internal_viz=True)
dim_x = q_dynamics_2d.dim_x
dim_u = q_dynamics_2d.dim_u
q_sim_py = q_dynamics_2d.q_sim_py
plant = q_sim_py.get_plant()
idx_a_l = plant.GetModelInstanceByName(robot_l_name)
idx_a_r = plant.GetModelInstanceByName(robot_r_name)
idx_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarHandContactSampler2D(q_dynamics, 0.5)

q_u0 = np.array([0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(np.hstack(([0.0], q_u0)))
x0 = contact_sampler.sample_contact(q_u0)

joint_limits = {
    idx_u: np.array([[0.3, 0.4], [-0.01, np.pi]]),
    idx_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
    idx_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])
}

#%% RRT testing
params = IrsRrtParams(q_model_path_2d, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 2000
params.goal = np.copy(x0)
params.goal[5] = np.pi
params.termination_tolerance = 1e-2
params.goal_as_subgoal_prob = 0.01
params.stepsize = 0.2
params.rewire = False
params.distance_metric = 'local_u'
params.regularization = 1e-6

# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.array([0.1, 0.1, 0.1, 0.1, 100.0, 1.0])


tree = IrsRrtRandomGrasp(params, contact_sampler)
tree.iterate()

#%%
tree.save_tree(f"tree_{params.max_size}_planar_hand_regrasp.pkl")

#print(tree.save_final_path())

#%%
#
# cProfile.runctx('tree.iterate()',
#                  globals=globals(), locals=locals(),
#                  filename='irs_rrt_profile.stat')