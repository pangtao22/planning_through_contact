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

from irs_rrt.irs_rrt import IrsNode, IrsRrt
from irs_rrt.irs_rrt_global import IrsRrtGlobal
from irs_rrt.rrt_params import IrsRrtGlobalParams

from planar_hand_setup import *

np.set_printoptions(precision=3, suppress=True)

#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h


# quasistatic dynamical system
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


# trajectory and initial conditions.
nq_a = 2
qa_l_knots = [-np.pi / 4, -np.pi / 4]
qa_r_knots = [np.pi / 4, np.pi / 4]
q_u0 = np.array([0.0, 0.35, 0])

q0_dict = {idx_u: q_u0,
           idx_a_l: qa_l_knots,
           idx_a_r: qa_r_knots}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-0.5, 0.5], [0.3, 0.6], [-0.01, np.pi]]),
    idx_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
    idx_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])
}

#%% RRT testing
params = IrsRrtGlobalParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 2000
params.goal = np.copy(x0)
params.goal[6] = np.pi
params.termination_tolerance = 1e-2
params.subgoal_prob = 0.5
params.global_metric = np.array([0.001, 0.001, 0.001, 0.001, 5.0, 5.0, 3.0])

tree = IrsRrtGlobal(params)
tree.iterate()
# np.save("q_mat_large.npy", tree.q_matrix)

#%%
tree.save_tree("examples/planar_hand/data/tree_2000_global.pkl")
#tree.save_final_path("examples/planar_hand/data/path_2000_global.pkl")

#%%
"""
cProfile.runctx('tree.iterate()',
                 globals=globals(), locals=locals(),
                 filename='irs_rrt_profile.stat')
"""