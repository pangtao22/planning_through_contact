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

from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsRrtParams

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
qa_l_knots = np.zeros((2, nq_a))
qa_l_knots[0] = [-np.pi / 4, -np.pi / 2]

q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 2]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0.0, 0.25, 0])

q0_dict = {idx_u: q_u0,
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-0.5, 0.5], [0.3, 0.6], [-0.01, np.pi]]),
    idx_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
    idx_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])
}

#%% RRT testing
params = IrsRrtParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 2000
params.goal = np.copy(x0)
params.goal[6] = np.pi
params.termination_tolerance = 1e-2
params.subgoal_prob = 0.5

tree = IrsRrt(params)
tree.iterate()
# np.save("q_mat_large.npy", tree.q_matrix)

#%%
tree.save_tree("examples/planar_hand/data/tree_2000.pkl")

#%%

cProfile.runctx('tree.iterate()',
                 globals=globals(), locals=locals(),
                 filename='irs_rrt_profile.stat')
