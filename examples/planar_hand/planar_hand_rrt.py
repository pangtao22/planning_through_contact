import os.path
import time
import matplotlib.pyplot as plt
import numpy as np

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

from irs_rrt.irs_rrt import IrsTree, IrsNode, IrsTreeParams

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
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]

q_robot_l_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_l_knots.T)

qa_r_knots = np.zeros((2, nq_a))
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_robot_r_traj = PiecewisePolynomial.ZeroOrderHold(
    [0, T * h], qa_r_knots.T)

q_a_traj_dict_str = {robot_l_name: q_robot_l_traj,
                     robot_r_name: q_robot_r_traj}

q_u0 = np.array([0.0, 0.35, 0])

q0_dict = {idx_u: q_u0,
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
print(x0)


#%% RRT testing
params = IrsTreeParams(q_dynamics)
params.root_node = IrsNode(x0, params)
params.root_node.value = 0.0
params.max_size = 1000
params.eps = 3.0
params.goal = np.copy(x0)
params.goal[6] = np.pi
params.termination_tolerance = 1e-3

tree = IrsTree(params)
tree.iterate()

np.save("q_mat_large.npy", tree.q_matrix)

"""
cProfile.runctx('tree.iterate()',
                 globals=globals(), locals=locals(),
                 filename='irs_rrt_profile.stat')
tree.iterate()
"""