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
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsRrtParams

from planar_pushing_setup import *


#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h


# quasistatic dynamical system
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)


# trajectory and initial conditions.
nq_a = 2
qa_knots = np.array([0.0, -0.1])
q_u0 = np.array([0.0, 0.5, 0])

q0_dict = {idx_u: q_u0, idx_a: qa_knots}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)

joint_limits = {
    idx_u: np.array([[-2.0, 2.0], [-2.0, 2.0], [-np.pi, np.pi]]),
    idx_a: np.array([[-2.0, 2.0], [-2.0, 2.0]]),
}

#%% RRT testing
params = IrsRrtParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 50
params.goal = np.copy(x0)
params.goal[1] = 0.1
params.goal[3] = -0.5
params.termination_tolerance = 1e-2
params.goal_as_subgoal_prob = 0.5

tree = IrsRrt(params)
tree.iterate()
# np.save("q_mat_large.npy", tree.q_matrix)

#%%
tree.save_tree("tree_50_pushing.pkl")

#%%
# cProfile.runctx('tree.iterate()',
#                  globals=globals(), locals=locals(),
#                  filename='irs_rrt_profile.stat')
