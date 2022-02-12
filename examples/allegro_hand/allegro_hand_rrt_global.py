import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

from pydrake.all import PiecewisePolynomial
from pydrake.math import RollPitchYaw

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import (
    QuasistaticDynamicsParallel)
from irs_mpc.irs_mpc_quasistatic import (
    IrsMpcQuasistatic)
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode, IrsRrt
from irs_rrt.irs_rrt_global import IrsRrtGlobalAllegro
from irs_rrt.rrt_params import IrsRrtGlobalParams
from pydrake.multibody.tree import JointIndex

from allegro_hand_setup import *

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
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)


# trajectory and initial conditions.
q_a0 = np.array([0.03501504, 0.75276565, 0.74146232, 0.83261002, 0.63256269,
                 1.02378254, 0.64089555, 0.82444782, -0.1438725, 0.74696812,
                 0.61908827, 0.70064279, -0.06922541, 0.78533142, 0.82942863,
                 0.90415436])


q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])

q0_dict = {idx_u: q_u0,
           idx_a: q_a0}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
num_joints = plant.num_joints() - 1 # The last joint is weldjoint (welded to the world)
joint_limits = {
    # idx_u: np.array([[0, 0],[0, 0], [0, 0], [0, 0], [-0.16, -0.02], [-0.06, 0.06], [0.05, 0.09]]),
    idx_u: np.array([[0, 0],[0, 0], [0, 0], [0, 0], [-0.081, -0.081], [0.001, 0.001], [0.071, 0.071]]),
    idx_a: np.zeros([num_joints, 2])
}

for i in range(num_joints):
    joint = plant.get_joint(JointIndex(i))
    low = joint.position_lower_limits()
    upp = joint.position_upper_limits()
    joint_limits[idx_a][i, :] = [low, upp]

#%% RRT testing
params = IrsRrtGlobalParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 300 #0
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(0, 0, np.pi / 8).ToQuaternion()
params.goal[:4] = Q_WB_d.wxyz()
params.termination_tolerance = 1e-2
params.subgoal_prob = 0.5
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
params.quat_metric = 5

tree = IrsRrtGlobalAllegro(params, num_joints)
tree.iterate()
# np.save("q_mat_large.npy", tree.q_matrix)

#%%
tree.save_tree("examples/allegro_hand/data/tree_{}_global_y.pkl".format(params.max_size))

#%%

cProfile.runctx('tree.iterate()',
                 globals=globals(), locals=locals(),
                 filename='irs_rrt_profile.stat')
