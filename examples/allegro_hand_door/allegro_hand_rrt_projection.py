import os.path
import time
import matplotlib.pyplot as plt
import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from contact_sampler_allegro_door import AllegroHandPlateContactSampler
from qsim_cpp import ForwardDynamicsMode

from allegro_hand_setup import *

np.set_printoptions(precision=3, suppress=True)

#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h


# quasistatic dynamical system
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
q_dynamics.update_default_sim_params(
    forward_mode=ForwardDynamicsMode.kSocpMp, log_barrier_weight=100
)

dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandPlateContactSampler(q_dynamics=q_dynamics)

# trajectory and initial conditions.
q_a0 = np.array(
    [
        -0.14775985,
        -0.07837441,
        -0.08875541,
        0.03732591,
        0.74914169,
        0.74059597,
        0.83309505,
        0.62379958,
        1.02520157,
        0.63739027,
        0.82612123,
        -0.14798914,
        0.73583272,
        0.61479455,
        0.7005708,
        -0.06922541,
        0.78533142,
        0.82942863,
        0.90415436,
    ]
)
q_u0 = np.array([0, 0.0])
x0 = contact_sampler.sample_contact(q_u0)

door_angle_goal = -np.pi / 12 * 5
joint_limits = {
    idx_u: np.array([[door_angle_goal, 0], [np.pi / 4, np.pi / 2]]),
    idx_a: np.zeros([q_dynamics.dim_u, 2]),
}


#%% RRT testing
params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.smoothing_mode = BundleMode.kFirstAnalytic
params.root_node = IrsNode(x0)
params.max_size = 1000
params.goal = np.copy(x0)
params.goal[q_dynamics.get_q_u_indices_into_x()] = [door_angle_goal, np.pi / 2]
params.termination_tolerance = 0
params.goal_as_subgoal_prob = 0.1
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[q_dynamics.get_q_u_indices_into_x()] = [1, 1]
std_u = 0.2 * np.ones(19)
std_u[0:3] = 0.02
# params.regularization = 1e-3
params.std_u = std_u
params.stepsize = 0.2
params.rewire = False
params.distance_metric = "local_u"
params.grasp_prob = 0.1


for i in range(5):
    prob_rrt = IrsRrtProjection(params, contact_sampler, q_sim, )
    prob_rrt.iterate()

    prob_rrt.save_tree(
        os.path.join(data_folder, "analytic", f"tree_{params.max_size}_{i}.pkl")
    )
