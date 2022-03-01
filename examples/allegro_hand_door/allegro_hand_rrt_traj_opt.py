import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_traj_opt_3d import IrsRrtTrajOpt
from irs_rrt.rrt_params import IrsRrtTrajOptParams

from contact_sampler_allegro_door import AllegroHandPlateContactSampler

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
q_a0 = np.array([-0.14775985, -0.07837441, -0.08875541, 0.03732591, 0.74914169,
                 0.74059597, 0.83309505, 0.62379958, 1.02520157, 0.63739027,
                 0.82612123, -0.14798914, 0.73583272, 0.61479455, 0.7005708,
                 -0.06922541, 0.78533142, 0.82942863, 0.90415436])
q_u0 = np.array([0, 0.])

q0_dict = {idx_u: q_u0, idx_a: q_a0}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
door_angle_goal = -np.pi / 12 * 5
joint_limits = {
    idx_u: np.array([[door_angle_goal, 0], [np.pi/4, np.pi / 2]]),
    idx_a: np.zeros([q_dynamics.dim_u, 2])
}
contact_sampler = AllegroHandPlateContactSampler(q_dynamics=q_dynamics)


#%% RRT testing
# IrsMpc params
mpc_params = IrsMpcQuasistaticParameters()
mpc_params.Q_dict = {
    idx_u: np.array([1, 1]),
    idx_a: np.ones(dim_u) * 1e-3}
mpc_params.Qd_dict = {
    model: Q_i * 100 for model, Q_i in mpc_params.Q_dict.items()}
mpc_params.R_dict = {idx_a: 10 * np.ones(dim_u)}
mpc_params.T = 5

mpc_params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

mpc_params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
mpc_params.std_u_initial = np.ones(dim_u) * 0.3

mpc_params.decouple_AB = True
mpc_params.num_samples = 100
mpc_params.bundle_mode = BundleMode.kFirst
mpc_params.parallel_mode = ParallelizationMode.kCppBundledB

params = IrsRrtTrajOptParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 100
params.goal = np.copy(x0)
params.goal[q_dynamics.get_q_u_indices_into_x()] = [door_angle_goal, np.pi / 2]
params.termination_tolerance = 0
params.goal_as_subgoal_prob = 0.1
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[q_dynamics.get_q_u_indices_into_x()] = [1, 1]
std_u = 0.2 * np.ones(19)
std_u[0: 3] = 0.03
params.std_u = std_u
params.rewire = False
params.regularization = 1e-6
params.distance_metric = 'local_u'
params.grasp_prob = 0.2


for i in range(2, 5):
    tree = IrsRrtTrajOpt(rrt_params=params,
                         mpc_params=mpc_params,
                         contact_sampler=contact_sampler)
    tree.iterate()

    name = "tree_traj_opt_{}_{}_{}.pkl".format(
        params.distance_metric, params.max_size, i)
    tree.save_tree(name)
