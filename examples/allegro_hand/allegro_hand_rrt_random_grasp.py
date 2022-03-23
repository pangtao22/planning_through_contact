import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.rrt_params import IrsRrtTrajOptParams, IrsRrtProjectionParams

from allegro_hand_setup import *
from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler

from pydrake.multibody.tree import JointIndex
from pydrake.math import RollPitchYaw

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

contact_sampler = AllegroHandContactSampler(q_dynamics)

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
    idx_u: np.array([
        [-0.1, 0.1],[-0.1, 0.1], [-0.1, np.pi + 0.1], [0, 0],
        [-0.086, -0.075], [-0.005, 0.005], [0.068, 0.075]]),
    idx_a: np.zeros([num_joints, 2])
}

for i in range(num_joints):
    joint = plant.get_joint(JointIndex(i))
    low = joint.position_lower_limits()
    upp = joint.position_upper_limits()
    joint_limits[idx_a][i, :] = [low[0], upp[0]]

#%% RRT testing
# IrsMpc params
mpc_params = IrsMpcQuasistaticParameters()
mpc_params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1]),
    idx_a: np.ones(dim_u) * 1e-3}
mpc_params.Qd_dict = {
    model: Q_i * 100 for model, Q_i in mpc_params.Q_dict.items()}
mpc_params.R_dict = {idx_a: 10 * np.ones(dim_u)}
mpc_params.T = 20

mpc_params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

mpc_params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
mpc_params.std_u_initial = np.ones(dim_u) * 0.3

mpc_params.decouple_AB = True
mpc_params.num_samples = 100
mpc_params.bundle_mode = BundleMode.kFirstRandomized
mpc_params.parallel_mode = ParallelizationMode.kCppBundledB

# IrsRrt params
params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 5000
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(0, 0, np.pi).ToQuaternion()
params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
params.termination_tolerance = 1  # used in irs_rrt.iterate() as cost threshold.
params.goal_as_subgoal_prob = 0.1
params.rewire = False
params.regularization = 1e-6
params.distance_metric = 'local_u'
# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
params.quat_metric = 5
params.distance_threshold = 3
params.stepsize = 0.3
params.std_u = 0.1
params.grasp_prob = 0.4


irs_rrt = IrsRrtProjection3D(params, contact_sampler)
irs_rrt.iterate()

#%%
irs_rrt.save_tree(f"tree_{params.max_size}_allegro_hand_random_grasp.pkl")
