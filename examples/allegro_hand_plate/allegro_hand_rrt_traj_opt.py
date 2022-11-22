import numpy as np
from allegro_hand_setup import *
from contact_sampler_allegro_plate import AllegroHandPlateContactSampler
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_traj_opt_3d import IrsRrtTrajOpt3D
from irs_rrt.rrt_params import IrsRrtTrajOptParams
from pydrake.math import RollPitchYaw
from pydrake.multibody.tree import JointIndex

# %% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandPlateContactSampler(q_dynamics)

q_u0 = np.array([1, 0, 0, 0, 0.0, -0.35, 0.07])
q = contact_sampler.sample_contact(q_u0)
q_a0 = q[q_dynamics.get_q_a_indices_into_x()]

q0_dict = {idx_u: q_u0, idx_a: q_a0}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
num_joints = 19  # The last joint is weldjoint (welded to the world)
joint_limits = {
    idx_u: np.array(
        [
            [-0.1, np.pi / 2 + 0.1],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [0, 0],
            [-0.1, 0.1],
            [-0.5, -0.3],
            [0.0, 0.3],
        ]
    ),
    idx_a: np.zeros([num_joints, 2]),
}

for i in range(num_joints):
    joint = plant.get_joint(JointIndex(i))
    if i >= 0:
        low = joint.position_lower_limits()
        upp = joint.position_upper_limits()
    joint_limits[idx_a][i, :] = [low[0], upp[0]]

joint_limits[idx_a][0, :] = joint_limits[idx_u][4, :]
joint_limits[idx_a][1, :] = joint_limits[idx_u][5, :]
joint_limits[idx_a][2, :] = joint_limits[idx_u][6, :]

# %% RRT testing
# IrsMpc params
mpc_params = IrsMpcQuasistaticParameters()
mpc_params.Q_dict = {
    idx_u: np.array([1, 1, 1, 1, 10, 10, 100]),
    idx_a: np.ones(dim_u) * 1e-3,
}
mpc_params.Qd_dict = {
    model: Q_i * 100 for model, Q_i in mpc_params.Q_dict.items()
}
mpc_params.R_dict = {idx_a: 10 * np.ones(dim_u)}
mpc_params.T = 20

mpc_params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h]
)

mpc_params.calc_std_u = lambda u_initial, i: u_initial / (i**0.8)
mpc_params.std_u_initial = np.ones(dim_u) * 0.3

mpc_params.decouple_AB = True
mpc_params.num_samples = 100
mpc_params.bundle_mode = BundleMode.kFirstRandomized
mpc_params.parallel_mode = ParallelizationMode.kCppBundledB

# IrsRrt params
params = IrsRrtTrajOptParams(q_model_path, joint_limits)
params.root_node = IrsNode(x0)
params.max_size = 50
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(np.pi / 2, 0, 0).ToQuaternion()
params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
params.goal[q_dynamics.get_q_u_indices_into_x()[5]] = -0.3
params.goal[q_dynamics.get_q_u_indices_into_x()[6]] = 0.3

params.termination_tolerance = 0  # used in irs_rrt.iterate() as cost threshold.
params.goal_as_subgoal_prob = 0.4
params.rewire = False
params.regularization = 1e-6
params.distance_metric = "local_u"
# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
params.quat_metric = 5
params.distance_threshold = np.inf
std_u = 0.2 * np.ones(19)
std_u[0:3] = 0.03
params.std_u = std_u
params.contact = False

runs = 5
iter = 1

while True:
    try:
        irs_rrt = IrsRrtTrajOpt3D(
            rrt_params=params,
            mpc_params=mpc_params,
            contact_sampler=contact_sampler,
        )
        irs_rrt.iterate()
        irs_rrt.save_tree(
            f"data/plate/trajopt/nocontact/tree_{params.max_size}_{iter}.pkl"
        )
        iter += 1

    except Exception as e:
        print(e)
        continue

    if iter > runs:
        break
