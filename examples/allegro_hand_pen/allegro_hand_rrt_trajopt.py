import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_traj_opt_3d import IrsRrtTrajOpt3D
from irs_rrt.rrt_params import IrsRrtTrajOptParams

from allegro_hand_setup import *
from contact_sampler_allegro_pen import AllegroHandPenContactSampler

from pydrake.multibody.tree import JointIndex
from pydrake.all import RollPitchYaw, RotationMatrix, Quaternion

import warnings
warnings.filterwarnings('ignore')

for iter in range(5):
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

    contact_sampler = AllegroHandPenContactSampler(q_dynamics)

    q_a0 = np.array([0.0069174, -0.00107283, 0., 0.03501504,
                     0.75276565,
                     0.74146232, 0.83261002, 0.63256269, 1.02378254, 0.64089555,
                     0.82444782, -0.1438725, 0.74696812, 0.61908827, 0.70064279,
                     -0.06922541, 0.78533142, 0.82942863, 0.90415436])
    q_u0 = np.array([0.99326894, 0.00660496, -0.08931768, 0.07345429,
                     -0.08546328, 0.01222016, 0.0311])

    q0_dict = {idx_u: q_u0,
               idx_a: q_a0}

    x0 = q_dynamics.get_x_from_q_dict(q0_dict)
    num_joints = 19 # The last joint is weldjoint (welded to the world)
    joint_limits = {
        idx_u: np.array([
            [0.0, 0.5], [0.0, 0.5], [0.0, 1.2], [0, 0],
            [-0.09, 0.2], [0.0, 0.2], [0.0, 0.2]]),
        idx_a: np.zeros([num_joints, 2])
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

    #%% RRT testing
    # IrsMpc params
    mpc_params = IrsMpcQuasistaticParameters()
    cost_u = 0.1 * np.ones(dim_u)
    cost_u[0:3] = 10
    mpc_params.Q_dict = {
        idx_u: np.array([1, 1, 1, 1, 10, 10, 10]),
        idx_a: np.ones(dim_u) * 1e-3}
    mpc_params.Qd_dict = {
        model: Q_i * 10 for model, Q_i in mpc_params.Q_dict.items()}
    mpc_params.R_dict = {idx_a: cost_u}
    mpc_params.T = 20

    u_bounds_abs = np.ones(dim_u) * h
    u_bounds_abs[:3] *= 0.5
    mpc_params.u_bounds_abs = np.array([-u_bounds_abs, u_bounds_abs])

    mpc_params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
    std_u_initial = 0.1 * np.ones(19)
    std_u_initial[0:3] = 0.02
    mpc_params.std_u_initial = std_u_initial

    mpc_params.decouple_AB = True
    mpc_params.num_samples = 100
    mpc_params.bundle_mode = BundleMode.kFirst
    mpc_params.parallel_mode = ParallelizationMode.kCppBundledB

    # IrsRrt params
    params = IrsRrtTrajOptParams(q_model_path, joint_limits)
    params.root_node = IrsNode(x0)
    params.max_size = 50
    params.goal = np.copy(x0)
    Q_WB_d = RollPitchYaw(0.3, 0.2, 1.0).ToQuaternion()
    # Q_WB_d = RollPitchYaw(0, 0, 0).ToQuaternion()
    params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
    params.goal[q_dynamics.get_q_u_indices_into_x()[4]] = 0.15
    params.goal[q_dynamics.get_q_u_indices_into_x()[5]] = 0.15
    params.goal[q_dynamics.get_q_u_indices_into_x()[6]] = 0.15
    params.termination_tolerance = 0  # used in irs_rrt.iterate() as cost
    # threshold.
    params.goal_as_subgoal_prob = 0.3
    params.rewire = False
    params.regularization = 1e-4
    params.distance_metric = 'local_u'
    # params.distance_metric = 'global'  # If using global metric
    params.global_metric = np.ones(x0.shape) * 0.1
    params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
    params.quat_metric = 5
    params.stepsize = 0.2
    std_u = 0.1 * np.ones(19)
    std_u[0:3] = 0.03
    params.std_u = std_u
    params.grasp_prob = 0.1
    params.distance_threshold = 2

    irs_rrt = IrsRrtTrajOpt3D(rrt_params=params,
                              mpc_params=mpc_params,
                              contact_sampler=contact_sampler)
    irs_rrt.iterate()

    #%%
    irs_rrt.save_tree(f"data/trajopt/nocontact/tree"
                      f"_{params.max_size}_allegro_hand_random_grasp_"
                      f"{iter+1}.pkl")
