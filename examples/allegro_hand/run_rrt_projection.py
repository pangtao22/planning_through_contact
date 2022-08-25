import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection_3d import IrsRrtProjection3D
from irs_rrt.rrt_params import IrsRrtTrajOptParams, IrsRrtProjectionParams

from qsim_cpp import ForwardDynamicsMode
from allegro_hand_setup import *
from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler

from pydrake.multibody.tree import JointIndex
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

#%% quasistatic dynamical system
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
q_dynamics.update_default_sim_params(
    forward_mode=ForwardDynamicsMode.kSocpMp,
    log_barrier_weight=200)

dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py
q_sim = q_dynamics.q_sim
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

contact_sampler = AllegroHandContactSampler(q_dynamics)

q_a0 = np.array([0.03501504, 0.75276565, 0.74146232, 0.83261002,
                 -0.1438725, 0.74696812, 0.61908827, 0.70064279,
                 -0.06922541, 0.78533142, 0.82942863, 0.90415436,
                 0.63256269, 1.02378254, 0.64089555, 0.82444782])


q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
x0 = contact_sampler.sample_contact(q_u0)

num_joints = q_sim.num_actuated_dofs()
joint_limits = {
    # The first four elements correspond to quaternions. However, we are being
    # a little hacky and interpreting the first three elements as rpy here.
    # TODO(terry-suh): this doesn't seem to be consistent with the way joint
    # limits are defined for allegro_hand_traj_opt, since IrsRrtProjection3D
    # and IrsRrtTrajopt3D have different sample_subgoal functions. Should make
    # this slightly more consistent.
    idx_u: np.array([
        [-0.1, 0.1], [-0.1, 0.1], [-0.1, np.pi + 0.1], [0, 0],
        [-0.086, -0.075], [-0.005, 0.005], [0.068, 0.075]]),
    idx_a: np.zeros([num_joints, 2])
}

q_a_limits_dict = q_sim.get_actuated_joint_limits()
joint_limits[idx_a][:, 0] = q_a_limits_dict[idx_a]["lower"]
joint_limits[idx_a][:, 0] = q_a_limits_dict[idx_a]["upper"]


#%% RRT testing
# IrsRrt params
params = IrsRrtProjectionParams(q_model_path, joint_limits)
params.bundle_mode = BundleMode.kFirstAnalytic
params.root_node = IrsNode(x0)
params.max_size = 1000
params.goal = np.copy(x0)
Q_WB_d = RollPitchYaw(0, 0, np.pi).ToQuaternion()
params.goal[q_dynamics.get_q_u_indices_into_x()[:4]] = Q_WB_d.wxyz()
params.termination_tolerance = 0.01  # used in irs_rrt.iterate() as cost
# threshold.
params.goal_as_subgoal_prob = 0.3
params.rewire = False
params.regularization = 1e-6
params.distance_metric = 'local_u'
# params.distance_metric = 'global'  # If using global metric
params.global_metric = np.ones(x0.shape) * 0.1
params.global_metric[num_joints:] = [0, 0, 0, 0, 1, 1, 1]
params.quat_metric = 5
params.distance_threshold = np.inf
params.stepsize = 0.2
params.std_u = 0.1
params.grasp_prob = 0.2
params.h = 0.1

#%% draw the goals
for i in range(5):
    prob_rrt = IrsRrtProjection3D(params, contact_sampler)
    AddTriad(
        vis=q_dynamics.q_sim_py.viz.vis,
        name='frame',
        prefix='drake/plant/sphere/sphere',
        length=0.1,
        radius=0.001,
        opacity=1)

    prob_rrt.iterate()

    d_batch = prob_rrt.calc_distance_batch(prob_rrt.params.goal)
    node_id_closest = np.argmin(d_batch)
    print("closest distance to goal", d_batch[node_id_closest])

    #%%
    prob_rrt.save_tree(os.path.join(
        data_folder,
        "randomized",
        f"tree_{params.max_size}_{i}.pkl"))
