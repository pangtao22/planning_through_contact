import time
import pickle
import numpy as np
import lcm

from pydrake.all import RigidTransform
from drake import lcmt_scope, lcmt_robot_state

from plan_runner_client.calc_plan_msg import calc_joint_space_plan_msg
from plan_runner_client.zmq_client import PlanManagerZmqClient


from qsim.parser import QuasistaticParser
from control.drake_sim import (load_ref_trajectories,
                               calc_q_and_u_extended_and_t_knots)
from control.controller_planar_iiwa_bimanual import kQIiwa0, kIndices3Into7
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import (q_model_path_planar, q_model_path_cylinder)
from state_estimator import kQEstimatedChannelName

n_qa = 14
kGoalPoseChannel = "GOAL_POSE"
kStartPoseChannel = "START_POSE"

#%%
zmq_client = PlanManagerZmqClient()
lc = lcm.LCM()

#%%
def q_a_2d_to_q_a_3d(q_a_2d: np.ndarray):
    q_left_3d = np.copy(kQIiwa0)
    q_left_3d[kIndices3Into7] = q_a_2d[:3]
    q_right_3d = np.copy(kQIiwa0)
    q_right_3d[kIndices3Into7] = q_a_2d[3:]
    return np.hstack([q_left_3d, q_right_3d])

q_parser_2d = QuasistaticParser(q_model_path_planar)
q_parser_3d = QuasistaticParser(q_model_path_cylinder)
q_sim_2d = q_parser_2d.make_simulator_cpp()
q_sim_3d = q_parser_3d.make_simulator_cpp()
h_ref_knot = 1.0

file_path = "./bimanual_patched_q_and_u_trj.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict['q_trj_list']
u_knots_ref_list = trj_dict['u_trj_list']

# pick one segment for now.
idx_trj_segment = 0


#%% run_joint_space_plan
q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21)
t_transition = 10.0
indices_q_u_into_q = q_sim_2d.get_q_u_indices_into_q()

for i in range(len(u_knots_ref_list)):
    print(f"================= {i} ==================")
    q_knots_ref, u_knots_ref_2d, _ = calc_q_and_u_extended_and_t_knots(
        q_knots_ref=q_knots_ref_list[i],
        u_knots_ref=u_knots_ref_list[i],
        q_sim=q_sim_2d,
        h_ref_knot=h_ref_knot)

    u_knots_ref = np.array([q_a_2d_to_q_a_3d(u) for u in u_knots_ref_2d])

    T = len(u_knots_ref)
    t_knots = np.linspace(0, T, T + 1) * h_ref_knot

    if i == 0:
        t_knots += t_transition * 2
        t_knots = np.hstack([0, t_transition, t_knots])
        q = np.array(q_msg.value)
        q_a0 = q[q_sim_3d.get_q_a_indices_into_q()]
        u_knots_ref_extended = np.vstack([q_a0,
                                          u_knots_ref[0],
                                          u_knots_ref])
    else:
        u_knots_ref_extended = u_knots_ref

    # publish start and goal.
    pose_msg = lcmt_robot_state()
    pose_msg.num_joints = 3
    pose_msg.joint_name = ["x", "y", "theta"]
    pose_msg.joint_position = q_knots_ref[0, indices_q_u_into_q]
    lc.publish(kStartPoseChannel, pose_msg.encode())

    pose_msg.joint_position = q_knots_ref[-1, indices_q_u_into_q]
    lc.publish(kGoalPoseChannel, pose_msg.encode())

    plan_msg = calc_joint_space_plan_msg(t_knots, u_knots_ref_extended)
    zmq_client.send_plan(plan_msg)
    zmq_client.wait_for_plan_to_finish()



