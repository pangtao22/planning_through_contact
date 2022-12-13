import time
import pickle
import numpy as np
import lcm

from pydrake.all import RigidTransform
from drake import lcmt_scope, lcmt_robot_state

from plan_runner_client.calc_plan_msg import calc_joint_space_plan_msg
from plan_runner_client.zmq_client import PlanManagerZmqClient


from qsim.parser import QuasistaticParser
from control.drake_sim import (
    load_ref_trajectories,
    calc_q_and_u_extended_and_t_knots,
)
from control.controller_planar_iiwa_bimanual import kQIiwa0, kIndices3Into7
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import q_model_path_planar, q_model_path_cylinder
from state_estimator import kQEstimatedChannelName

n_qa = 14
kGoalPoseChannel = "GOAL_POSE"
kStartPoseChannel = "START_POSE"


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


file_path = "./bimanual_patched_q_and_u_trj.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

#%%
for i in range(len(u_knots_ref_list) - 1):
    if i == 0:
        u_knot_ref_start = None
    else:
        u_knot_ref_start = u_knots_ref_list[i - 1][-1]

    _, u_knots_ref_2d_i, t_knots = calc_q_and_u_extended_and_t_knots(
        q_knots_ref=q_knots_ref_list[i],
        u_knots_ref=u_knots_ref_list[i],
        u_knot_ref_start=u_knot_ref_start,
        v_limit=0.1,
    )

    _, u_knots_ref_2d_i1, _ = calc_q_and_u_extended_and_t_knots(
        q_knots_ref=q_knots_ref_list[i + 1],
        u_knots_ref=u_knots_ref_list[i + 1],
        u_knot_ref_start=u_knots_ref_list[i][-1],
        v_limit=0.1,
    )

    print(i, np.linalg.norm(u_knots_ref_2d_i[-1] - u_knots_ref_2d_i1[0]))
    print(t_knots)


#%% run_joint_space_plan
zmq_client = PlanManagerZmqClient()
lc = lcm.LCM()

#%%
q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21
)
t_transition = 10.0
indices_q_u_into_q = q_sim_2d.get_q_u_indices_into_q()

for i in range(len(u_knots_ref_list)):
    print(f"================= {i} ==================")
    if i == 0:
        u_knot_ref_start = None
    else:
        u_knot_ref_start = u_knots_ref_list[i - 1][-1]

    if i % 2 == 0:
        # A contact segment.
        v_limit = 0.1
    else:
        # A collision-free segment.
        v_limit = 0.4

    q_knots_ref_2d = q_knots_ref_list[i]
    _, u_knots_ref_2d, t_knots = calc_q_and_u_extended_and_t_knots(
        q_knots_ref=q_knots_ref_list[i],
        u_knots_ref=u_knots_ref_list[i],
        u_knot_ref_start=u_knot_ref_start,
        v_limit=v_limit,
    )

    u_knots_ref = np.array([q_a_2d_to_q_a_3d(u) for u in u_knots_ref_2d])

    if i == 0:
        t_knots += t_transition * 2
        t_knots = np.hstack([0, t_transition, t_knots])
        q = np.array(q_msg.value)
        q_a0 = q[q_sim_3d.get_q_a_indices_into_q()]
        u_knots_ref_extended = np.vstack([q_a0, u_knots_ref[0], u_knots_ref])
    else:
        u_knots_ref_extended = u_knots_ref

    # publish start and goal.
    pose_msg = lcmt_robot_state()
    pose_msg.num_joints = 3
    pose_msg.joint_name = ["x", "y", "theta"]
    pose_msg.joint_position = q_knots_ref_2d[0, indices_q_u_into_q]
    lc.publish(kStartPoseChannel, pose_msg.encode())

    pose_msg.joint_position = q_knots_ref_2d[-1, indices_q_u_into_q]
    lc.publish(kGoalPoseChannel, pose_msg.encode())

    plan_msg = calc_joint_space_plan_msg(t_knots, u_knots_ref_extended)
    zmq_client.send_plan(plan_msg)
    zmq_client.wait_for_plan_to_finish()
