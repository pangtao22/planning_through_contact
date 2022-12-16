import pickle
import time

import matplotlib.pyplot as plt
import lcm
import numpy as np
from drake import lcmt_iiwa_command, lcmt_scope, lcmt_robot_state
from pydrake.all import PiecewisePolynomial
from qsim.parser import QuasistaticParser

from control.controller_planar_iiwa_bimanual import (
    Controller,
    kQIiwa0,
    kIndices3Into7,
    IiwaBimanualPlanarControllerSystem,
)
from control.drake_sim import calc_q_and_u_extended_and_t_knots
from control.systems_utils import wait_for_msg
from iiwa_bimanual_setup import (
    q_model_path_planar,
    q_model_path_cylinder,
    controller_params_2d,
)
from state_estimator import kQEstimatedChannelName


kGoalPoseChannel = "GOAL_POSE"
kStartPoseChannel = "START_POSE"

# %%


def q_a_2d_to_q_a_3d(q_a_2d: np.ndarray):
    q_left_3d = np.copy(kQIiwa0)
    q_left_3d[kIndices3Into7] = q_a_2d[:3]
    q_right_3d = np.copy(kQIiwa0)
    q_right_3d[kIndices3Into7] = q_a_2d[3:]
    return np.hstack([q_left_3d, q_right_3d])


def q_a_3d_to_q_a_2d(q_a_3d: np.ndarray):
    q_left_2d = q_a_3d[:7][kIndices3Into7]
    q_right_2d = q_a_3d[7:][kIndices3Into7]
    return np.hstack([q_left_2d, q_right_2d])


q_parser_2d = QuasistaticParser(q_model_path_planar)
q_parser_3d = QuasistaticParser(q_model_path_cylinder)
q_sim_2d = q_parser_2d.make_simulator_cpp()
q_sim_3d = q_parser_3d.make_simulator_cpp()
h_ref_knot = 0.5

file_path = "bimanual_optimized_q_and_u_trj_1.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

# pick one segment for now.
idx_trj_segment = 2
q_knots_ref_2d = q_knots_ref_list[idx_trj_segment]
_, u_knots_ref_2d, t_knots = calc_q_and_u_extended_and_t_knots(
    q_knots_ref=q_knots_ref_list[idx_trj_segment],
    u_knots_ref=u_knots_ref_list[idx_trj_segment],
    u_knot_ref_start=q_knots_ref_2d[0, q_sim_2d.get_q_a_indices_into_q()],
    v_limit=0.05,
)

u_ref_2d_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref_2d.T)


# 3d trajectories.
u_knots_ref_3d = np.array([q_a_2d_to_q_a_3d(u) for u in u_knots_ref_2d])
q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21
)

t_transition = 10.0
t_knots += t_transition * 2
t_knots = np.hstack([0, t_transition, t_knots])
q = np.array(q_msg.value)
q_a0 = q[q_sim_3d.get_q_a_indices_into_q()]
u_knots_ref_3d = np.vstack([q_a0, u_knots_ref_3d[0], u_knots_ref_3d])
u_ref_3d_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref_3d.T)

# LCM callback.
first_status_msg_time = None
arc_length_list = []
oh_no_times = []


def calc_iiwa_command(channel, data):
    q_msg = lcmt_scope.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = q_msg.utime / 1e6

    t = q_msg.utime / 1e6 - first_status_msg_time

    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = q_msg.utime
    cmd_msg.joint_position = u_ref_3d_trj.value(t).squeeze()
    cmd_msg.num_joints = len(cmd_msg.joint_position)

    lc.publish("IIWA_COMMAND", cmd_msg.encode())


lc = lcm.LCM()

subscription = lc.subscribe(kQEstimatedChannelName, calc_iiwa_command)
subscription.set_queue_capacity(1)

indices_q_u_into_q = q_sim_2d.get_q_u_indices_into_q()
try:
    # publish start and goal.
    pose_msg = lcmt_robot_state()
    pose_msg.num_joints = 3
    pose_msg.joint_name = ["x", "y", "theta"]
    pose_msg.joint_position = q_knots_ref_2d[0, indices_q_u_into_q]
    lc.publish(kStartPoseChannel, pose_msg.encode())

    pose_msg.joint_position = q_knots_ref_2d[-1, indices_q_u_into_q]
    lc.publish(kGoalPoseChannel, pose_msg.encode())

    # Run the plan!
    t_start = time.time()
    while True:
        lc.handle()
        dt = time.time() - t_start
        if dt > t_knots[-1] + 10:
            break
    print("Done!")

except KeyboardInterrupt:
    pass
