import time
import pickle
import numpy as np

import lcm
from pydrake.all import PiecewisePolynomial
from drake import lcmt_iiwa_command, lcmt_iiwa_status, lcmt_scope

from qsim.parser import QuasistaticParser

from control.drake_sim import (load_ref_trajectories,
                               calc_u_extended_and_t_knots)
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import (q_model_path_planar, q_model_path_cylinder)
from state_estimator import kQEstimatedChannelName

from control.controller_system import Controller
from control.controller_planar_iiwa_bimanual import kQIiwa0, kIndices3Into7
#%%


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

file_path = "./hand_optimized_q_and_u_trj.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict['q_trj_list']
u_knots_ref_list = trj_dict['u_trj_list']

# pick one segment for now.
idx_trj_segment = 0
q_knots_ref, u_knots_ref_2d, t_knots = calc_u_extended_and_t_knots(
    q_knots_ref=q_knots_ref_list[idx_trj_segment],
    u_knots_ref=u_knots_ref_list[idx_trj_segment], u_knots_ref_start=, v_limit=)

u_knots_ref = np.array([q_a_2d_to_q_a_3d(u) for u in u_knots_ref_2d])


q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21)

t_transition = 10.0
t_knots += t_transition * 2
t_knots = np.hstack([0, t_transition, t_knots])
q = np.array(q_msg.value)
q_a0 = q[q_sim_3d.get_q_a_indices_into_q()]
u_knots_ref = np.vstack([q_a0,
                         u_knots_ref[0],
                         u_knots_ref])
u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)


# LCM callback.
first_status_msg_time = None


def calc_iiwa_command(channel, data):
    q_msg = lcmt_scope.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = q_msg.utime / 1e6

    t = q_msg.utime / 1e6 - first_status_msg_time

    u_nominal_3d = u_ref_trj.value(t).squeeze()

    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = q_msg.utime
    cmd_msg.num_joints = len(u_nominal_3d)
    cmd_msg.joint_position = u_nominal_3d

    lc.publish("IIWA_COMMAND", cmd_msg.encode())


lc = lcm.LCM()

subscription = lc.subscribe(kQEstimatedChannelName, calc_iiwa_command)
subscription.set_queue_capacity(1)

try:
    t_start = time.time()
    while True:
        lc.handle()
        dt = time.time() - t_start
        if dt > t_knots[-1] + 5:
            break
    print("Done!")

except KeyboardInterrupt:
    pass
