import time

import numpy as np

import lcm
from pydrake.all import PiecewisePolynomial
from drake import lcmt_iiwa_command, lcmt_iiwa_status

from qsim.parser import QuasistaticParser

from control.drake_sim import load_ref_trajectories
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import q_model_path


#%%
q_parser = QuasistaticParser(q_model_path)
q_sim = q_parser.make_simulator_cpp(has_objects=False)

h_ref_knot = 0.5
q_knots_ref, u_knots_ref, t_knots = load_ref_trajectories(
    file_path="hand_trj.pkl", h_ref_knot=h_ref_knot, q_sim=q_sim)

iiwa_status_msg = wait_for_msg(
    "IIWA_STATUS", lcmt_iiwa_status, lambda msg: msg.num_joints == 14)

t_knots += 10.0
t_knots = np.hstack([0, t_knots])
u_knots_ref = np.vstack([iiwa_status_msg.joint_position_measured, u_knots_ref])
u_ref_trj = PiecewisePolynomial.FirstOrderHold(
    t_knots, u_knots_ref.T)


first_status_msg_time = None


def calc_iiwa_command(channel, data):
    status_msg = lcmt_iiwa_status.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = status_msg.utime / 1e6

    t = status_msg.utime / 1e6 - first_status_msg_time
    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = status_msg.utime
    cmd_msg.num_joints = status_msg.num_joints
    cmd_msg.joint_position = u_ref_trj.value(t).squeeze()

    lc.publish("IIWA_COMMAND", cmd_msg.encode())


lc = lcm.LCM()

subscription = lc.subscribe("IIWA_STATUS", calc_iiwa_command)
subscription.set_queue_capacity(1)

try:
    while True:
        lc.handle()

except KeyboardInterrupt:
    pass
