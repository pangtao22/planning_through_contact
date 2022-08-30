import time

import numpy as np

import lcm
from pydrake.all import PiecewisePolynomial
from drake import lcmt_iiwa_command, lcmt_iiwa_status, lcmt_scope

from qsim.parser import QuasistaticParser

from control.drake_sim import load_ref_trajectories
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import q_model_path
from state_estimator import kQEstimatedChannelName

#%%
q_parser = QuasistaticParser(q_model_path)
q_sim = q_parser.make_simulator_cpp(has_objects=False)

h_ref_knot = 0.5
q_knots_ref, u_knots_ref, t_knots = load_ref_trajectories(
    file_path="hand_trj.pkl", h_ref_knot=h_ref_knot, q_sim=q_sim)

q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21)

t_knots += 10.0
t_knots = np.hstack([0, t_knots])
q = np.array(q_msg.value)
q_a0 = q[q_sim.get_q_a_indices_into_q()]
u_knots_ref = np.vstack([q_a0, u_knots_ref])
u_ref_trj = PiecewisePolynomial.FirstOrderHold(
    t_knots, u_knots_ref.T)


first_status_msg_time = None


def calc_iiwa_command(channel, data):
    q_msg = lcmt_scope.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = q_msg.utime / 1e6

    t = q_msg.utime / 1e6 - first_status_msg_time
    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = q_msg.utime
    cmd_msg.num_joints = 14
    cmd_msg.joint_position = u_ref_trj.value(t).squeeze()

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
