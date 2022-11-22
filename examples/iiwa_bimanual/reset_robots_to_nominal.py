import time
import numpy as np
import lcm

from pydrake.all import PiecewisePolynomial
from drake import lcmt_iiwa_status, lcmt_iiwa_command
from control.systems_utils import wait_for_msg
from control.controller_planar_iiwa_bimanual import kQIiwa0

q_msg = wait_for_msg(
    "IIWA_STATUS", lcmt_iiwa_status, lambda msg: msg.num_joints == 14
)

q_a_nominal = np.zeros(14)
q_a_nominal[:7] = kQIiwa0
q_a_nominal[7:] = kQIiwa0

q_knots = np.zeros((2, 14))
q_knots[0] = q_msg.joint_position_measured
q_knots[1] = q_a_nominal
duration = 10.0
q_trj = PiecewisePolynomial.FirstOrderHold([0, duration], q_knots.T)


first_status_msg_time = None


def calc_iiwa_command(channel, data):
    status_msg = lcmt_iiwa_status.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = status_msg.utime / 1e6

    t = status_msg.utime / 1e6 - first_status_msg_time

    q_ref = q_trj.value(t).squeeze()
    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = status_msg.utime
    cmd_msg.num_joints = len(q_ref)
    cmd_msg.joint_position = q_ref

    lc.publish("IIWA_COMMAND", cmd_msg.encode())


lc = lcm.LCM()

subscription = lc.subscribe("IIWA_STATUS", calc_iiwa_command)
subscription.set_queue_capacity(1)

try:
    t_start = time.time()
    while True:
        lc.handle()
        dt = time.time() - t_start
        if dt > duration + 5:
            break
    print("Done!")

except KeyboardInterrupt:
    pass
