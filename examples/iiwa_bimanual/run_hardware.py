import time

import numpy as np

import lcm
from pydrake.all import PiecewisePolynomial
from drake import lcmt_iiwa_command, lcmt_iiwa_status, lcmt_scope

from qsim.parser import QuasistaticParser

from control.drake_sim import load_ref_trajectories
from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import q_model_path, controller_params_3d
from state_estimator import kQEstimatedChannelName

from control.controller_system import Controller

#%%
q_parser = QuasistaticParser(q_model_path)
q_sim = q_parser.make_simulator_cpp(has_objects=True)

h_ref_knot = 1.0
q_knots_ref, u_knots_ref, t_knots = load_ref_trajectories(
    file_path="hand_trj.pkl", h_ref_knot=h_ref_knot, q_sim=q_sim
)

q_msg = wait_for_msg(
    kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size == 21
)

t_transition = 10.0
t_knots += t_transition
t_knots = np.hstack([0, t_knots])
q = np.array(q_msg.value)
q_a0 = q[q_sim.get_q_a_indices_into_q()]
u_knots_ref = np.vstack([q_a0, u_knots_ref])
q_knots_ref = np.vstack([q, q_knots_ref])
u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)
q_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_ref.T)

# Controller
controller_params_3d.control_period = 0.005
R_diag = np.zeros(14)
R_diag[:7] = [1, 1, 0.5, 0.5, 0.5, 0.5, 0.2]
R_diag[7:] = R_diag[:7]
controller_params_3d.R = np.diag(5 * R_diag)
controller = Controller(q_sim=q_sim, controller_params=controller_params_3d)


# LCM callback.
first_status_msg_time = None


def calc_iiwa_command(channel, data):
    q_msg = lcmt_scope.decode(data)
    global first_status_msg_time
    if first_status_msg_time is None:
        first_status_msg_time = q_msg.utime / 1e6

    t = q_msg.utime / 1e6 - first_status_msg_time

    u_nominal = u_ref_trj.value(t).squeeze()
    if t < t_transition:
        u = u_nominal
    else:
        q_goal = q_ref_trj.value(t).squeeze()
        u_goal = u_ref_trj.value(t).squeeze()
        q = np.array(q_msg.value)
        q_nominal, u_nominal = controller.find_closest_on_nominal_path(q)
        u = controller.calc_u(q_nominal, u_nominal, q)
        # u = u_nominal

    cmd_msg = lcmt_iiwa_command()
    cmd_msg.utime = q_msg.utime
    cmd_msg.num_joints = len(u)
    cmd_msg.joint_position = u

    lc.publish("IIWA_COMMAND", cmd_msg.encode())


lc = lcm.LCM()

subscription = lc.subscribe(kQEstimatedChannelName, calc_iiwa_command)
subscription.set_queue_capacity(1)

try:
    t_start = time.time()
    while True:
        lc.handle()
    #     dt = time.time() - t_start
    #     if dt > t_knots[-1] + 5:
    #         break
    # print("Done!")

except KeyboardInterrupt:
    pass
