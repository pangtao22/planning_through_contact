import pickle

import numpy as np
from examples.allegro_hand.sliders_active import (wait_for_status_msg,
    kAllegroCommandChannel, kAllegroStatusChannel)

from pydrake.all import PiecewisePolynomial, DiagramBuilder
from drake import lcmt_allegro_status, lcmt_allegro_command
from qsim.parser import QuasistaticParser

from allegro_hand_setup import robot_name, q_model_path_hardware



def build_diagram():
    pass



q_parser = QuasistaticParser(q_model_path_hardware)
q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()
allegro_model = plant.GetModelInstanceByName(robot_name)
joint_limits = q_sim.get_actuated_joint_limits()
lower_limits = joint_limits[allegro_model]["lower"]
upper_limits = joint_limits[allegro_model]["upper"]

# 1. Read current hand configuration.
allegro_status_msg = wait_for_status_msg()
q_a0 = np.clip(
    allegro_status_msg.joint_position_measured, lower_limits, upper_limits)

#%%
h = 0.2
with open("hand_trj.pkl", "rb") as f:
    trj_dict = pickle.load(f)
q_knots_ref = trj_dict["x_trj"]
u_knots_ref = trj_dict["u_trj"]
T = len(u_knots_ref)
t_knots = np.linspace(0, T, T + 1) * h

q_a_start = q_knots_ref[0, q_sim.get_q_a_indices_into_q()]
q0 = np.copy(q_knots_ref[0])
q0[q_sim.get_q_a_indices_into_q()] = q_a0


q_knots_ref = np.vstack([q0, q_knots_ref])
u_knots_ref = np.vstack([q_a0, q_a_start, u_knots_ref])

# Move from q0 to q_knots_ref[0].
t_knots += 2.0
t_knots = np.hstack([0, t_knots])


