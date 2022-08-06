import pickle

import numpy as np
from examples.allegro_hand.sliders_active import (wait_for_status_msg,
    kAllegroCommandChannel, kAllegroStatusChannel)

from pydrake.all import (PiecewisePolynomial, DiagramBuilder,
    LcmSubscriberSystem, LcmPublisherSystem, DrakeLcm, Simulator)
from drake import lcmt_allegro_status, lcmt_allegro_command
from qsim.parser import QuasistaticParser

from allegro_hand_setup import robot_name, q_model_path_hardware
from allegro_controller_system import (
    add_controller_system_to_diagram, ControllerSystem, CommandVec2LcmSystem)

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
h = 0.3
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
t_knots += 5.0
t_knots = np.hstack([0, t_knots])

# Diagram
h_ctrl = 0.02
drake_lcm = DrakeLcm()
builder = DiagramBuilder()
ctrller_allegro, q_ref_trj, u_ref_trj = add_controller_system_to_diagram(
    builder=builder,
    t_knots=t_knots,
    u_knots_ref=u_knots_ref,
    q_knots_ref=q_knots_ref,
    h_ctrl=h_ctrl,
    q_sim=q_sim,
    closed_loop=False)

# LCM status sub
allegro_status_sub = builder.AddSystem(
    LcmSubscriberSystem.Make(
        channel=kAllegroStatusChannel,
        lcm_type=lcmt_allegro_status,
        lcm=drake_lcm))

# LCM command pub.
allegro_lcm_pub = builder.AddSystem(
    LcmPublisherSystem.Make(
        channel=kAllegroCommandChannel,
        lcm_type=lcmt_allegro_command,
        lcm=drake_lcm,
        publish_period=h_ctrl))

cmd_v2l = CommandVec2LcmSystem(q_sim)
builder.AddSystem(cmd_v2l)
builder.Connect(
    ctrller_allegro.position_cmd_output_ports[allegro_model],
    cmd_v2l.q_cmd_input_port)

builder.Connect(
    allegro_status_sub.get_output_port(0),
    cmd_v2l.status_input_port)

builder.Connect(
    cmd_v2l.cmd_output_port,
    allegro_lcm_pub.get_input_port(0))

diagram = builder.Build()
# render_system_with_graphviz(diagram)


# Run simulator.
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Make sure that the first status message read my the sliders is the real
# status of the hand.
context = simulator.get_context()
context_sub = allegro_status_sub.GetMyContextFromRoot(context)
context_sub.SetAbstractState(0, allegro_status_msg)
context_ctrller = ctrller_allegro.GetMyContextFromRoot(context)
ctrller_allegro.q_input_port.FixValue(context_ctrller, q0)

print("Running!")
simulator.AdvanceTo(np.inf)


