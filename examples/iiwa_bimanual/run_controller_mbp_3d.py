import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Quaternion, AngleAxis, Simulator, AddTriad
from qsim.parser import QuasistaticParser
from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant,
)

from control.drake_sim import (
    calc_q_and_u_extended_and_t_knots,
    make_controller_mbp_diagram,
)
from control.systems_utils import render_system_with_graphviz
from iiwa_bimanual_setup import (
    q_model_path,
    iiwa_l_name,
    iiwa_r_name,
    controller_params_3d,
)

# %%
h_ref_knot = 0.2
h_ctrl = 0.005
R_diag = np.zeros(14)
R_diag[:7] = [1, 1, 0.5, 0.5, 0.5, 0.5, 0.2]
R_diag[7:] = R_diag[:7]
controller_params_3d.R = np.diag(0.1 * R_diag)
controller_params_3d.control_period = h_ctrl

q_parser = QuasistaticParser(q_model_path)
q_sim = q_parser.make_simulator_cpp()

with open("box_flipping_trj.pkl", "rb") as f:
    trj_dict = pickle.load(f)
q_knots_ref = trj_dict["x_trj"]
u_knots_ref, t_knots = calc_q_and_u_extended_and_t_knots(
    None,
    u_knots_ref=trj_dict["u_trj"],
    u_knot_ref_start=q_knots_ref[0, q_sim.get_q_a_indices_into_q()],
    v_limit=0.1,
)

controller_plant_makers = {
    iiwa_r_name: lambda gravity: create_iiwa_controller_plant(gravity)[0]
}
controller_plant_makers[iiwa_l_name] = controller_plant_makers[iiwa_r_name]

diagram_and_contents = make_controller_mbp_diagram(
    q_parser_mbp=q_parser,
    q_sim_mbp=q_sim,
    q_sim_q_control=q_sim,
    t_knots=t_knots,
    u_knots_ref=u_knots_ref,
    q_knots_ref=q_knots_ref,
    controller_params=controller_params_3d,
    create_controller_plant_functions=controller_plant_makers,
    closed_loop=True,
)

# unpack return values.
diagram = diagram_and_contents["diagram"]
controller_robots = diagram_and_contents["controller_robots"]
robot_internal_controllers = diagram_and_contents["robot_internal_controllers"]
plant = diagram_and_contents["plant"]
meshcat_vis = diagram_and_contents["meshcat_vis"]
loggers_cmd = diagram_and_contents["loggers_cmd"]
q_ref_trj = diagram_and_contents["q_ref_trj"]
u_ref_trj = diagram_and_contents["u_ref_trj"]
logger_x = diagram_and_contents["logger_x"]
loggers_contact_torque = diagram_and_contents["loggers_contact_torque"]

render_system_with_graphviz(diagram)
model_a_l = plant.GetModelInstanceByName(iiwa_l_name)
model_a_r = plant.GetModelInstanceByName(iiwa_r_name)

# %% Run sim.
sim = Simulator(diagram)
context = sim.get_context()

for model_a in q_sim.get_actuated_models():
    controller = robot_internal_controllers[model_a]
    controller.tau_feedforward_input_port.FixValue(
        controller.GetMyContextFromRoot(context),
        np.zeros(controller.tau_feedforward_input_port.size()),
    )

context_plant = plant.GetMyContextFromRoot(context)
q0 = copy.copy(q_knots_ref[0])
plant.SetPositions(context_plant, q0)

context_controller = controller_robots.GetMyContextFromRoot(context)
context_controller.SetDiscreteState(q0)

sim.Initialize()

AddTriad(
    vis=meshcat_vis.vis,
    name="frame",
    prefix="drake/plant/box/box",
    length=0.5,
    radius=0.01,
    opacity=1,
)

# sim.set_target_realtime_rate(1.0)
meshcat_vis.reset_recording()
meshcat_vis.start_recording()
sim.AdvanceTo(t_knots[-1] + 1.0)
meshcat_vis.publish_recording()

# %% plots
# 1. cmd vs nominal u.
u_logs = {
    model_a: loggers_cmd[model_a].FindLog(context)
    for model_a in q_sim.get_actuated_models()
}

u_log_l = u_logs[model_a_l]

n_q = q_sim.num_actuated_dofs() + q_sim.num_unactuated_dofs()
T = u_logs[model_a_l].data().shape[1]
u_logged = np.zeros((T, n_q))

for model_a in q_sim.get_actuated_models():
    indices = q_sim.get_position_indices()[model_a]
    u_logged[:, indices] = u_logs[model_a].data().T

u_logged = u_logged[:, q_sim.get_q_a_indices_into_q()]

u_refs = np.array(
    [u_ref_trj.value(t).squeeze() for t in u_log_l.sample_times()]
)

u_diff = np.linalg.norm(u_refs[:-1] - u_logged[1:], axis=1)

# %% 2. q_u_nominal vs q_u.
x_log = logger_x.FindLog(context)
q_log = x_log.data()[: plant.num_positions()].T
q_u_log = q_log[:, q_sim.get_q_u_indices_into_q()]
angle_error = []
position_error = []


def get_quaternion(q_unnormalized: np.ndarray):
    return Quaternion(q_unnormalized / np.linalg.norm(q_unnormalized))


for i, t in enumerate(x_log.sample_times()):
    q_u_ref = q_ref_trj.value(t).squeeze()[q_sim.get_q_u_indices_into_q()]
    q_u = q_u_log[i]

    Q_WB_ref = get_quaternion(q_u_ref[:4])
    Q_WB = get_quaternion(q_u[:4])
    angle_error.append(AngleAxis(Q_WB.inverse().multiply(Q_WB_ref)).angle())
    position_error.append(np.linalg.norm(q_u_ref[4:] - q_u[4:]))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time steps"
axes[0].plot(u_diff)
axes[0].grid(True)
axes[0].set_title("||u - u_ref||")
axes[0].set_xlabel(x_axis_label)

axes[1].set_title("||q_u - q_u_ref||")
axes[1].grid(True)
axes[1].plot(angle_error, label="angle")
axes[1].plot(position_error, label="pos")
axes[1].set_xlabel(x_axis_label)
axes[1].legend()
plt.show()

# %% 3. joint torque due to contact.
contact_torque_logs = {
    model_a: loggers_contact_torque[model_a].FindLog(context)
    for model_a in q_sim.get_actuated_models()
}

contact_torque_left = contact_torque_logs[model_a_l]
contact_torque_right = contact_torque_logs[model_a_r]

t = contact_torque_left.sample_times()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time [s]"
for i_axes, model in enumerate(sorted(q_sim.get_actuated_models())):
    contact_torque_log = contact_torque_logs[model]
    for i in range(7):
        axes[i_axes].plot(t, contact_torque_log.data()[i], label=f"joint_{i}")
        axes[i_axes].grid(True)
        axes[i_axes].set_title(f"{plant.GetModelInstanceName(model)}")
        axes[i_axes].set_xlabel(x_axis_label)
        axes[i_axes].legend()

plt.show()
