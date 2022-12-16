import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    Quaternion,
    AngleAxis,
    Simulator,
    RigidTransform,
    RollPitchYaw,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from qsim.parser import QuasistaticParser
from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant,
)

from iiwa_bimanual_setup import (
    q_model_path_cylinder,
    q_model_path_planar,
    iiwa_l_name,
    iiwa_r_name,
    controller_params_2d,
    calc_z_height,
)
from control.drake_sim import (
    load_ref_trajectories,
    make_controller_mbp_diagram,
    calc_q_and_u_extended_and_t_knots,
)
from control.controller_planar_iiwa_bimanual import kIndices3Into7
from control.systems_utils import render_system_with_graphviz

# %%
h_ref_knot = 1.0
h_ctrl = 0.005
controller_params_2d.control_period = h_ctrl

q_parser_2d = QuasistaticParser(q_model_path_planar)
q_parser_3d = QuasistaticParser(q_model_path_cylinder)
q_sim_2d = q_parser_2d.make_simulator_cpp()
q_sim_3d = q_parser_3d.make_simulator_cpp()

# %% Trajectory.
file_path = "bimanual_optimized_q_and_u_trj.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

# pick one segment for now.
idx_trj_segment = 5
q_knots_ref, u_knots_ref, t_knots = calc_q_and_u_extended_and_t_knots(
    q_knots_ref=q_knots_ref_list[idx_trj_segment],
    u_knots_ref=u_knots_ref_list[idx_trj_segment],
    u_knot_ref_start=q_knots_ref_list[idx_trj_segment][
        0, q_sim_2d.get_q_a_indices_into_q()
    ],
    v_limit=0.1,
)

controller_plant_makers = {
    iiwa_r_name: lambda gravity: create_iiwa_controller_plant(gravity)[0]
}
controller_plant_makers[iiwa_l_name] = controller_plant_makers[iiwa_r_name]

diagram_and_contents = make_controller_mbp_diagram(
    q_parser_mbp=q_parser_3d,
    q_sim_mbp=q_sim_3d,
    q_sim_q_control=q_sim_2d,
    t_knots=t_knots,
    u_knots_ref=u_knots_ref,
    q_knots_ref=q_knots_ref,
    controller_params=controller_params_2d,
    create_controller_plant_functions=controller_plant_makers,
    closed_loop=False,
)

# unpack return values.
diagram = diagram_and_contents["diagram"]
controller_robots = diagram_and_contents["controller_robots"]
robot_internal_controllers = diagram_and_contents["robot_internal_controllers"]
plant_3d = diagram_and_contents["plant"]
meshcat_vis = diagram_and_contents["meshcat_vis"]
loggers_cmd = diagram_and_contents["loggers_cmd"]
q_ref_trj = diagram_and_contents["q_ref_trj"]
u_ref_trj = diagram_and_contents["u_ref_trj"]
logger_x = diagram_and_contents["logger_x"]
loggers_contact_torque = diagram_and_contents["loggers_contact_torque"]
meshcat = diagram_and_contents["meshcat"]

render_system_with_graphviz(diagram)
model_a_l_3d = plant_3d.GetModelInstanceByName(iiwa_l_name)
model_a_r_3d = plant_3d.GetModelInstanceByName(iiwa_r_name)

plant_2d = q_sim_2d.get_plant()
model_a_l_2d = plant_2d.GetModelInstanceByName(iiwa_l_name)
model_a_r_2d = plant_2d.GetModelInstanceByName(iiwa_r_name)

# %% Run sim.
sim = Simulator(diagram)
context = sim.get_context()

for model_a in q_sim_3d.get_actuated_models():
    controller = robot_internal_controllers[model_a]
    controller.tau_feedforward_input_port.FixValue(
        controller.GetMyContextFromRoot(context),
        np.zeros(controller.tau_feedforward_input_port.size()),
    )

context_plant = plant_3d.GetMyContextFromRoot(context)
q0 = controller_robots.calc_q_3d_from_q_2d(q_knots_ref[0])
plant_3d.SetPositions(context_plant, q0)

context_controller = controller_robots.GetMyContextFromRoot(context)
context_controller.SetDiscreteState(q0)

sim.Initialize()

AddMeshcatTriad(
    meshcat=meshcat,
    path="visualizer/box/box/frame",
    length=0.4,
    radius=0.005,
    opacity=1,
)
q_u_goal = q_knots_ref[-1]
AddMeshcatTriad(
    meshcat=meshcat,
    path="goal/frame",
    length=0.4,
    radius=0.01,
    opacity=0.7,
    X_PT=RigidTransform(
        RollPitchYaw(0, 0, q_u_goal[2]),
        np.hstack([q_u_goal[:2], [calc_z_height(plant_2d)]]),
    ),
)

sim.set_target_realtime_rate(1.0)
meshcat_vis.DeleteRecording()
meshcat_vis.StartRecording()
sim.AdvanceTo(t_knots[-1] * 1.05)
meshcat_vis.PublishRecording()

# %% plots
# 1. cmd vs nominal u.
u_logs = {
    model_a: loggers_cmd[model_a].FindLog(context)
    for model_a in q_sim_3d.get_actuated_models()
}

u_log_l = u_logs[model_a_l_3d]

n_q = q_sim_2d.get_plant().num_positions()
T = u_logs[model_a_l_3d].data().shape[1]
u_logged = np.zeros((T, n_q))

indices_map_2d = q_sim_2d.get_position_indices()

indices = indices_map_2d[model_a_r_2d]
u_logged[:, indices] = u_logs[model_a_r_3d].data()[kIndices3Into7].T

indices = indices_map_2d[model_a_l_2d]
u_logged[:, indices] = u_logs[model_a_l_3d].data()[kIndices3Into7].T

u_logged = u_logged[:, q_sim_2d.get_q_a_indices_into_q()]

u_refs = np.array(
    [u_ref_trj.value(t).squeeze() for t in u_log_l.sample_times()]
)

u_diff = np.linalg.norm(u_refs[:-1] - u_logged[1:], axis=1)

# %% 2. q_u_nominal vs q_u.
x_log = logger_x.FindLog(context)
q_3d_log = x_log.data()[: plant_3d.num_positions()].T
q_2d_log = [controller_robots.calc_q_2d_from_q_3d(q_3d) for q_3d in q_3d_log]
q_2d_log = np.array(q_2d_log)
q_u_2d_log = q_2d_log[:, q_sim_2d.get_q_u_indices_into_q()]
angle_error = []
position_error = []

for i, t in enumerate(x_log.sample_times()):
    q_u_2d_ref = q_ref_trj.value(t).squeeze()[q_sim_2d.get_q_u_indices_into_q()]
    q_u_2d = q_u_2d_log[i]

    angle_error.append(abs(q_u_2d_ref[2] - q_u_2d[2]))
    position_error.append(np.linalg.norm(q_u_2d_ref[:2] - q_u_2d[:2]))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time [s]"
axes[0].plot(x_log.sample_times()[1:], u_diff)
axes[0].grid(True)
axes[0].set_title("||u - u_ref||")
axes[0].set_xlabel(x_axis_label)

axes[1].set_title("||q_u - q_u_ref||")
axes[1].grid(True)
axes[1].plot(x_log.sample_times(), angle_error, label="angle")
axes[1].plot(x_log.sample_times(), position_error, label="pos")
axes[1].set_xlabel(x_axis_label)
axes[1].legend()
plt.show()

# %% 3. joint torque due to contact.
contact_torque_logs = {
    model_a: loggers_contact_torque[model_a].FindLog(context)
    for model_a in q_sim_3d.get_actuated_models()
}

contact_torque_left = contact_torque_logs[model_a_l_3d]
contact_torque_right = contact_torque_logs[model_a_r_3d]

t = contact_torque_left.sample_times()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time [s]"
for i_axes, model in enumerate(sorted(q_sim_3d.get_actuated_models())):
    contact_torque_log = contact_torque_logs[model]
    for i in range(7):
        axes[i_axes].plot(t, contact_torque_log.data()[i], label=f"joint_{i}")
        axes[i_axes].grid(True)
        axes[i_axes].set_title(f"{plant_3d.GetModelInstanceName(model)}")
        axes[i_axes].set_xlabel(x_axis_label)
        axes[i_axes].legend()

plt.show()

# %%
t, indices = controller_robots.controller.calc_t_and_indices_for_q(
    q_knots_ref[-1]
)

s = controller_robots.controller.calc_arc_length(t, indices)

controller_robots.controller.calc_q_and_u_from_arc_length(s + 0.1)
