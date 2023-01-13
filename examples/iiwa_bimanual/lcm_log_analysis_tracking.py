import pathlib
from typing import Dict
import pickle

import matplotlib.pyplot as plt
import numpy as np
import lcm

from drake import lcmt_scope, lcmt_iiwa_command, lcmt_iiwa_status

from pydrake.all import RigidTransform, Quaternion, RotationMatrix, RollPitchYaw

from control.drake_sim import (
    calc_q_and_u_extended_and_t_knots,
)

from qsim.parser import QuasistaticParser
from iiwa_bimanual_setup import q_model_path_planar

kIiwaStatusChannelName = "IIWA_STATUS"
kIiwaCommandChannelName = "IIWA_COMMAND"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"

q_parser_2d = QuasistaticParser(q_model_path_planar)
q_sim_2d = q_parser_2d.make_simulator_cpp()

# %% Load reference trajectory
experiment_index = 0
ref_path = pathlib.Path.joinpath(
    pathlib.Path(__file__).parent,
    f"bimanual_patched_q_and_u_trj_{experiment_index}.pkl",
)

with open(ref_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

segment_index = 0

# %%
q_knots_ref_2d, u_knots_ref_2d, t_knots = calc_q_and_u_extended_and_t_knots(
    q_knots_ref=q_knots_ref_list[segment_index],
    u_knots_ref=u_knots_ref_list[segment_index],
    u_knot_ref_start=q_knots_ref_list[segment_index][
        0, q_sim_2d.get_q_a_indices_into_q()
    ],
    v_limit=0.05,  # This is hard-coded.
)

q_u_knots_ref_2d = q_knots_ref_2d[:, q_sim_2d.get_q_u_indices_into_q()]
q_a_knots_ref_2d = q_knots_ref_2d[:, q_sim_2d.get_q_a_indices_into_q()]

# %%
log_path = pathlib.Path.joinpath(
    pathlib.Path.home(),
    "ptc_data",
    "iiwa_bimanual",
    "segments",
    str(experiment_index),
    f"lcmlog-2022-12-16.0{segment_index}",
)
lcm_log = lcm.EventLog(str(log_path))

# %%
q_estimated_msgs_dict = {}
u_msgs_dict = {}
iiwa_status_msgs = []

for event in lcm_log:
    if event.channel == kQEstimatedChannelName:
        msg = lcmt_scope.decode(event.data)
        q_estimated_msgs_dict[event.timestamp] = msg.value

    if event.channel == kIiwaStatusChannelName:
        msg = lcmt_iiwa_status.decode(event.data)
        iiwa_status_msgs.append(msg)
        u_msgs_dict[event.timestamp] = msg.joint_position_commanded

q_list = np.array([q for t, q in q_estimated_msgs_dict.items()])
t_q_list = np.array([t / 1e6 for t, q in q_estimated_msgs_dict.items()])


# %%
def find_idx_q_moving_from_u(u_msgs_dict: Dict, du_norm_threshold=1e-8):
    """
    This function detects the first and last indices into q_list when du (
    change in commanded joint angles) exceeds du_norm_threshold.

    This function can be confused when there are more than one segments in
    u_msg_dict where du is large.
    """
    u_list = np.array([u for t, u in u_msgs_dict.items()])
    t_u_list = [t / 1e6 for t, u in u_msgs_dict.items()]
    du = np.linalg.norm(u_list[1:] - u_list[:-1], axis=1)
    is_robot_moving = du > du_norm_threshold
    plt.plot(is_robot_moving)
    plt.title("Is robot moving?")
    plt.show()

    idx_u_start = np.argmax(is_robot_moving)
    idx_u_end = -np.argmax(np.flip(is_robot_moving))
    assert idx_u_start != 0
    assert idx_u_end != len(is_robot_moving) - 1
    t_u_start = t_u_list[idx_u_start]
    t_u_end = t_u_list[idx_u_end]

    idx_q_start = np.argmin(np.abs(t_q_list - t_u_start))
    idx_q_end = np.argmin(np.abs(t_q_list - t_u_end))

    return idx_q_start, idx_q_end


idx_q_start, idx_q_end = find_idx_q_moving_from_u(u_msgs_dict)

# %%
q_list_trimmed = q_list[idx_q_start]
yaw_angles = np.array(
    [RollPitchYaw(Quaternion(q[-7:-3])).yaw_angle() for q in q_list]
)
x_positions = q_list[:, -3]
y_positions = q_list[:, -2]
yaw_angles_recentered = np.array(
    [y if y < np.pi / 2 else y - 2 * np.pi for y in yaw_angles]
)

# %% plotting for sanity check.
t_q_shifted = np.array(t_q_list[idx_q_start:idx_q_end]) - t_q_list[idx_q_start]

plt.figure()
plt.plot(
    t_q_shifted,
    yaw_angles_recentered[idx_q_start:idx_q_end],
    color="b",
    label="trajectory",
)
plt.plot(
    t_knots, q_u_knots_ref_2d[:, 2], linestyle="--", color="b", label="ref"
)
plt.ylabel("[rad]")
plt.xlabel("t [s]")
plt.title("yaw angle tracking")
plt.legend()
plt.show()

plt.figure()
plt.plot(
    t_q_shifted, x_positions[idx_q_start:idx_q_end], color="r", label="x_traj"
)
plt.plot(
    t_knots, q_u_knots_ref_2d[:, 0], linestyle="--", color="r", label="x_goal"
)
plt.plot(
    t_q_shifted, y_positions[idx_q_start:idx_q_end], color="b", label="y_traj"
)
plt.plot(
    t_knots, q_u_knots_ref_2d[:, 1], linestyle="--", color="b", label="x_goal"
)
plt.ylabel("[m]")
plt.xlabel("t [s]")
plt.title("position tracking")
plt.legend()
plt.show()

# %% logging
q_u_log = np.vstack([x_positions, y_positions, yaw_angles_recentered]).T
q_u_log = q_u_log[idx_q_start:idx_q_end]

log_dict = {
    "q_a_ref": q_a_knots_ref_2d,
    # q_a ref trajectory for visualization.
    "q_u_ref": q_u_knots_ref_2d,
    # q_u ref trajectory,
    "u_ref": u_knots_ref_2d,
    "t_knots": np.array(t_knots),
    "q_u_mbp": np.array(q_u_log),  # q_u HARDWARE(!!!) trajectory,
    "x_log_data": None,  # logger state,
    "x_log_time": t_q_shifted,  # logger state,
}

offset = 8
filename = pathlib.Path(
    pathlib.Path.home(),
    "ptc_data",
    "iiwa_bimanual",
    "sim2real",
    f"{offset + segment_index:03}.pkl",
)

with open(filename, "wb") as f:
    pickle.dump(log_dict, f)
