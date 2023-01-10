import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import lcm

from drake import lcmt_scope, lcmt_iiwa_command, lcmt_iiwa_status

from pydrake.all import RigidTransform, Quaternion, RotationMatrix, RollPitchYaw

kIiwaStatusChannelName = "IIWA_STATUS"
kIiwaCommandChannelName = "IIWA_COMMAND"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"

log_path = pathlib.Path.joinpath(
    pathlib.Path.home(),
    "ptc_data",
    "iiwa_bimanual",
    "segments",
    "0",
    "lcmlog-2022-12-16.00",
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
def find_idx_q_start_from_u(u_msgs_dict: Dict, dq_norm_threshold=1e-8):
    u_list = np.array([u for t, u in u_msgs_dict.items()])
    t_u_list = [t / 1e6 for t, u in u_msgs_dict.items()]
    idx_u_start = np.argmax(
        np.linalg.norm(u_list[1:] - u_list[:-1], axis=1) > dq_norm_threshold
    )
    assert idx_u_start != 0
    t_u_start = t_u_list[idx_u_start]

    idx_q_start = np.argmin(np.abs(t_q_list - t_u_start))
    assert idx_q_start != 0

    return idx_q_start


idx_q_start = find_idx_q_start_from_u(u_msgs_dict)

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

# %%
q_u_list = np.vstack([x_positions, y_positions, yaw_angles_recentered]).T

delta_q_u_list = q_u_list[1:] - q_u_list[:-1]
# %%
yaw_angle_goal = -2.96
x_goal = 0.475
y_goal = 0.030
idx_start = 5000

plt.figure()
plt.plot(
    t_q_list[idx_start:],
    yaw_angles_recentered[idx_start:],
    color="b",
    label="trajectory",
)
plt.axhline(yaw_angle_goal, linestyle="--", color="b", label="goal")
plt.ylabel("[rad]")
plt.xlabel("t [s]")
plt.title("yaw angle tracking")
plt.legend()
plt.show()

plt.figure()
plt.plot(
    t_q_list[idx_start:], x_positions[idx_start:], color="r", label="x_traj"
)
plt.axhline(x_goal, linestyle="--", color="r", label="x_goal")
plt.plot(
    t_q_list[idx_start:], y_positions[idx_start:], color="b", label="y_traj"
)
plt.axhline(y_goal, linestyle="--", color="b", label="y_goal")
plt.ylabel("[m]")
plt.xlabel("t [s]")
plt.title("position tracking")
plt.legend()
plt.show()

# %%
t_q_list = np.array([msg.utime / 1e6 for msg in iiwa_status_msgs])
t_q_list -= t_q_list[0]
joint_torques = np.array(
    [msg.joint_torque_external for msg in iiwa_status_msgs]
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time [s]"
for i_axes in range(2):
    for i in range(7):
        i_start = 7 * i_axes
        axes[i_axes].plot(
            t_q_list, joint_torques[:, i_start + i], label=f"joint_{i}"
        )
        axes[i_axes].grid(True)
        axes[i_axes].set_title(f"iiwa_{i_axes}_external_torque")
        axes[i_axes].set_xlabel(x_axis_label)
        axes[i_axes].legend()

plt.show()
