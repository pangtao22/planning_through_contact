import matplotlib.pyplot as plt
import numpy as np
import lcm

from drake import lcmt_scope, lcmt_iiwa_command, lcmt_iiwa_status

from pydrake.all import RigidTransform, Quaternion, RotationMatrix, RollPitchYaw

kIiwaStatusChannelName = "IIWA_STATUS"
kIiwaCommandChannelName = "IIWA_COMMAND"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"

log_path = "lcmlog-2022-09-23.01"
lcm_log = lcm.EventLog(log_path)

#%%
q_estimated_msgs = {}
iiwa_status_msgs = []

for event in lcm_log:
    if event.channel == kQEstimatedChannelName:
        msg = lcmt_scope.decode(event.data)
        q_estimated_msgs[msg.utime] = msg.value

    if event.channel == kIiwaStatusChannelName:
        msg = lcmt_iiwa_status.decode(event.data)
        iiwa_status_msgs.append(msg)


#%%
q_estimated_msgs_sorted = sorted(
    [(t / 1e6, np.array(q)) for t, q in q_estimated_msgs.items()]
)
q_list = np.array([t_q[1] for t_q in q_estimated_msgs_sorted])
t_list = np.array([t_q[0] for t_q in q_estimated_msgs_sorted])
t_list -= t_list[0]

yaw_angles = np.array(
    [RollPitchYaw(Quaternion(q[-7:-3])).yaw_angle() for q in q_list]
)
x_positions = q_list[:, -3]
y_positions = q_list[:, -2]
yaw_angles_recentered = np.array(
    [y if y < np.pi / 2 else y - 2 * np.pi for y in yaw_angles]
)

#%%
yaw_angle_goal = -2.96
x_goal = 0.475
y_goal = 0.030
idx_start = 5000

plt.figure()
plt.plot(
    t_list[idx_start:],
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
plt.plot(t_list[idx_start:], x_positions[idx_start:], color="r", label="x_traj")
plt.axhline(x_goal, linestyle="--", color="r", label="x_goal")
plt.plot(t_list[idx_start:], y_positions[idx_start:], color="b", label="y_traj")
plt.axhline(y_goal, linestyle="--", color="b", label="y_goal")
plt.ylabel("[m]")
plt.xlabel("t [s]")
plt.title("position tracking")
plt.legend()
plt.show()


#%%
t_list = np.array([msg.utime / 1e6 for msg in iiwa_status_msgs])
t_list -= t_list[0]
joint_torques = np.array(
    [msg.joint_torque_external for msg in iiwa_status_msgs]
)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x_axis_label = "Time [s]"
for i_axes in range(2):
    for i in range(7):
        i_start = 7 * i_axes
        axes[i_axes].plot(
            t_list, joint_torques[:, i_start + i], label=f"joint_{i}"
        )
        axes[i_axes].grid(True)
        axes[i_axes].set_title(f"iiwa_{i_axes}_external_torque")
        axes[i_axes].set_xlabel(x_axis_label)
        axes[i_axes].legend()

plt.show()
