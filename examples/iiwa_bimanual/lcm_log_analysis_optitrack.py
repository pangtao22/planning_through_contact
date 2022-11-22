import matplotlib.pyplot as plt
import numpy as np
import lcm

from drake import lcmt_scope
from optitrack import optitrack_frame_t

from pydrake.all import RigidTransform, Quaternion, RotationMatrix, RollPitchYaw


#%%
kOptitrackChannelName = "OPTITRACK_FRAMES"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"

log_path = "lcmlog-2022-09-14.00"
lcm_log = lcm.EventLog(log_path)

q_estimated_msgs = []
optitrack_frame_msgs = []

for event in lcm_log:
    if event.channel == kOptitrackChannelName:
        optitrack_frame_msgs.append(optitrack_frame_t.decode(event.data))
    elif event.channel == kQEstimatedChannelName:
        q_estimated_msgs.append(lcmt_scope.decode(event.data))

#%%
quaternions_filtered = np.array([msg.value[-7:-3] for msg in q_estimated_msgs])

plt.figure()
plt.plot(quaternions_filtered[:, 0], label="w")
plt.plot(quaternions_filtered[:, 1], label="x")
plt.plot(quaternions_filtered[:, 2], label="y")
plt.plot(quaternions_filtered[:, 3], label="z")
plt.legend()
plt.show()


#%%
yaw_angles = np.array(
    [RollPitchYaw(Quaternion(q)).yaw_angle() for q in quaternions_filtered]
)

yaw_angles_recentered = np.array(
    [y if y < np.pi / 2 else y - 2 * np.pi for y in yaw_angles]
)

plt.figure()
plt.plot(yaw_angles)
plt.plot(yaw_angles_recentered)
plt.show()
