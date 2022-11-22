import sys

for i, path in enumerate(sys.path):
    if "2.7" in path:
        sys.path.pop(i)

import lcm
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from drake import lcmt_allegro_status

JOINT_STATE_TOPIC = "/allegroHand/joint_states"

lc = lcm.LCM()


class AllegroStatusSubscriber:
    def __init__(self):
        self.allegro_state = JointState()
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self.callback)
        rospy.init_node("allegro_status_to_lcm")

    def callback(self, msg: JointState):
        self.allegro_state = msg
        lcm_msg = lcmt_allegro_status()

        t = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs)
        lcm_msg.utime = int(t.to_nsec() / 1000)

        n_joints = len(msg.position)
        lcm_msg.num_joints = n_joints
        lcm_msg.joint_position_measured = msg.position
        lcm_msg.joint_velocity_estimated = msg.velocity
        lcm_msg.joint_torque_commanded = msg.effort
        lcm_msg.joint_position_commanded = np.full(n_joints, np.nan)

        lc.publish("ALLEGRO_STATUS", lcm_msg.encode())


if __name__ == "__main__":
    sub = AllegroStatusSubscriber()
    rospy.spin()
