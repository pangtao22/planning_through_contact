import sys

for i, path in enumerate(sys.path):
    if "2.7" in path:
        sys.path.pop(i)

import lcm
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from drake import lcmt_allegro_command

DESIRED_STATE_TOPIC = "/allegroHand/joint_cmd"
GRAV_ROT_TOPIC = "/j2n6s300_driver/hand_gravity_vector"


class AllegroCmdPublisher:
    def __init__(self):
        self.allegro_state = JointState()
        self.cmd_pub = rospy.Publisher(
            DESIRED_STATE_TOPIC, JointState, queue_size=1
        )
        self.gravity_pub = rospy.Publisher(
            GRAV_ROT_TOPIC, Float64MultiArray, queue_size=1
        )

    def handle_allegro_cmd_msg(self, channel, data):
        lcm_cmd = lcmt_allegro_command.decode(data)
        ros_cmd = JointState()

        t = rospy.Time(nsecs=lcm_cmd.utime * 1000)
        ros_cmd.header.stamp.secs = t.secs
        ros_cmd.header.stamp.nsecs = t.nsecs
        ros_cmd.position = lcm_cmd.joint_position

        ros_gravity = Float64MultiArray()
        ros_gravity.data = np.array([-9.8, 0, 0])

        self.cmd_pub.publish(ros_cmd)
        self.gravity_pub.publish(ros_gravity)

    def run(self):
        lc = lcm.LCM()
        lc.subscribe("ALLEGRO_CMD", self.handle_allegro_cmd_msg)

        while True:
            lc.handle_timeout(10)


if __name__ == "__main__":
    pub = AllegroCmdPublisher()
    rospy.init_node("allegro_cmd_from_lcm")
    pub.run()
