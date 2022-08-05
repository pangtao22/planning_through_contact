import sys
for i, path in enumerate(sys.path):
    if "2.7" in path:
        sys.path.pop(i)

import lcm
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from drake import lcmt_allegro_command

DESIRED_STATE_TOPIC = "/allegroHand/joint_cmd"


class AllegroCmdPublisher:
    def __init__(self):
        self.allegro_state = JointState()
        self.pub = rospy.Publisher(
            DESIRED_STATE_TOPIC, JointState, queue_size=1)

    def handle_allegro_cmd_msg(self, channel, data):
        lcm_msg = lcmt_allegro_command.decode(data)
        ros_msg = JointState()

        t = rospy.Time(nsecs=lcm_msg.utime * 1000)
        ros_msg.header.stamp.secs = t.secs
        ros_msg.header.stamp.nsecs = t.nsecs
        ros_msg.position = lcm_msg.joint_position

        self.pub.publish(ros_msg)

    def run(self):
        lc = lcm.LCM()
        lc.subscribe("ALLEGRO_CMD", self.handle_allegro_cmd_msg)

        while True:
            lc.handle_timeout(10)


if __name__ == "__main__":
    pub = AllegroCmdPublisher()
    rospy.init_node('allegro_cmd_from_lcm')
    pub.run()
