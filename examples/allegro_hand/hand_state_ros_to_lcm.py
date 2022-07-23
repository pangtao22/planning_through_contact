import lcm
import rospy
from sensor_msgs.msg import JointState

JOINT_STATE_TOPIC = '/allegroHand/joint_states'


class Sub:
    def __init__(self):
        self.allegro_state = JointState()
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self.callback)
        rospy.init_node('hand_state_to_lcm')

    def callback(self, msg):
        self.allegro_state = msg
        print(msg.name)
        print(msg.position)
        print(msg.velocity)
        print(msg.effort)


if __name__ == "__main__":
    sub = Sub()
    rospy.spin()

