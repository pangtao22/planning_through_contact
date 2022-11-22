import os
import time

import numpy as np

import rospy
from sensor_msgs.msg import JointState
from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    JointIndex,
    Role,
    StartMeshcat,
)
from qsim.simulator import QuasistaticSimulator
from qsim.model_paths import models_dir

JOINT_COMM_TOPIC = "/allegroHand/joint_cmd"
JOINT_STATE_TOPIC = "/allegroHand/joint_states"

allegro_file = os.path.join(
    models_dir, "allegro_hand_description_right_spheres.sdf"
)


class MeshcatJointSliders:
    """
    Adds one slider per joint of the MultibodyPlant.  Any positions that are
    not associated with joints (e.g. floating-base "mobilizers") are held
    constant at the default value obtained from robot.CreateDefaultContext().
    .. pydrake_system::
        name: JointSliders
        output_ports:
        - positions
    In addition to being used inside a Diagram that is being simulated with
    Simulator, this class also offers a `Run` method that runs its own simple
    event loop, querying the slider values and calling `Publish`.  It does not
    simulate any state dynamics.
    """

    def __init__(
        self,
        meshcat,
        mvp: MeshcatVisualizerParams,
        lower_limit=-10.0,
        upper_limit=10.0,
        resolution=0.01,
    ):
        """
        Creates an meshcat slider for each joint in the plant.
        Args:
            meshcat:      A Meshcat instance.
            lower_limit:  A scalar or vector of length robot.num_positions().
                          The lower limit of the slider will be the maximum
                          value of this number and any limit specified in the
                          Joint.
            upper_limit:  A scalar or vector of length robot.num_positions().
                          The upper limit of the slider will be the minimum
                          value of this number and any limit specified in the
                          Joint.
            resolution:   A scalar or vector of length robot.num_positions()
                          that specifies the step argument of the FloatSlider.
        """

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        # make diagram.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        Parser(plant, scene_graph).AddModelFromFile(allegro_file)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("hand_root"))
        plant.Finalize()
        self.visualizer = MeshcatVisualizerCpp.AddToBuilder(
            builder, scene_graph, meshcat, mvp
        )
        diagram = builder.Build()

        lower_limit = _broadcast(lower_limit, plant.num_positions())
        upper_limit = _broadcast(upper_limit, plant.num_positions())
        resolution = _broadcast(resolution, plant.num_positions())

        self._diagram = diagram
        self._meshcat = meshcat
        self._plant = plant

        self.context = diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.context)
        self.vis_context = self.visualizer.GetMyContextFromRoot(self.context)

        self._sliders = {}
        slider_num = 0
        positions = []
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
                index = joint.position_start() + j
                description = joint.name()
                if joint.num_positions() > 1:
                    description += "_" + joint.position_suffix(j)
                lower_limit[slider_num] = max(low[j], lower_limit[slider_num])
                upper_limit[slider_num] = min(upp[j], upper_limit[slider_num])
                value = (lower_limit[slider_num] + upper_limit[slider_num]) / 2
                positions.append(value)
                meshcat.AddSlider(
                    value=value,
                    min=lower_limit[slider_num],
                    max=upper_limit[slider_num],
                    step=resolution[slider_num],
                    name=description,
                )
                self._sliders[index] = description
                slider_num += 1

        self._plant.SetPositions(self.plant_context, positions)
        self.visualizer.Publish(self.vis_context)

        # ROS
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self.state_callback)
        self.cmd_pub = rospy.Publisher("abc", JointState, queue_size=1)
        self.allegro_state = JointState()
        self.allegro_cmd = JointState()

    def state_callback(self, msg):
        self.allegro_state = msg

    def run(self, event=None):

        # old_positions = self._plant.GetPositions(plant_context)
        # positions = self._positions
        # for i, s in self._sliders.items():
        #     positions[i] = self._meshcat.GetSliderValue(s)
        # if not np.array_equal(positions, old_positions):
        #     self._plant.SetPositions(plant_context, positions)
        #     if callback:
        #         callback(plant_context)
        #     self.visualizer.Publish(vis_context)
        self._plant.SetPositions(
            self.plant_context, self.allegro_state.position
        )
        print(self.allegro_state.position)
        self.visualizer.Publish(self.vis_context)
        self.cmd_pub.publish(self.allegro_state)


if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    sliders = MeshcatJointSliders(meshcat, mvp)

    rospy.init_node("hand_sliders")
    rospy.Timer(rospy.Duration(nsecs=33 * 1000000), sliders.run)

    rospy.spin()
