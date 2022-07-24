import os
import time

import numpy as np

import rospy
from sensor_msgs.msg import JointState
from pydrake.all import (LeafSystem, MultibodyPlant, DiagramBuilder, Parser,
                         AddMultibodyPlantSceneGraph, MeshcatVisualizerCpp,
                         MeshcatVisualizerParams, JointIndex, Role,
                         StartMeshcat, DrakeLcm, AbstractValue,
                         LcmSubscriberSystem, LcmInterfaceSystem, Simulator)

from drake import lcmt_allegro_status
from qsim.simulator import QuasistaticSimulator
from qsim.model_paths import models_dir

allegro_file = os.path.join(
    models_dir, "allegro_hand_description_right_spheres.sdf")


class MeshcatJointSliders(LeafSystem):
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

    def __init__(self,
                 meshcat,
                 mvp: MeshcatVisualizerParams,
                 drake_lcm: DrakeLcm,
                 lower_limit=-10.,
                 upper_limit=10.,
                 resolution=0.01):
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
        LeafSystem.__init__(self)
        self.set_name('allegro_hand_sliders_passive')
        self.drake_lcm = drake_lcm
        self.DeclarePeriodicPublish(1. / 60, 0.0)  # draw at 30fps
        self.DeclareAbstractInputPort("allegro_status",
                                      AbstractValue.Make(lcmt_allegro_status()))

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        # make diagram.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.0)

        Parser(plant, scene_graph).AddModelFromFile(allegro_file)
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("hand_root"))
        plant.Finalize()
        self.visualizer = MeshcatVisualizerCpp.AddToBuilder(
            builder, scene_graph, meshcat, mvp)
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
                    description += '_' + joint.position_suffix(j)
                lower_limit[slider_num] = max(low[j], lower_limit[slider_num])
                upper_limit[slider_num] = min(upp[j], upper_limit[slider_num])
                value = (lower_limit[slider_num] + upper_limit[slider_num]) / 2
                positions.append(value)
                meshcat.AddSlider(value=value,
                                  min=lower_limit[slider_num],
                                  max=upper_limit[slider_num],
                                  step=resolution[slider_num],
                                  name=description)
                self._sliders[index] = description
                slider_num += 1

        self._plant.SetPositions(self.plant_context, positions)
        self.visualizer.Publish(self.vis_context)

    def DoPublish(self, context, event):
        while self.drake_lcm.HandleSubscriptions(10) == 0:
            continue
        LeafSystem.DoPublish(self, context, event)
        status_msg = self.EvalAbstractInput(context, 0).get_value()
        if status_msg.num_joints == 0:
            print("empty allegro status msg!")
            return

        positions = status_msg.joint_position_measured
        for i, slider_name in self._sliders.items():
            self._meshcat.SetSliderValue(slider_name, positions[i])

        self._plant.SetPositions(self.plant_context, positions)
        self.visualizer.Publish(self.vis_context)


if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    drake_lcm = DrakeLcm()

    sliders = MeshcatJointSliders(meshcat, mvp, drake_lcm)

    builder = DiagramBuilder()
    builder.AddSystem(sliders)

    iiwa_lcm_sub = builder.AddSystem(LcmSubscriberSystem.Make(
        channel="ALLEGRO_STATUS", lcm_type=lcmt_allegro_status, lcm=drake_lcm))
    builder.Connect(iiwa_lcm_sub.get_output_port(0),
                    sliders.get_input_port(0))
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    diagram = builder.Build()
    # RenderSystemWithGraphviz(diagram)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    simulator.AdvanceTo(np.inf)

