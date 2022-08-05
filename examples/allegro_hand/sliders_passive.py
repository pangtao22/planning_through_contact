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

from drake import lcmt_allegro_status, lcmt_allegro_command
from qsim.simulator import QuasistaticSimulator
from qsim.model_paths import models_dir

from sliders_active import (wait_for_status_msg, kAllegroStatusChannel,
                            kAllegroCommandChannel, make_visualizer_diagram)


class MeshcatJointSliders(LeafSystem):
    def __init__(self,
                 meshcat,
                 mvp: MeshcatVisualizerParams,
                 drake_lcm: DrakeLcm):
        """
        Sliders and black hand show measured joint angles.
        Golden hand show commanded joint angles.
        """
        LeafSystem.__init__(self)
        self.set_name('allegro_hand_sliders_passive')
        self.drake_lcm = drake_lcm
        self.DeclarePeriodicPublish(1 / 32, 0.0)  # draw at 30fps
        self.status_input_port = self.DeclareAbstractInputPort(
            "allegro_status",
            AbstractValue.Make(lcmt_allegro_status()))
        self.cmd_input_port = self.DeclareAbstractInputPort(
            "allegro_cmd",
            AbstractValue.Make(lcmt_allegro_command()))
        self.meshcat = meshcat

        # make diagram.
        (self.plant, self.visualizer, self.diagram, self.model_real,
         self.model_cmd) = make_visualizer_diagram(meshcat, mvp)

        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.vis_context = self.visualizer.GetMyContextFromRoot(self.context)

        n_qa = self.plant.num_positions(self.model_real)
        lower_limit = np.zeros(n_qa)
        upper_limit = np.zeros(n_qa)
        positions_default = np.zeros(n_qa)

        self.sliders = {}
        slider_num = 0
        for i in self.plant.GetJointIndices(self.model_real):
            joint = self.plant.get_joint(i)
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
                description = joint.name()
                if joint.num_positions() > 1:
                    description += '_' + joint.position_suffix(j)
                lower_limit[slider_num] = low[j]
                upper_limit[slider_num] = upp[j]
                value = (lower_limit[slider_num] + upper_limit[slider_num]) / 2
                positions_default[slider_num] = value
                meshcat.AddSlider(value=value,
                                  min=lower_limit[slider_num],
                                  max=upper_limit[slider_num],
                                  step=0.01,
                                  name=description)
                self.sliders[slider_num] = description
                slider_num += 1

        self.lower_limits = lower_limit
        self.upper_limits = upper_limit

        # Add button for changing the color of the controlled hand.
        self.button_name = "Golden Hand"
        self.meshcat.AddButton(self.button_name)
        self.n_clicks = self.meshcat.GetButtonClicks(self.button_name)

        self.positions_default = positions_default
        self.plant.SetPositions(self.plant_context, self.model_real,
                                positions_default)
        self.plant.SetPositions(self.plant_context, self.model_cmd,
                                positions_default)
        self.visualizer.Publish(self.vis_context)

    def set_slider_values(self, values):
        for i, slider_name in self.sliders.items():
            self.meshcat.SetSliderValue(slider_name, values[i])

    def DoPublish(self, context, event):
        while self.drake_lcm.HandleSubscriptions(10) == 0:
            continue
        LeafSystem.DoPublish(self, context, event)
        status_msg = self.status_input_port.Eval(context)
        cmd_msg = self.cmd_input_port.Eval(context)

        positions_measured = status_msg.joint_position_measured
        positions_commanded = cmd_msg.joint_position
        if len(positions_commanded) == 0:
            positions_commanded = self.positions_default

        self.set_slider_values(positions_measured)
        self.plant.SetPositions(self.plant_context, self.model_real,
                                positions_measured)
        self.plant.SetPositions(self.plant_context, self.model_cmd,
                                positions_commanded)
        self.visualizer.Publish(self.vis_context)

        # update button
        n_clicks_new = self.meshcat.GetButtonClicks(self.button_name)
        if n_clicks_new != self.n_clicks:
            meshcat.SetProperty("/drake/visualizer/allegro_cmd", "color",
                                [1, 0.84, 0., 0.7])
            self.n_clicks = n_clicks_new


if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    drake_lcm = DrakeLcm()

    sliders = MeshcatJointSliders(meshcat, mvp, drake_lcm)

    builder = DiagramBuilder()
    builder.AddSystem(sliders)
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    allegro_stats_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroStatusChannel,
            lcm_type=lcmt_allegro_status,
            lcm=drake_lcm))
    builder.Connect(allegro_stats_sub.get_output_port(0),
                    sliders.status_input_port)

    allegro_cmd_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroCommandChannel,
            lcm_type=lcmt_allegro_command,
            lcm=drake_lcm))
    builder.Connect(allegro_cmd_sub.get_output_port(0),
                    sliders.cmd_input_port)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    print("Waiting for first Allegro Status msg...")
    allegro_status = wait_for_status_msg()
    context_sub = allegro_stats_sub.GetMyContextFromRoot(
        simulator.get_context())
    context_sub.SetAbstractState(0, allegro_status)
    print("Running!")

    simulator.AdvanceTo(np.inf)
