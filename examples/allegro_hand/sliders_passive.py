import os
import time

import numpy as np

import rospy
from sensor_msgs.msg import JointState
from pydrake.all import (LeafSystem, MultibodyPlant, DiagramBuilder, Parser,
                         AddMultibodyPlantSceneGraph, MeshcatVisualizerCpp,
                         MeshcatVisualizerParams, JointIndex, Role, Meshcat,
                         StartMeshcat, DrakeLcm, AbstractValue, Sphere,
                         LcmSubscriberSystem, LcmInterfaceSystem, Simulator,
                         RigidTransform)

from drake import lcmt_allegro_status, lcmt_allegro_command
from optitrack import optitrack_frame_t

from qsim.simulator import QuasistaticSimulator
from qsim.model_paths import models_dir

from sliders_active import (wait_for_msg, wait_for_status_msg,
                            kAllegroStatusChannel,
                            kAllegroCommandChannel, make_visualizer_diagram)

kOptitrackChannelName = "OPTITRACK_FRAMES"
kAllegroPalmName = "allegro_palm"  # 3 markers on the palm.
kAllegroBackName = "allegro_back"  # 3 markers on the back.


class MeshcatAllegroBallVisualizer(LeafSystem):
    def __init__(self,
                 meshcat: Meshcat,
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
        self.optitrack_input_port = self.DeclareAbstractInputPort(
            "optitrack",
            AbstractValue.Make(optitrack_frame_t))
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
        optitrack_msg = self.optitrack_input_port.Eval(context)

        # Hands.
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

        # update button (for changing hand color)
        n_clicks_new = self.meshcat.GetButtonClicks(self.button_name)
        if n_clicks_new != self.n_clicks:
            meshcat.SetProperty("/drake/visualizer/allegro_cmd", "color",
                                [1, 0.84, 0., 0.7])
            self.n_clicks = n_clicks_new


def is_optitrack_message_good(msg: optitrack_frame_t):
    if msg.num_marker_sets == 0:
        return False

    has_palm = False
    has_back = False
    for marker_set in msg.marker_sets:
        if marker_set.name == kAllegroPalmName:
            for p in marker_set.xyz:
                if np.linalg.norm(p) < 1e-3:
                    return False
            has_palm = True

        elif marker_set.name == kAllegroBackName:
            for p in marker_set.xyz:
                if np.linalg.norm(p) < 1e-3:
                    return False
            has_back = True

    return has_back and has_palm
#%%


def get_marker_sets_points(marker_set_name: str,
                           msg: optitrack_frame_t):
    for marker_set in msg.marker_sets:
        if marker_set.name == marker_set_name:
            return np.array(marker_set.xyz)


def draw_allegro_markers(
        meshcat: Meshcat, p_palm: np.ndarray, p_back: np.ndarray):
    # Add markers
    for i in range(3):
        meshcat.SetObject(
            f"optitrack/{kAllegroPalmName}/{i}",
            Sphere(0.00635))
        meshcat.SetTransform(
            f"optitrack/{kAllegroPalmName}/{i}",
            RigidTransform(p_palm[i]))

    for i in range(3):
        meshcat.SetObject(
            f"optitrack/{kAllegroBackName}/{i}",
            Sphere(0.00635))
        meshcat.SetTransform(
            f"optitrack/{kAllegroBackName}/{i}",
            RigidTransform(p_back[i]))

#%%
if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    drake_lcm = DrakeLcm()

    sliders = MeshcatAllegroBallVisualizer(meshcat, mvp, drake_lcm)

    builder = DiagramBuilder()
    builder.AddSystem(sliders)
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    allegro_status_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroStatusChannel,
            lcm_type=lcmt_allegro_status,
            lcm=drake_lcm))
    builder.Connect(allegro_status_sub.get_output_port(0),
                    sliders.status_input_port)

    allegro_cmd_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroCommandChannel,
            lcm_type=lcmt_allegro_command,
            lcm=drake_lcm))
    builder.Connect(allegro_cmd_sub.get_output_port(0),
                    sliders.cmd_input_port)

    optitrack_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kOptitrackChannelName,
            lcm_type=optitrack_frame_t,
            lcm=drake_lcm))
    builder.Connect(optitrack_sub.get_output_port(0),
                    sliders.optitrack_input_port)

    diagram = builder.Build()

    #%% Find World frame relative to Lab frame.
    optitrack_msg = wait_for_msg(
        channel_name=kOptitrackChannelName,
        lcm_type=optitrack_frame_t,
        is_message_good=is_optitrack_message_good)

    p_palm = get_marker_sets_points(kAllegroPalmName, optitrack_msg)
    p_back = get_marker_sets_points(kAllegroBackName, optitrack_msg)
    draw_allegro_markers(meshcat, p_palm=p_palm, p_back=p_back)

#%%
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    allegro_status = wait_for_status_msg()
    context_sub = allegro_status_sub.GetMyContextFromRoot(
        simulator.get_context())
    context_sub.SetAbstractState(0, allegro_status)
    print("Running!")

    # simulator.AdvanceTo(np.inf)
