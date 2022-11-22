from typing import Callable
import os
import pickle

import numpy as np

from pydrake.all import (
    LeafSystem,
    MultibodyPlant,
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    Role,
    RollPitchYaw,
    RigidTransform,
    Meshcat,
    StartMeshcat,
    DrakeLcm,
    AbstractValue,
    LcmSubscriberSystem,
    LcmInterfaceSystem,
    Simulator,
    LcmPublisherSystem,
)

from drake import lcmt_allegro_status, lcmt_allegro_command
from qsim.model_paths import models_dir

from control.systems_utils import wait_for_msg

allegro_file = os.path.join(
    models_dir, "allegro_hand_description_right_spheres.sdf"
)

kAllegroStatusChannel = "ALLEGRO_STATUS"
kAllegroCommandChannel = "ALLEGRO_CMD"


def wait_for_status_msg() -> lcmt_allegro_status:
    return wait_for_msg(
        channel_name=kAllegroStatusChannel,
        lcm_type=lcmt_allegro_status,
        is_message_good=lambda msg: msg.num_joints > 0,
    )


def make_visualizer_diagram(meshcat: Meshcat, mvp: MeshcatVisualizerParams):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser = Parser(plant, scene_graph)
    model_real = parser.AddModelFromFile(allegro_file, "allegro_real")
    model_cmd = parser.AddModelFromFile(allegro_file, "allegro_cmd")
    X = RigidTransform(RollPitchYaw(0, -np.pi / 2, 0), np.zeros(3))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("hand_root", model_cmd), X
    )
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("hand_root", model_real), X
    )
    plant.Finalize()
    visualizer = MeshcatVisualizerCpp.AddToBuilder(
        builder, scene_graph, meshcat, mvp
    )
    diagram = builder.Build()

    return plant, visualizer, diagram, model_real, model_cmd


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

    def __init__(
        self,
        meshcat: Meshcat,
        mvp: MeshcatVisualizerParams,
        drake_lcm: DrakeLcm,
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
        super().__init__()
        self.set_name("allegro_hand_sliders_passive")
        self.drake_lcm = drake_lcm
        self.DeclarePeriodicPublish(0.02, 0.0)
        self.status_input_port = self.DeclareAbstractInputPort(
            "allegro_status", AbstractValue.Make(lcmt_allegro_status())
        )
        self.cmd_output_port = self.DeclareAbstractOutputPort(
            "allegro_cmd",
            lambda: AbstractValue.Make(lcmt_allegro_command()),
            self.copy_allegro_cmd_out,
        )
        self.allegro_stats_msg = None
        self.meshcat = meshcat

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        # make diagram.
        (
            self.plant,
            self.visualizer,
            self.diagram,
            self.model_real,
            self.model_cmd,
        ) = make_visualizer_diagram(meshcat, mvp)

        n_qa = self.plant.num_positions(self.model_real)
        lower_limit = _broadcast(lower_limit, n_qa)
        upper_limit = _broadcast(upper_limit, n_qa)
        resolution = _broadcast(resolution, n_qa)

        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.vis_context = self.visualizer.GetMyContextFromRoot(self.context)

        self.sliders = {}
        slider_num = 0
        positions = []
        for i in self.plant.GetJointIndices(self.model_real):
            joint = self.plant.get_joint(i)
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
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
                self.sliders[slider_num] = description
                slider_num += 1

        self.lower_limits = lower_limit
        self.upper_limits = upper_limit

        self.plant.SetPositions(self.plant_context, self.model_real, positions)
        self.plant.SetPositions(self.plant_context, self.model_cmd, positions)
        self.visualizer.Publish(self.vis_context)

        # Add button for changing the color of the controlled hand.
        self.color_button_name = "Golden Hand"
        self.meshcat.AddButton(self.color_button_name)
        self.n_clicks_color = self.meshcat.GetButtonClicks(
            self.color_button_name
        )

        # Add button for moving to a pre-defined joint angle configuration.
        self.move_button_name = "Move to q_a0"
        self.meshcat.AddButton(self.move_button_name)
        self.n_clicks_move = self.meshcat.GetButtonClicks(self.move_button_name)

        with open("hand_trj.pkl", "rb") as f:
            trj_dict = pickle.load(f)
        q_trj = trj_dict["x_trj"]
        # TODO: hard-coding joint indices is bad.
        self.q_a0 = q_trj[0, :16]

    def get_slider_values(self):
        values = np.zeros(len(self.sliders))
        for i, s in self.sliders.items():
            values[i] = self.meshcat.GetSliderValue(s)
        return values

    def set_slider_values(self, values):
        values_clipped = np.clip(values, self.lower_limits, self.upper_limits)
        for i, slider_name in self.sliders.items():
            self.meshcat.SetSliderValue(slider_name, values_clipped[i])

    def copy_allegro_cmd_out(self, context, output):
        msg = output.get_value()
        msg.utime = self.allegro_stats_msg.utime
        positions = self.get_slider_values()
        msg.num_joints = len(positions)
        msg.joint_position = positions

    def DoPublish(self, context, event):
        super().DoPublish(context, event)
        status_msg = self.EvalAbstractInput(context, 0).get_value()

        if self.allegro_stats_msg is None:
            # "Initialization" of slider and golden hand.
            self.set_slider_values(status_msg.joint_position_measured)

        # update color button
        n_clicks_new = self.meshcat.GetButtonClicks(self.color_button_name)
        if n_clicks_new != self.n_clicks_color:
            meshcat.SetProperty(
                "/drake/visualizer/allegro_cmd", "color", [1, 0.84, 0.0, 0.7]
            )
            self.n_clicks_color = n_clicks_new

        # update move button
        n_clicks_new = self.meshcat.GetButtonClicks(self.move_button_name)
        if n_clicks_new != self.n_clicks_move:
            self.set_slider_values(self.q_a0)
            self.n_clicks_move = n_clicks_new

        self.allegro_stats_msg = status_msg
        positions = status_msg.joint_position_measured
        self.plant.SetPositions(self.plant_context, self.model_real, positions)
        self.plant.SetPositions(
            self.plant_context, self.model_cmd, self.get_slider_values()
        )
        self.visualizer.Publish(self.vis_context)


if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    drake_lcm = DrakeLcm()

    sliders = MeshcatJointSliders(meshcat, mvp, drake_lcm)

    builder = DiagramBuilder()
    builder.AddSystem(sliders)
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    # Allegro status subscriber.
    allegro_lcm_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroStatusChannel,
            lcm_type=lcmt_allegro_status,
            lcm=drake_lcm,
        )
    )
    builder.Connect(
        allegro_lcm_sub.get_output_port(0), sliders.status_input_port
    )

    # Allegro command publisher.
    allegro_lcm_pub = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel=kAllegroCommandChannel,
            lcm_type=lcmt_allegro_command,
            lcm=drake_lcm,
            publish_period=0.01,
        )
    )
    builder.Connect(sliders.cmd_output_port, allegro_lcm_pub.get_input_port(0))

    diagram = builder.Build()
    # RenderSystemWithGraphviz(diagram)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    # Make sure that the first status message read my the sliders is the real
    # status of the hand.
    print("Waiting for first Allegro Status msg...")
    allegro_status = wait_for_status_msg()
    context_sub = allegro_lcm_sub.GetMyContextFromRoot(simulator.get_context())
    context_sub.SetAbstractState(0, allegro_status)
    print("Running!")

    simulator.AdvanceTo(np.inf)
