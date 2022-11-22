import pickle

import numpy as np

from pydrake.all import (
    LeafSystem,
    DiagramBuilder,
    MeshcatVisualizerParams,
    Role,
    Meshcat,
    StartMeshcat,
    DrakeLcm,
    AbstractValue,
    Sphere,
    LcmSubscriberSystem,
    LcmInterfaceSystem,
    Simulator,
    RigidTransform,
    Quaternion,
    BasicVector,
    LcmScopeSystem,
)

from drake import lcmt_allegro_status, lcmt_allegro_command
from optitrack import optitrack_frame_t

from qsim.parser import QuasistaticParser

from sliders_active import (
    wait_for_msg,
    wait_for_status_msg,
    kAllegroStatusChannel,
    kAllegroCommandChannel,
    make_visualizer_diagram,
)

from optitrack_pose_estimator import (
    OptitrackPoseEstimator,
    is_optitrack_message_good,
    kBallName,
    kAllegroPalmName,
    kMarkerRadius,
)

from control.systems_utils import render_system_with_graphviz, add_triad
from allegro_hand_setup import q_model_path_hardware

kOptitrackChannelName = "OPTITRACK_FRAMES"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"


class MeshcatAllegroBallVisualizer(LeafSystem):
    def __init__(
        self,
        meshcat: Meshcat,
        mvp: MeshcatVisualizerParams,
        drake_lcm: DrakeLcm,
        pose_estimator: OptitrackPoseEstimator,
    ):
        """
        Sliders and black hand show measured joint angles.
        Golden hand show commanded joint angles.
        """
        LeafSystem.__init__(self)
        parser = QuasistaticParser(q_model_path_hardware)
        self.q_sim = parser.make_simulator_cpp()

        self.set_name("allegro_hand_sliders_passive")
        self.drake_lcm = drake_lcm
        self.DeclarePeriodicPublish(1 / 32, 0.0)  # draw at 30fps
        self.allegro_status_input_port = self.DeclareAbstractInputPort(
            "allegro_status", AbstractValue.Make(lcmt_allegro_status())
        )
        self.cmd_input_port = self.DeclareAbstractInputPort(
            "allegro_cmd", AbstractValue.Make(lcmt_allegro_command())
        )
        self.optitrack_input_port = self.DeclareAbstractInputPort(
            "optitrack", AbstractValue.Make(optitrack_frame_t)
        )
        self.q_estimated_output_port = self.DeclareVectorOutputPort(
            "q_estimated",
            BasicVector(self.q_sim.get_plant().num_positions()),
            self.calc_q_estimated,
        )
        self.meshcat = meshcat
        self.pose_estimator = pose_estimator

        # make diagram.
        (
            self.plant,
            self.visualizer,
            self.diagram,
            self.model_real,
            self.model_cmd,
        ) = make_visualizer_diagram(meshcat, mvp)

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
                    description += "_" + joint.position_suffix(j)
                lower_limit[slider_num] = low[j]
                upper_limit[slider_num] = upp[j]
                value = (lower_limit[slider_num] + upper_limit[slider_num]) / 2
                positions_default[slider_num] = value
                meshcat.AddSlider(
                    value=value,
                    min=lower_limit[slider_num],
                    max=upper_limit[slider_num],
                    step=0.01,
                    name=description,
                )
                self.sliders[slider_num] = description
                slider_num += 1

        self.lower_limits = lower_limit
        self.upper_limits = upper_limit

        self.positions_default = positions_default
        self.plant.SetPositions(
            self.plant_context, self.model_real, positions_default
        )
        self.plant.SetPositions(
            self.plant_context, self.model_cmd, positions_default
        )
        self.visualizer.Publish(self.vis_context)

        # Add button for changing the color of the controlled hand.
        self.color_button_name = "Golden Hand"
        self.meshcat.AddButton(self.color_button_name)
        self.n_clicks_color = self.meshcat.GetButtonClicks(
            self.color_button_name
        )

        self.reset_button_name = "Reset ball orientation"
        self.meshcat.AddButton(self.reset_button_name)
        self.n_clicks_reset = self.meshcat.GetButtonClicks(
            self.reset_button_name
        )

        # Add objects for markers and the sphere.
        # Palm markers.
        for i in range(3):
            meshcat.SetObject(
                f"optitrack/{kAllegroPalmName}/{i}", Sphere(kMarkerRadius)
            )
            meshcat.SetTransform(
                f"optitrack/{kAllegroPalmName}/{i}",
                RigidTransform(pose_estimator.p_palm_W[i]),
            )

        # Palm frame.
        # palm_frame_path = f"optitrack/{kAllegroPalmName}/frame"
        # add_triad(
        #     vis=self.meshcat,
        #     prefix=palm_frame_path,
        #     name="triad",
        #     length=0.075,
        #     radius=0.0015,
        #     opacity=0.1)
        # self.meshcat.SetTransform(palm_frame_path, pose_estimator.X_WP)

        # Sphere markers.
        for i in range(4):
            name = f"optitrack/{kBallName}/{i}"
            meshcat.SetObject(name, Sphere(kMarkerRadius))
        # Sphere.
        # meshcat.SetObject(
        #     f"optitrack/{kBallName}/body",
        #     Sphere(0.06), Rgba(0.5, 0.5, 0.5, 0.6))
        add_triad(
            vis=self.meshcat,
            prefix=f"optitrack/{kBallName}/body",
            name="triad",
            length=0.075,
            radius=0.0015,
        )

        # Draw start and goal frames.
        with open("hand_trj.pkl", "rb") as f:
            trj_dict = pickle.load(f)
        q_trj = trj_dict["x_trj"]
        # TODO: hard-coding joint indices is bad.
        q_u0 = q_trj[0, -7:]
        q_u_goal = q_trj[-1, -7:]

        def draw_q_u_frame(name: str, q_u):
            """
            q_u[:4] quaternion in World frame.
            q_u[4:] translation in World frame.
            """
            q_path = f"trj_tracking/{name}"
            add_triad(
                vis=self.meshcat,
                prefix=q_path,
                name="triad",
                length=0.075,
                radius=0.003,
                opacity=0.3,
            )
            X_WB = RigidTransform(Quaternion(q_u[:4]), q_u[4:])
            meshcat.SetTransform(q_path, X_WB)

        draw_q_u_frame("start", q_u0)
        draw_q_u_frame("goal", q_u_goal)

    def set_slider_values(self, values):
        for i, slider_name in self.sliders.items():
            self.meshcat.SetSliderValue(slider_name, values[i])

    def calc_q_estimated(self, context, output):
        status_msg = self.allegro_status_input_port.Eval(context)
        optitrack_msg = self.optitrack_input_port.Eval(context)
        q_a = np.array(status_msg.joint_position_measured)
        self.pose_estimator.update_X_WB(optitrack_msg)
        X_WB = self.pose_estimator.get_X_WB()
        q_u = np.hstack(
            [X_WB.rotation().ToQuaternion().wxyz(), X_WB.translation()]
        )
        q = np.zeros(self.q_sim.get_plant().num_positions())
        q[self.q_sim.get_q_a_indices_into_q()] = q_a
        q[self.q_sim.get_q_u_indices_into_q()] = q_u

        output.SetFromVector(q)

    def DoPublish(self, context, event):
        while self.drake_lcm.HandleSubscriptions(10) == 0:
            continue
        LeafSystem.DoPublish(self, context, event)
        status_msg = self.allegro_status_input_port.Eval(context)
        cmd_msg = self.cmd_input_port.Eval(context)
        optitrack_msg = self.optitrack_input_port.Eval(context)

        # Hands.
        positions_measured = status_msg.joint_position_measured
        positions_commanded = cmd_msg.joint_position
        if len(positions_commanded) == 0:
            positions_commanded = self.positions_default

        self.set_slider_values(positions_measured)
        self.plant.SetPositions(
            self.plant_context, self.model_real, positions_measured
        )
        self.plant.SetPositions(
            self.plant_context, self.model_cmd, positions_commanded
        )
        self.visualizer.Publish(self.vis_context)

        # update button (for changing hand color)
        n_clicks_new = self.meshcat.GetButtonClicks(self.color_button_name)
        if n_clicks_new != self.n_clicks_color:
            meshcat.SetProperty(
                "/drake/visualizer/allegro_cmd", "color", [1, 0.84, 0.0, 0.7]
            )
            self.n_clicks_color = n_clicks_new

        # update button for reset ball orientation.
        n_clicks_new = self.meshcat.GetButtonClicks(self.reset_button_name)
        if n_clicks_new != self.n_clicks_reset:
            self.pose_estimator.reset_ball_orientation(optitrack_msg)
            self.n_clicks_reset = n_clicks_new

        # Markers and sphere.
        (
            p_ball_surface_W,
            X_WB,
        ) = self.pose_estimator.get_p_ball_surface_W_and_X_WB(optitrack_msg)

        self.meshcat.SetTransform(f"optitrack/{kBallName}/body", X_WB)
        for i in range(len(p_ball_surface_W)):
            name = f"optitrack/{kBallName}/{i}"
            meshcat.SetTransform(name, RigidTransform(p_ball_surface_W[i]))


#%%
if __name__ == "__main__":
    meshcat = StartMeshcat()
    mvp = MeshcatVisualizerParams()
    mvp.role = Role.kIllustration
    drake_lcm = DrakeLcm()

    optitrack_msg = wait_for_msg(
        channel_name=kOptitrackChannelName,
        lcm_type=optitrack_frame_t,
        is_message_good=is_optitrack_message_good,
    )
    pose_estimator = OptitrackPoseEstimator(optitrack_msg)

    sliders = MeshcatAllegroBallVisualizer(
        meshcat, mvp, drake_lcm, pose_estimator
    )

    builder = DiagramBuilder()
    builder.AddSystem(sliders)
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    allegro_status_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroStatusChannel,
            lcm_type=lcmt_allegro_status,
            lcm=drake_lcm,
        )
    )
    builder.Connect(
        allegro_status_sub.get_output_port(0), sliders.allegro_status_input_port
    )

    allegro_cmd_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kAllegroCommandChannel,
            lcm_type=lcmt_allegro_command,
            lcm=drake_lcm,
        )
    )
    builder.Connect(allegro_cmd_sub.get_output_port(0), sliders.cmd_input_port)

    optitrack_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kOptitrackChannelName,
            lcm_type=optitrack_frame_t,
            lcm=drake_lcm,
        )
    )
    builder.Connect(
        optitrack_sub.get_output_port(0), sliders.optitrack_input_port
    )

    LcmScopeSystem.AddToBuilder(
        builder=builder,
        lcm=drake_lcm,
        signal=sliders.q_estimated_output_port,
        channel=kQEstimatedChannelName,
        publish_period=0.02,
    )

    diagram = builder.Build()

    render_system_with_graphviz(diagram, "sliders_passive.gz")

    #%%
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    allegro_status_msg = wait_for_status_msg()
    context_allegro_sub = allegro_status_sub.GetMyContextFromRoot(
        simulator.get_context()
    )
    context_allegro_sub.SetAbstractState(0, allegro_status_msg)

    context_optitrack_sub = optitrack_sub.GetMyContextFromRoot(
        simulator.get_context()
    )
    context_optitrack_sub.SetAbstractState(0, optitrack_msg)

    print("Running!")

    simulator.AdvanceTo(np.inf)
