import pickle
from typing import Callable, Dict, Set

import numpy as np
from pydrake.all import (
    MultibodyPlant,
    DiagramBuilder,
    LcmSubscriberSystem,
    LcmPublisherSystem,
    LcmInterfaceSystem,
    LeafSystem,
    Demultiplexer,
    LogVectorOutput,
    DrakeLcm,
    BasicVector,
    AbstractValue,
    PortDataType,
    Simulator,
    ModelInstanceIndex,
    LcmScopeSystem,
    MeshcatVisualizer,
    StartMeshcat,
)
from qsim.parser import QuasistaticParser
from qsim_cpp import QuasistaticSimulatorCpp

from drake import lcmt_iiwa_command, lcmt_iiwa_status

from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant,
)

from control.drake_sim import add_mbp_scene_graph, add_internal_controllers
from control.systems_utils import render_system_with_graphviz

from iiwa_bimanual_setup import (
    q_model_path,
    iiwa_l_name,
    iiwa_r_name,
    q_model_path_cylinder,
)
from state_estimator import kQEstimatedChannelName


#%%
class CmdLcm2VecSystem(LeafSystem):
    def __init__(self):
        super().__init__()
        self.set_name("command_lcm_to_vec")
        self.q_sim = q_sim

        self.command_input_port = self.DeclareAbstractInputPort(
            "iiwa_command", AbstractValue.Make(lcmt_iiwa_command())
        )

        self.cmd_left_output_port = self.DeclareVectorOutputPort(
            "left_iiwa_command", BasicVector(7), self.calc_left_command
        )

        self.cmd_right_output_port = self.DeclareVectorOutputPort(
            "right_iiwa_command", BasicVector(7), self.calc_right_command
        )

    def calc_left_command(self, context, output):
        iiwa_cmd_msg = self.command_input_port.Eval(context)
        # THE LEFT IIWA COMES FIRST. THIS IS HARD-CODED!
        output.SetFromVector(iiwa_cmd_msg.joint_position[:7])

    def calc_right_command(self, context, output):
        iiwa_cmd_msg = self.command_input_port.Eval(context)
        # THE LEFT IIWA COMES FIRST. THIS IS HARD-CODED!
        output.SetFromVector(iiwa_cmd_msg.joint_position[7:])


class StatusVec2LcmSystem(LeafSystem):
    def __init__(self, q_sim: QuasistaticSimulatorCpp):
        super().__init__()
        self.set_name("status_vec_to_lcm")
        self.iiwa_cmd_input_port = self.DeclareAbstractInputPort(
            "iiwa_cmd", AbstractValue.Make(lcmt_iiwa_command())
        )
        self.q_sim = q_sim
        self.plant = q_sim.get_plant()

        self.q_a_indices_into_q = self.q_sim.get_q_a_indices_into_q()

        self.x_input_port = self.DeclareInputPort(
            "q_v",
            PortDataType.kVectorValued,
            self.plant.num_positions() + self.plant.num_velocities(),
        )

        self.status_output_port = self.DeclareAbstractOutputPort(
            "iiwa_status",
            lambda: AbstractValue.Make(lcmt_iiwa_status()),
            self.calc_iiwa_status,
        )

    def calc_iiwa_status(self, context, output):
        iiwa_cmd_msg = self.iiwa_cmd_input_port.Eval(context)
        x = self.x_input_port.Eval(context)
        n_q = self.plant.num_positions()
        q = x[:n_q]
        v = x[n_q:]

        iiwa_status_msg = output.get_value()
        iiwa_status_msg.utime = int(1e6 * context.get_time())
        iiwa_status_msg.num_joints = self.q_sim.num_actuated_dofs()
        iiwa_status_msg.joint_position_measured = q[self.q_a_indices_into_q]
        # TODO: indexing into v using indices for q could be wrong...
        iiwa_status_msg.joint_velocity_estimated = v[self.q_a_indices_into_q]
        iiwa_status_msg.joint_position_commanded = iiwa_cmd_msg.joint_position

        if len(iiwa_cmd_msg.joint_torque) > 0:
            iiwa_status_msg.joint_torque_commanded = iiwa_cmd_msg.joint_torque
        else:
            iiwa_status_msg.joint_torque_commanded = np.zeros(14)

            # Fields not populated yet.
        iiwa_status_msg.joint_position_ipo = np.full(14, np.nan)
        iiwa_status_msg.joint_torque_measured = np.full(14, np.nan)
        iiwa_status_msg.joint_torque_external = np.full(14, np.nan)


#%%
q_parser = QuasistaticParser(q_model_path_cylinder)
has_objects = True
q_sim = q_parser.make_simulator_cpp(has_objects)
model_l_iiwa = q_sim.get_plant().GetModelInstanceByName(iiwa_l_name)
model_r_iiwa = q_sim.get_plant().GetModelInstanceByName(iiwa_r_name)

controller_plant_makers = {
    iiwa_r_name: lambda gravity: create_iiwa_controller_plant(gravity)[0]
}
controller_plant_makers[iiwa_l_name] = controller_plant_makers[iiwa_r_name]

builder = DiagramBuilder()
# MBP and SceneGraph.
plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
    q_parser, builder, has_objects=has_objects, mbp_time_step=5e-4
)

# Add visualizer.
meshcat = StartMeshcat()
meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Impedance (PD) controller for robots, with gravity compensation.
models_actuated = q_sim.get_actuated_models()
robot_internal_controllers = add_internal_controllers(
    models_actuated=models_actuated,
    q_parser=q_parser,
    plant=plant,
    builder=builder,
    controller_plant_makers=controller_plant_makers,
)


drake_lcm = DrakeLcm()
builder.AddSystem(LcmInterfaceSystem(drake_lcm))

# LCM iiwa command receiver.
iiwa_cmd_sub = builder.AddSystem(
    LcmSubscriberSystem.Make(
        channel="IIWA_COMMAND", lcm_type=lcmt_iiwa_command, lcm=drake_lcm
    )
)

cmd_2_vec = CmdLcm2VecSystem()
builder.AddSystem(cmd_2_vec)
builder.Connect(iiwa_cmd_sub.get_output_port(0), cmd_2_vec.command_input_port)
builder.Connect(
    cmd_2_vec.cmd_left_output_port,
    robot_internal_controllers[model_l_iiwa].joint_angle_commanded_input_port,
)
builder.Connect(
    cmd_2_vec.cmd_right_output_port,
    robot_internal_controllers[model_r_iiwa].joint_angle_commanded_input_port,
)

# LCM iiwa status publisher.
iiwa_status_pub = builder.AddSystem(
    LcmPublisherSystem.Make(
        channel="IIWA_STATUS",
        lcm_type=lcmt_iiwa_status,
        lcm=drake_lcm,
        publish_period=0.005,
    )
)

status_2_lcm = StatusVec2LcmSystem(q_sim)
builder.AddSystem(status_2_lcm)
builder.Connect(
    status_2_lcm.status_output_port, iiwa_status_pub.get_input_port(0)
)
builder.Connect(plant.get_state_output_port(), status_2_lcm.x_input_port)
builder.Connect(
    iiwa_cmd_sub.get_output_port(0), status_2_lcm.iiwa_cmd_input_port
)

# Publish q on lcm_scope.
demux_mbp = Demultiplexer([plant.num_positions(), plant.num_velocities()])
builder.AddSystem(demux_mbp)
builder.Connect(plant.get_state_output_port(), demux_mbp.get_input_port(0))
LcmScopeSystem.AddToBuilder(
    builder=builder,
    lcm=drake_lcm,
    signal=demux_mbp.get_output_port(0),
    channel=kQEstimatedChannelName,
    publish_period=0.005,
)

diagram = builder.Build()
render_system_with_graphviz(diagram, "mock_station.gz")


#%% Run sim.
sim = Simulator(diagram)
context = sim.get_context()

# Fix internal controller feedforward torque ports.
for model_a in q_sim.get_actuated_models():
    controller = robot_internal_controllers[model_a]
    controller.tau_feedforward_input_port.FixValue(
        controller.GetMyContextFromRoot(context),
        np.zeros(controller.tau_feedforward_input_port.size()),
    )

# Initial state for plants.
q_a0 = np.zeros(14)

q_a0[:7] = [0.393, np.pi / 2, np.pi / 2, 1.258, 0, -0.327, np.pi / 4 * 3]
# q_a0[:7] = [0, np.pi / 2, np.pi / 2, 0, 0, 0, 0]
q_a0[7:] = [-0.353, np.pi / 2, np.pi / 2, -1.569, 0, 1.462, np.pi / 4 * 3]

q0 = np.zeros(plant.num_positions())
q_a0[q_sim.get_q_a_indices_into_q()] = q_a0

if has_objects:
    q_u0 = np.array([0.676, 0, 0, -0.736, 0.528, 0.045, 0.25])
    q_u0[:4] /= np.linalg.norm(q_u0[:4])
    q0[q_sim.get_q_u_indices_into_q()] = q_u0

# Set initial positions for plant.
context_plant = plant.GetMyContextFromRoot(context)
plant.SetPositions(context_plant, q0)

# initial lcm iiwa command message.
iiwa_cmd_msg = lcmt_iiwa_command()
iiwa_cmd_msg.num_joints = len(q_a0)
iiwa_cmd_msg.joint_position = q_a0
iiwa_cmd_msg.joint_torque = np.zeros(14)
context_cmd_sub = iiwa_cmd_sub.GetMyContextFromRoot(context)
context_cmd_sub.SetAbstractState(0, iiwa_cmd_msg)

sim.set_target_realtime_rate(1.0)
sim.AdvanceTo(np.inf)
