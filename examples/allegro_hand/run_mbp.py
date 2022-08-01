import os
import pickle
import copy

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (MultibodyPlant, Parser, ProcessModelDirectives,
                         LoadModelDirectives, DiagramBuilder, TrajectorySource,
                         PiecewisePolynomial, ConnectMeshcatVisualizer,
                         Simulator, AddTriad, Demultiplexer, LogVectorOutput,
                         Quaternion, AngleAxis)

from qsim.simulator import QuasistaticSimulator
from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir, add_package_paths_local
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController)

from allegro_hand_setup import robot_name, q_model_path_hardware
from allegro_controller_system import ControllerSystem


def create_allegro_controller_plant(gravity):
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    add_package_paths_local(parser)
    ProcessModelDirectives(
        LoadModelDirectives(os.path.join(models_dir, 'allegro_hand.yml')),
        plant, parser)
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    plant.Finalize()

    return plant


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)

#%%
with open("hand_trj.pkl", "rb") as f:
    trj_dict = pickle.load(f)

'''
If u_knots_nominal has length T, then q_knots_nominal has length T + 1. 
During execution, u_knots_nominal is prepended with q_knots_nominal[0], 
so that they have the same length.
'''
q_knots_nominal = trj_dict["x_trj"]
u_knots_nominal = trj_dict["u_trj"]


#%%
q_parser = QuasistaticParser(q_model_path_hardware)
gravity = q_parser.get_gravity()

plant_allegro = create_allegro_controller_plant(gravity=gravity)

builder = DiagramBuilder()
plant, scene_graph, robot_models, object_models = \
    QuasistaticSimulator.create_plant_with_robots_and_objects(
        builder=builder,
        model_directive_path=q_parser.model_directive_path,
        robot_names=[name for name in q_parser.robot_stiffness_dict.keys()],
        object_sdf_paths=q_parser.object_sdf_paths,
        time_step=1e-4,  # Only useful for MBP simulations.
        gravity=gravity)
allegro_model = plant.GetModelInstanceByName(robot_name)

# Add visualizer.
meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

# robot trajectory source
u_knots_nominal_extended = np.vstack(
    [plant.GetPositionsFromArray(allegro_model, q_knots_nominal[0]),
     u_knots_nominal])

# Trajectory sources.
h = 0.1
T = len(u_knots_nominal)
t_knots = np.linspace(0, T, T + 1) * h
u_nominal_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_nominal_extended.T)
q_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_nominal.T)
trj_src_u = TrajectorySource(u_nominal_trj)
trj_src_q = TrajectorySource(q_trj)
trj_src_u.set_name("u_src")
trj_src_q.set_name("q_src")

# Impedance (PD) controller for the allegro hand, with gravity compensation.
# The output of controller_allegro is the 16 joint torques for the hand.
ctrller_allegro = RobotInternalController(
    plant_robot=plant_allegro,
    joint_stiffness=q_parser.get_robot_stiffness_by_name(robot_name),
    controller_mode="impedance")

# The output of controller_sys is the 16 joint position commands for the hand.
h_ctrl = 0.02
ctrller_sys = ControllerSystem(control_period=h_ctrl,
                               x0_nominal=q_knots_nominal[0], q_parser=q_parser,
                               closed_loop=True)

# Demux the MBP state x := [q, v] into q and v.
demux = Demultiplexer([plant.num_positions(), plant.num_velocities()])

# Make connections!
builder.AddSystem(trj_src_u)
builder.AddSystem(trj_src_q)
builder.AddSystem(ctrller_allegro)
builder.AddSystem(ctrller_sys)
builder.AddSystem(demux)

builder.Connect(plant.get_state_output_port(),
                demux.get_input_port(0))
builder.Connect(demux.get_output_port(0),
                ctrller_sys.q_input_port)
builder.Connect(trj_src_q.get_output_port(),
                ctrller_sys.q_ref_input_port)
builder.Connect(trj_src_u.get_output_port(),
                ctrller_sys.u_ref_input_port)

builder.Connect(ctrller_allegro.GetOutputPort("joint_torques"),
                plant.get_actuation_input_port(allegro_model))
builder.Connect(plant.get_state_output_port(allegro_model),
                ctrller_allegro.robot_state_input_port)

# TODO: change this to a for loop over all actuated model instances.
builder.Connect(ctrller_sys.position_cmd_output_ports[allegro_model],
                ctrller_allegro.joint_angle_commanded_input_port)

# Logging.
logger_cmd = LogVectorOutput(
    ctrller_sys.position_cmd_output_ports[allegro_model],
    builder, h_ctrl)
logger_x = LogVectorOutput(
    plant.get_state_output_port(), builder, h_ctrl)

diagram = builder.Build()
# render_system_with_graphviz(diagram)


#%% Run sim.
sim = Simulator(diagram)
context = sim.get_context()

context_controller = diagram.GetSubsystemContext(ctrller_allegro, context)
ctrller_allegro.tau_feedforward_input_port.FixValue(
    context_controller,
    np.zeros(ctrller_allegro.tau_feedforward_input_port.size()))

context_plant = plant.GetMyContextFromRoot(context)
q0 = copy.copy(q_knots_nominal[0])
q0[-3] += 0.002
plant.SetPositions(context_plant, q0)

sim.Initialize()

AddTriad(
    vis=meshcat_vis.vis,
    name='frame',
    prefix='drake/plant/sphere/sphere',
    length=0.1,
    radius=0.001,
    opacity=1)

# sim.set_target_realtime_rate(1.0)
meshcat_vis.reset_recording()
meshcat_vis.start_recording()
sim.AdvanceTo(u_nominal_trj.end_time() + 1.0)
meshcat_vis.publish_recording()

#%% plots
# 1. cmd vs nominal u.
u_log = logger_cmd.FindLog(context)
u_nominals = np.array(
    [u_nominal_trj.value(t).squeeze() for t in u_log.sample_times()])

u_diff = np.linalg.norm(u_nominals[:-1] - u_log.data().T[1:], axis=1)

#%% 2. q_u_nominal vs q_u.
q_sim = ctrller_sys.q_sim
x_log = logger_x.FindLog(context)
q_log = x_log.data()[:plant.num_positions()].T
q_u_log = q_log[:, q_sim.get_q_u_indices_into_q()]
angle_error = []
position_error = []


def get_quaternion(q_unnormalized: np.ndarray):
    return Quaternion(q_unnormalized / np.linalg.norm(q_unnormalized))


for i, t in enumerate(x_log.sample_times()):
    q_u_ref = q_trj.value(t).squeeze()[q_sim.get_q_u_indices_into_q()]
    q_u = q_u_log[i]

    Q_WB_ref = get_quaternion(q_u_ref[:4])
    Q_WB = get_quaternion(q_u[:4])
    angle_error.append(AngleAxis(Q_WB.inverse().multiply(Q_WB_ref)).angle())
    position_error.append(np.linalg.norm(q_u_ref[4:] - q_u[4:]))


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(u_diff)
axes[0].grid(True)
axes[0].set_title("u diff")

axes[1].set_title("q_u error")
axes[1].grid(True)
axes[1].plot(angle_error, label="angle")
axes[1].plot(position_error, label="pos")
axes[1].legend()
plt.show()






# 3. v_u, i.e. object velocity.




