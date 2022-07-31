import os
import pickle

import numpy as np
from pydrake.all import (MultibodyPlant, Parser, ProcessModelDirectives,
                         LoadModelDirectives, DiagramBuilder, TrajectorySource,
                         PiecewisePolynomial, ConnectMeshcatVisualizer,
                         Simulator, AddTriad, Demultiplexer)

from qsim.simulator import QuasistaticSimulator
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
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

# robot trajectory source
u_knots_nominal_extended = np.vstack(
    [plant.GetPositionsFromArray(allegro_model, q_knots_nominal[0]),
     u_knots_nominal])

# Trajectory sources.
h = 0.1
T = len(u_knots_nominal)
t_knots = np.linspace(0, T, T + 1) * h
u_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_nominal_extended.T)
q_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_nominal.T)
trj_src_u = TrajectorySource(u_trj)
trj_src_q = TrajectorySource(q_trj)
trj_src_u.set_name("u_src")
trj_src_q.set_name("q_src")

# Impedance (PD) controller for the allegro hand, with gravity compensation.
# The output of controller_allegro is the 16 joint torques for the hand.
controller_allegro = RobotInternalController(
    plant_robot=plant_allegro,
    joint_stiffness=q_parser.get_robot_stiffness_by_name(robot_name),
    controller_mode="impedance")

# The output of controller_sys is the 16 joint position commands for the hand.
controller_sys = ControllerSystem(control_period=0.01,
                                  x0_nominal=q_knots_nominal[0],
                                  q_parser=q_parser)

# Demux the MBP state x := [q, v] into q and v.
demux = Demultiplexer([plant.num_positions(), plant.num_velocities()])

# Make connections!
builder.AddSystem(trj_src_u)
builder.AddSystem(trj_src_q)
builder.AddSystem(controller_allegro)
builder.AddSystem(controller_sys)
builder.AddSystem(demux)

builder.Connect(plant.get_state_output_port(),
                demux.get_input_port(0))
builder.Connect(demux.get_output_port(0),
                controller_sys.q_input_port)
builder.Connect(trj_src_q.get_output_port(),
                controller_sys.q_nominal_input_port)
builder.Connect(trj_src_u.get_output_port(),
                controller_sys.u_nominal_input_port)

builder.Connect(controller_allegro.GetOutputPort("joint_torques"),
                plant.get_actuation_input_port(allegro_model))
builder.Connect(plant.get_state_output_port(allegro_model),
                controller_allegro.robot_state_input_port)

# TODO: change this to a for loop over all actuated model instances.
builder.Connect(controller_sys.position_cmd_output_ports[allegro_model],
                controller_allegro.joint_angle_commanded_input_port)

meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)
diagram = builder.Build()
render_system_with_graphviz(diagram)


#%%
sim = Simulator(diagram)
context = sim.get_context()

context_controller = diagram.GetSubsystemContext(controller_allegro, context)
controller_allegro.tau_feedforward_input_port.FixValue(
    context_controller,
    np.zeros(controller_allegro.tau_feedforward_input_port.size()))

context_plant = plant.GetMyContextFromRoot(context)
plant.SetPositions(context_plant, q_knots_nominal[0])

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
sim.AdvanceTo(u_trj.end_time() + 0.2)
meshcat_vis.publish_recording()
