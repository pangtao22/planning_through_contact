import pickle
from typing import Callable, Dict, Set, Tuple

import numpy as np
from pydrake.all import (MultibodyPlant, DiagramBuilder,
                         Demultiplexer, LogVectorOutput,
                         ModelInstanceIndex)
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.meshcat_visualizer import (ConnectMeshcatVisualizer,
                                                MeshcatContactVisualizer)

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController)

from .controller_system import (ControllerParams, ControllerSystem)
from .controller_planar_iiwa_bimanual import IiwaBimanualPlanarControllerSystem
CreateControllerPlantFunction = Callable[[np.ndarray], MultibodyPlant]


def calc_q_and_u_extended_and_t_knots(
        q_knots_ref: np.ndarray, u_knots_ref: np.ndarray, h_ref_knot: float,
        q_sim: QuasistaticSimulatorCpp):
    """
    If u_knots_ref has length T, then q_knots_ref has length T + 1.
    In this function, u_knots_ref is prepended with q_knots_ref[0],
    so that they have the same length.
    """
    idx_qa_into_q = q_sim.get_q_a_indices_into_q()
    T = len(u_knots_ref)
    t_knots = np.linspace(0, T, T + 1) * h_ref_knot
    # Extends u_knots_ref so that it has the same length (T + 1) as q_knots_ref.
    u_knots_ref_extended = np.vstack(
        [q_knots_ref[0, idx_qa_into_q], u_knots_ref])

    return q_knots_ref, u_knots_ref_extended, t_knots


def load_ref_trajectories(file_path: str, h_ref_knot: float,
                          q_sim: QuasistaticSimulatorCpp):
    with open(file_path, "rb") as f:
        trj_dict = pickle.load(f)
    q_knots_ref = trj_dict["x_trj"]
    u_knots_ref = trj_dict["u_trj"]

    return calc_q_and_u_extended_and_t_knots(
        q_knots_ref, u_knots_ref, h_ref_knot, q_sim)


def add_mbp_scene_graph(q_parser: QuasistaticParser, builder: DiagramBuilder,
                        has_objects=True, mbp_time_step=1e-4):
    object_sdf_paths = q_parser.object_sdf_paths if has_objects else {}
    plant, scene_graph, robot_models, object_models = \
        QuasistaticSimulator.create_plant_with_robots_and_objects(
            builder=builder,
            model_directive_path=q_parser.model_directive_path,
            robot_names=[name for name in q_parser.robot_stiffness_dict.keys()],
            object_sdf_paths=object_sdf_paths,
            time_step=mbp_time_step,  # Only useful for MBP simulations.
            gravity=q_parser.get_gravity())

    return plant, scene_graph, robot_models, object_models


def add_internal_controllers(
        models_actuated: Set[ModelInstanceIndex],
        plant: MultibodyPlant,
        q_parser: QuasistaticParser,
        builder: DiagramBuilder,
        controller_plant_makers: Dict[str,
                                      CreateControllerPlantFunction],
):
    robot_internal_controllers = {}
    for model_a in models_actuated:
        model_name = plant.GetModelInstanceName(model_a)
        make_plant = controller_plant_makers[model_name]

        plant_for_control = make_plant(q_parser.get_gravity())
        controller_internal = RobotInternalController(
            plant_robot=plant_for_control,
            joint_stiffness=q_parser.get_robot_stiffness_by_name(model_name),
            controller_mode="impedance")
        controller_internal.set_name(f"{model_name}_internal_controller")
        builder.AddSystem(controller_internal)

        builder.Connect(controller_internal.GetOutputPort("joint_torques"),
                        plant.get_actuation_input_port(model_a))
        builder.Connect(plant.get_state_output_port(model_a),
                        controller_internal.robot_state_input_port)

        robot_internal_controllers[model_a] = controller_internal

    return robot_internal_controllers


def make_controller_mbp_diagram(q_parser_mbp: QuasistaticParser,
                                q_sim_mbp: QuasistaticSimulatorCpp,
                                q_sim_q_control: QuasistaticSimulatorCpp,
                                t_knots: np.ndarray,
                                u_knots_ref: np.ndarray,
                                q_knots_ref: np.ndarray,
                                controller_params: ControllerParams,
                                create_controller_plant_functions: Dict[
                                    str, CreateControllerPlantFunction],
                                closed_loop: bool):
    """
    @param: h_ref_knot: the duration (in seconds) between adjacent know
        points in the reference trajectory stored in plan_path.
    """
    builder = DiagramBuilder()

    # MBP and SceneGraph.
    plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
        q_parser_mbp, builder)

    # Add visualizer.
    meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

    # Impedance (PD) controller for robots, with gravity compensation.
    models_actuated = q_sim_mbp.get_actuated_models()
    robot_internal_controllers = add_internal_controllers(
        models_actuated=models_actuated,
        q_parser=q_parser_mbp,
        plant=plant,
        builder=builder,
        controller_plant_makers=create_controller_plant_functions)

    # Add Quasistatic Robot Controller System and trajectory sources
    controller_robots, q_ref_trj, u_ref_trj = add_controller_system_to_diagram(
        builder=builder,
        t_knots=t_knots,
        u_knots_ref=u_knots_ref,
        q_knots_ref=q_knots_ref,
        controller_params=controller_params,
        q_sim_mbp=q_sim_mbp,
        q_sim_q_control=q_sim_q_control,
        closed_loop=closed_loop)

    # Demux the MBP state x := [q, v] into q and v.
    demux_mbp = Demultiplexer([plant.num_positions(), plant.num_velocities()])
    builder.AddSystem(demux_mbp)
    builder.Connect(plant.get_state_output_port(),
                    demux_mbp.get_input_port(0))
    builder.Connect(demux_mbp.get_output_port(0),
                    controller_robots.q_input_port)

    for model_a in models_actuated:
        builder.Connect(
            controller_robots.position_cmd_output_ports[model_a],
            robot_internal_controllers[
                model_a].joint_angle_commanded_input_port)

    # Contact Viusalizer

    contact_viz = MeshcatContactVisualizer(meshcat_vis, plant=plant)
    builder.AddSystem(contact_viz)
    builder.Connect(plant.get_contact_results_output_port(),
                    contact_viz.GetInputPort("contact_results"))

    # Logging
    h_ctrl = controller_params.control_period
    logger_x = LogVectorOutput(
        plant.get_state_output_port(), builder, h_ctrl)
    loggers_cmd = {}
    loggers_contact_torque = {}
    for model in models_actuated:
        loggers_cmd[model] = LogVectorOutput(
            controller_robots.position_cmd_output_ports[model], builder, h_ctrl)
        loggers_contact_torque[model] = LogVectorOutput(
            plant.get_generalized_contact_forces_output_port(model),
            builder, h_ctrl)

    diagram = builder.Build()

    return {'diagram': diagram,
            'plant': plant,
            'controller_robots': controller_robots,
            'robot_internal_controllers': robot_internal_controllers,
            'meshcat_vis': meshcat_vis,
            'logger_x': logger_x,
            'loggers_cmd': loggers_cmd,
            'q_ref_trj': q_ref_trj,
            'u_ref_trj': u_ref_trj,
            'loggers_contact_torque': loggers_contact_torque}


def add_controller_system_to_diagram(
        builder: DiagramBuilder,
        t_knots: np.ndarray,
        u_knots_ref: np.ndarray,
        q_knots_ref: np.ndarray,
        controller_params: ControllerParams,
        q_sim_mbp: QuasistaticSimulatorCpp,
        q_sim_q_control: QuasistaticSimulatorCpp,
        closed_loop: bool) -> Tuple[ControllerSystem, PiecewisePolynomial,
                                    PiecewisePolynomial]:
    """
    Adds the following three system to the diagram, and makes the following
     two connections.
    |trj_src_q| ---> |                  |
                     | ControllerSystem |
    |trj_src_u| ---> |                  |
    """
    # Create trajectory sources.
    u_ref_trj = PiecewisePolynomial.FirstOrderHold(
        t_knots, u_knots_ref.T)
    q_ref_trj = PiecewisePolynomial.FirstOrderHold(
        t_knots, q_knots_ref.T)
    trj_src_u = TrajectorySource(u_ref_trj)
    trj_src_q = TrajectorySource(q_ref_trj)
    trj_src_u.set_name("u_src")
    trj_src_q.set_name("q_src")

    # Allegro controller system.
    if q_sim_mbp == q_sim_q_control:
        q_controller = ControllerSystem(
            q_sim_mbp=q_sim_mbp,
            q_sim_q_control=q_sim_q_control,
            controller_params=controller_params,
            closed_loop=closed_loop)
    else:
        # q_sim_mbp and q_sim_q_control are different, i.e. for the planar
        # iiwa bimanual system.
        q_controller = IiwaBimanualPlanarControllerSystem(
            q_sim_2d=q_sim_q_control,
            q_sim_3d=q_sim_mbp,
            controller_params=controller_params,
            closed_loop=closed_loop)

    builder.AddSystem(trj_src_u)
    builder.AddSystem(trj_src_q)
    builder.AddSystem(q_controller)

    # Make connections.
    builder.Connect(trj_src_q.get_output_port(),
                    q_controller.q_ref_input_port)
    builder.Connect(trj_src_u.get_output_port(),
                    q_controller.u_ref_input_port)

    return q_controller, q_ref_trj, u_ref_trj
