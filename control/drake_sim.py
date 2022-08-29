import pickle
from typing import Callable, Dict, Set

import numpy as np
from pydrake.all import (MultibodyPlant, DiagramBuilder,
                         ConnectMeshcatVisualizer,
                         Demultiplexer, LogVectorOutput,
                         ModelInstanceIndex)
from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController)

from .controller_system import (add_controller_system_to_diagram,
                                ControllerParams)

CreateControllerPlantFunction = Callable[[np.ndarray], MultibodyPlant]


def load_ref_trajectories(file_path: str, h_ref_knot: float,
                          q_sim: QuasistaticSimulatorCpp):
    """
    If u_knots_nominal has length T, then q_knots_nominal has length T + 1.
    During execution, u_knots_nominal is prepended with q_knots_nominal[0],
    so that they have the same length.
    """
    with open(file_path, "rb") as f:
        trj_dict = pickle.load(f)
    q_knots_ref = trj_dict["x_trj"]
    u_knots_ref = trj_dict["u_trj"]

    idx_qa_into_q = q_sim.get_q_a_indices_into_q()
    T = len(u_knots_ref)
    t_knots = np.linspace(0, T, T + 1) * h_ref_knot
    # Extends u_knots_ref so that it has the same length (T + 1) as q_knots_ref.
    u_knots_ref_extended = np.vstack(
        [q_knots_ref[0, idx_qa_into_q], u_knots_ref])

    return q_knots_ref, u_knots_ref_extended, t_knots


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


def make_controller_mbp_diagram(
        q_parser: QuasistaticParser,
        q_sim: QuasistaticSimulatorCpp,
        t_knots: np.ndarray,
        u_knots_ref: np.ndarray,
        q_knots_ref: np.ndarray,
        controller_params: ControllerParams,
        create_controller_plant_functions: Dict[str,
                                                CreateControllerPlantFunction],
        closed_loop: bool):
    """
    @param: h_ref_knot: the duration (in seconds) between adjacent know
        points in the reference trajectory stored in plan_path.
    """
    builder = DiagramBuilder()

    # MBP and SceneGraph.
    plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
        q_parser, builder)

    # Add visualizer.
    meshcat_vis = ConnectMeshcatVisualizer(builder, scene_graph)

    # Impedance (PD) controller for robots, with gravity compensation.
    models_actuated = q_sim.get_actuated_models()
    robot_internal_controllers = add_internal_controllers(
        models_actuated=models_actuated,
        q_parser=q_parser,
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
        q_sim=q_sim,
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
