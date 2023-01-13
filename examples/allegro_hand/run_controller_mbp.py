import os
import copy
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    Quaternion,
    AngleAxis,
    Simulator,
    PiecewisePolynomial,
    RigidTransform,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir, add_package_paths_local

from control.drake_sim import (
    load_ref_trajectories,
    make_controller_mbp_diagram,
    kQTrjSrcName,
    kUTrjSrcName,
)
from control.systems_utils import render_system_with_graphviz

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from allegro_hand_setup import (
    robot_name,
    object_name,
    q_model_path_hardware,
    controller_params,
)

kPtcDataRepoPath = Path.home()


def create_allegro_controller_plant(gravity: np.ndarray):
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    add_package_paths_local(parser)
    ProcessModelDirectives(
        LoadModelDirectives(os.path.join(models_dir, "allegro_hand.yml")),
        plant,
        parser,
    )
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    plant.Finalize()

    return plant


def get_quaternion(q_unnormalized: np.ndarray):
    return Quaternion(q_unnormalized / np.linalg.norm(q_unnormalized))


# %%
h_ref_knot = 0.2
h_ctrl = 0.02
controller_params.control_period = h_ctrl

q_parser = QuasistaticParser(q_model_path_hardware)
q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()
dim_q = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()

diagram_and_contents = make_controller_mbp_diagram(
    q_parser_mbp=q_parser,
    q_sim_mbp=q_sim,
    q_sim_q_control=q_sim,
    t_knots=None,
    u_knots_ref=np.zeros((dim_u, 1)),
    q_knots_ref=np.zeros((dim_q, 1)),
    controller_params=controller_params,
    create_controller_plant_functions={
        robot_name: create_allegro_controller_plant
    },
    closed_loop=False,
)

# unpack return values.
diagram = diagram_and_contents["diagram"]
controller_robots = diagram_and_contents["controller_robots"]
robot_internal_controllers = diagram_and_contents["robot_internal_controllers"]
plant = diagram_and_contents["plant"]
meshcat_vis = diagram_and_contents["meshcat_vis"]
loggers_cmd = diagram_and_contents["loggers_cmd"]
logger_x = diagram_and_contents["logger_x"]
meshcat = diagram_and_contents["meshcat"]
q_trj_src = diagram.GetSubsystemByName(kQTrjSrcName)
u_trj_src = diagram.GetSubsystemByName(kUTrjSrcName)

# render_system_with_graphviz(diagram)
model_a = plant.GetModelInstanceByName(robot_name)
model_u = plant.GetModelInstanceByName(object_name)

# %% Run sim.
# Load trajectories.
q_knots_ref_list, u_knots_ref_list, t_knots_list = load_ref_trajectories(
    file_path="hand_optimized_q_and_u_trj.pkl", v_limit=0.1
)

for idx_segment, (q_knots_ref, u_knots_ref, t_knots) in enumerate(
    zip(q_knots_ref_list, u_knots_ref_list, t_knots_list)
):
    q_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_ref.T)
    u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)
    q_trj_src.UpdateTrajectory(q_ref_trj)
    u_trj_src.UpdateTrajectory(u_ref_trj)

    context = diagram.CreateDefaultContext()
    for model_a in q_sim.get_actuated_models():
        controller = robot_internal_controllers[model_a]
        controller.tau_feedforward_input_port.FixValue(
            controller.GetMyContextFromRoot(context),
            np.zeros(controller.tau_feedforward_input_port.size()),
        )

    context_plant = plant.GetMyContextFromRoot(context)
    q0 = copy.copy(q_knots_ref[0])
    plant.SetPositions(context_plant, q0)

    context_controller = controller_robots.GetMyContextFromRoot(context)
    context_controller.SetDiscreteState(q0)

    sim = Simulator(diagram, context)
    sim.Initialize()

    AddMeshcatTriad(
        meshcat=meshcat,
        path="visualizer/sphere/sphere/frame",
        length=0.1,
        radius=0.001,
        opacity=1,
    )

    q_u_goal = q_knots_ref[-1, q_sim.get_q_u_indices_into_q()]
    AddMeshcatTriad(
        meshcat=meshcat,
        path="goal/frame",
        length=0.1,
        radius=0.003,
        opacity=0.5,
        X_PT=RigidTransform(Quaternion(q_u_goal[:4]), q_u_goal[4:]),
    )

    meshcat_vis.DeleteRecording()
    meshcat_vis.StartRecording()
    sim.AdvanceTo(t_knots[-1] * 1.05)
    meshcat_vis.PublishRecording()

    # %% plots
    # 1. cmd vs nominal u.
    logger_cmd = loggers_cmd[model_a]
    u_log = logger_cmd.FindLog(context)
    u_nominals = np.array(
        [u_ref_trj.value(t).squeeze() for t in u_log.sample_times()]
    )

    u_diff = np.linalg.norm(u_nominals[:-1] - u_log.data().T[1:], axis=1)

    # %% 2. q_u_nominal vs q_u.
    x_log = logger_x.FindLog(context)
    q_log = x_log.data()[: plant.num_positions()].T
    q_u_log = q_log[:, q_sim.get_q_u_indices_into_q()]
    angle_error = []
    position_error = []

    for i, t in enumerate(x_log.sample_times()):
        q_u_ref = q_ref_trj.value(t).squeeze()[q_sim.get_q_u_indices_into_q()]
        q_u = q_u_log[i]

        Q_WB_ref = get_quaternion(q_u_ref[:4])
        Q_WB = get_quaternion(q_u[:4])
        angle_error.append(AngleAxis(Q_WB.inverse().multiply(Q_WB_ref)).angle())
        position_error.append(np.linalg.norm(q_u_ref[4:] - q_u[4:]))

    v_data = x_log.data()[plant.num_positions() :, :]
    v_u = np.array(
        [
            plant.GetVelocitiesFromArray(model_u, v_data[:, i])
            for i in range(v_data.shape[1])
        ]
    )
    v_u_norm = np.linalg.norm(v_u, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(4, 12))
    x_axis_label = "Time steps"
    axes[0].plot(u_diff)
    axes[0].grid(True)
    axes[0].set_title("||u - u_ref||")

    axes[1].set_title("||q_u - q_u_ref||")
    axes[1].grid(True)
    axes[1].plot(angle_error, label="angle")
    axes[1].plot(position_error, label="pos")
    axes[1].legend()

    axes[2].set_title("||v_u||")
    axes[2].grid(True)
    axes[2].plot(v_u_norm)
    axes[2].axhline(np.mean(v_u_norm), linestyle="--", color="r")
    axes[2].set_xlabel(x_axis_label)
    axes[2].set_yscale("log")

    plt.show()

    #%% logging
    log_dict = {
        "q_a_ref": q_knots_ref[:, q_sim.get_q_a_indices_into_q()],
        # q_a ref trajectory for visualization.
        "q_u_ref": q_knots_ref[:, q_sim.get_q_u_indices_into_q()],
        # q_u ref trajectory,
        "u_ref": u_knots_ref,
        "t_knots": np.array(t_knots),
        "q_u_mbp": np.array(q_u_log),  # q_u mbp trajectory,
        "x_log_data": np.array(x_log.data()),  # logger state,
        "x_log_time": np.array(x_log.sample_times()),  # logger state,
    }

    filename = Path(
        Path.home(),
        "ptc_data",
        "allegro_hand",
        "sim2real",
        f"{idx_segment:03}.pkl",
    )

    with open(filename, "wb") as f:
        pickle.dump(log_dict, f)
