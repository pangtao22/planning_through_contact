import os
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pydot

from pydrake.all import (
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    Quaternion,
    AngleAxis,
    Simulator,
    plot_system_graphviz,
    StartMeshcat,
    DiagramBuilder,
    DiagramBuilder_,
    AutoDiffXd,
    TrajectorySource_,
    PiecewisePolynomial_,
    PidController,
    ContactVisualizer_,
    MeshcatVisualizer_,
    Simulator_,
    LogVectorOutput,
    InitializeAutoDiff,
    ExtractGradient,
)
from manipulation.meshcat_utils import AddMeshcatTriad

from qsim.parser import QuasistaticParser
from qsim.model_paths import models_dir, add_package_paths_local

from control.drake_sim import (
    load_ref_trajectories,
    make_controller_mbp_diagram,
    add_mbp_scene_graph,
    calc_q_and_u_extended_and_t_knots,
)
from control.systems_utils import render_system_with_graphviz

from planar_pushing_setup import *

# %%
q_parser = QuasistaticParser(q_model_path)
q_sim = q_parser.make_simulator_cpp()
plant = q_sim.get_plant()
model_a = plant.GetModelInstanceByName(robot_name)

meshcat = StartMeshcat()

# %%
file_path = "./hand_optimized_q_and_u_trj.pkl"
with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict["q_trj_list"]
u_knots_ref_list = trj_dict["u_trj_list"]

idx_trj_segment = 0
q_knots_ref, u_knots_ref, t_knots = calc_q_and_u_extended_and_t_knots(
    q_knots_ref=q_knots_ref_list[idx_trj_segment],
    u_knots_ref=u_knots_ref_list[idx_trj_segment],
    u_knot_ref_start=q_knots_ref_list[idx_trj_segment][
        0, q_sim.get_q_a_indices_into_q()
    ],
    v_limit=0.1,
)
# %%
builder = DiagramBuilder()

# MBP and SceneGraph.
plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
    q_parser, builder
)

# %%
builder_ad = DiagramBuilder_[AutoDiffXd]()
plant_ad = plant.ToAutoDiffXd()
scene_graph_ad = scene_graph.ToAutoDiffXd()

builder_ad.AddSystem(plant_ad)
builder_ad.AddSystem(scene_graph_ad)

builder_ad.Connect(
    plant_ad.get_geometry_poses_output_port(),
    scene_graph_ad.get_source_pose_port(plant_ad.get_source_id()),
)

builder_ad.Connect(
    scene_graph_ad.get_query_output_port(),
    plant_ad.get_geometry_query_input_port(),
)

# Trajectory source
trajectory_source_ad = TrajectorySource_[AutoDiffXd](
    PiecewisePolynomial_[AutoDiffXd].FirstOrderHold(t_knots, u_knots_ref.T),
    output_derivative_order=1,
)
builder_ad.AddSystem(trajectory_source_ad)

# Critically damped. According to the sdf, the sphere has mass 1.0kg.
n_actuation = q_sim.num_actuated_dofs()
kp = q_parser.get_robot_stiffness_by_name(robot_name)
pid_controller = PidController(
    kp=kp, ki=np.zeros(n_actuation), kd=2 * np.sqrt(kp)
)
pid_controller_ad = pid_controller.ToAutoDiffXd()
builder_ad.AddSystem(pid_controller_ad)

builder_ad.Connect(
    pid_controller_ad.get_output_port(),
    plant_ad.get_actuation_input_port(model_a),
)

builder_ad.Connect(
    plant_ad.get_state_output_port(model_a),
    pid_controller_ad.GetInputPort(
        pid_controller.get_input_port_estimated_state().get_name()
    ),
)

builder_ad.Connect(
    trajectory_source_ad.get_output_port(),
    pid_controller_ad.GetInputPort(
        pid_controller.get_input_port_desired_state().get_name()
    ),
)

logger_x = LogVectorOutput(plant_ad.get_state_output_port(), builder_ad, 1e-1)

ContactVisualizer_[AutoDiffXd].AddToBuilder(builder_ad, plant_ad, meshcat)
meshcat_vis = MeshcatVisualizer_[AutoDiffXd].AddToBuilder(
    builder_ad, scene_graph_ad, meshcat
)

diagram_ad = builder_ad.Build()
render_system_with_graphviz(diagram_ad, "diagram_ad.gz")


# %% Run sim.
sim = Simulator_[AutoDiffXd](diagram_ad)
context = sim.get_context()

context_plant_ad = plant_ad.GetMyContextFromRoot(context)
q0 = InitializeAutoDiff(q_knots_ref[0])
plant_ad.SetPositions(context_plant_ad, q0)

sim.Initialize()

AddMeshcatTriad(
    meshcat=meshcat,
    path="visualizer/box/box",
    length=0.6,
    radius=0.01,
    opacity=1,
)

# sim.set_target_realtime_rate(1.0)
meshcat_vis.DeleteRecording()
meshcat_vis.StartRecording()
sim.AdvanceTo(t_knots[-1] + 1.0)
meshcat_vis.PublishRecording()

# %%
x_log = logger_x.FindLog(context)
D_x_final_D_q0 = ExtractGradient(x_log.data()[:, -1])
