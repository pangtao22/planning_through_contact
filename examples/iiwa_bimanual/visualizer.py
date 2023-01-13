import time
import pickle
import os
import pathlib
import numpy as np

from pydrake.all import (
    LeafSystem,
    Meshcat,
    DrakeLcm,
    RigidTransform,
    BasicVector,
    StartMeshcat,
    DiagramBuilder,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    LcmInterfaceSystem,
    LcmSubscriberSystem,
    LcmScopeSystem,
    MeshcatVisualizer,
    Quaternion,
)

from drake import lcmt_scope, lcmt_robot_state

from qsim.parser import QuasistaticParser

from control.systems_utils import wait_for_msg, add_triad
from control.drake_sim import (
    add_mbp_scene_graph,
    load_ref_trajectories,
    calc_q_and_u_extended_and_t_knots,
)

from iiwa_bimanual_setup import q_model_path_planar, q_model_path_cylinder
from state_estimator import kQEstimatedChannelName

import lcm

kZHeight = 0.25
kGoalPoseChannel = "GOAL_POSE"
kStartPoseChannel = "START_POSE"

parser = QuasistaticParser(q_model_path_cylinder)
parser_2d = QuasistaticParser(q_model_path_planar)
q_sim = parser.make_simulator_cpp()
q_sim_2d = parser_2d.make_simulator_cpp()

meshcat = StartMeshcat()

# Build diagram with MeshcatVisualizer
builder = DiagramBuilder()
plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
    parser, builder
)
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()

# Create context for diagram.
context = diagram.CreateDefaultContext()
context_vis = visualizer.GetMyContextFromRoot(context)
context_plant = plant.GetMyContextFromRoot(context)

add_triad(
    meshcat, name="frame", prefix="goal", length=0.35, radius=0.03, opacity=0.5
)

add_triad(
    meshcat, name="frame", prefix="start", length=0.6, radius=0.005, opacity=0.7
)

add_triad(
    meshcat,
    name="frame",
    prefix="plant/box/box",
    length=0.4,
    radius=0.01,
    opacity=1,
)

# 2D rendering camera
R_WC = RotationMatrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).T)
X_WC = RigidTransform(R_WC, [0, 0, 1])
meshcat.Set2dRenderMode(X_WC)


def draw_q(channel, data):
    q_scope_msg = lcmt_scope.decode(data)
    q = np.array(q_scope_msg.value)
    plant.SetPositions(context_plant, q)
    visualizer.ForcedPublish(context_vis)
    q_u = q[q_sim.get_q_u_indices_into_q()]
    X_WB = RigidTransform(Quaternion(q_u[:4]), q_u[4:])
    meshcat.SetTransform("plant/box/box/frame", X_WB)
    time.sleep(0.03)


def transform_from_xy_theta(x: float, y: float, theta: float):
    return RigidTransform(RollPitchYaw(0, 0, theta), [x, y, kZHeight])


def draw_start_pose(channel, data):
    msg = lcmt_robot_state.decode(data)
    x = msg.joint_position[0]
    y = msg.joint_position[1]
    theta = msg.joint_position[2]
    meshcat.SetTransform("start", transform_from_xy_theta(x, y, theta))


def draw_goal_pose(channel, data):
    msg = lcmt_robot_state.decode(data)
    x = msg.joint_position[0]
    y = msg.joint_position[1]
    theta = msg.joint_position[2]
    meshcat.SetTransform("goal", transform_from_xy_theta(x, y, theta))


lc = lcm.LCM()
subscription = lc.subscribe(kQEstimatedChannelName, draw_q)
subscription.set_queue_capacity(1)

lc.subscribe(kStartPoseChannel, draw_start_pose)
lc.subscribe(kGoalPoseChannel, draw_goal_pose)

try:
    while True:
        lc.handle()

except KeyboardInterrupt:
    pass
