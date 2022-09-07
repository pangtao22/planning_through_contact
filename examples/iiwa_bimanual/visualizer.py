import time
import pickle
import os
import pathlib
import numpy as np

from pydrake.all import (LeafSystem, Meshcat, DrakeLcm, RigidTransform,
                         BasicVector, StartMeshcat, DiagramBuilder,
                         RollPitchYaw,
                         Simulator, LcmInterfaceSystem, LcmSubscriberSystem,
                         LcmScopeSystem, MeshcatVisualizerCpp, Quaternion)

from drake import lcmt_scope

from qsim.parser import QuasistaticParser

from control.systems_utils import wait_for_msg, add_triad
from control.drake_sim import (add_mbp_scene_graph, load_ref_trajectories,
                               calc_q_and_u_extended_and_t_knots)

from iiwa_bimanual_setup import q_model_path_planar, q_model_path_cylinder
from state_estimator import kQEstimatedChannelName

import lcm

parser = QuasistaticParser(q_model_path_cylinder)
parser_2d = QuasistaticParser(q_model_path_planar)
q_sim = parser.make_simulator_cpp()
q_sim_2d = parser_2d.make_simulator_cpp()

meshcat = StartMeshcat()

# Build diagram with MeshcatVisualizer
builder = DiagramBuilder()
plant, scene_graph, robot_models, object_models = add_mbp_scene_graph(
    parser, builder)
visualizer = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()

# Create context for diagram.
context = diagram.CreateDefaultContext()
context_vis = visualizer.GetMyContextFromRoot(context)
context_plant = plant.GetMyContextFromRoot(context)

# Draw frames for box and goal.
file_path = os.path.join(
    str(pathlib.Path(__file__).parent.resolve()),
    "./hand_optimized_q_and_u_trj.pkl")

with open(file_path, "rb") as f:
    trj_dict = pickle.load(f)

q_knots_ref_list = trj_dict['q_trj_list']
u_knots_ref_list = trj_dict['u_trj_list']
idx_trj_segment = 0
q_knots_ref = q_knots_ref_list[idx_trj_segment]

z_height = 0.25
q_u_goal = q_knots_ref[-1, q_sim_2d.get_q_u_indices_into_q()]
X_WG = RigidTransform(RollPitchYaw(0, 0, q_u_goal[2]),
                      [q_u_goal[0], q_u_goal[1], z_height])
add_triad(meshcat, name='frame', prefix="goal", length=0.4, radius=0.03,
          opacity=0.5)
meshcat.SetTransform("goal", X_WG)

q_u_start = q_knots_ref[0, q_sim_2d.get_q_u_indices_into_q()]
X_WS = RigidTransform(RollPitchYaw(0, 0, q_u_start[2]),
                      [q_u_start[0], q_u_start[1], z_height])
add_triad(meshcat, name='frame', prefix="start", length=0.6, radius=0.02,
          opacity=0.7)
meshcat.SetTransform("start", X_WS)

add_triad(meshcat, name='frame', prefix='plant/box/box',
          length=0.4, radius=0.01, opacity=1)


def draw_q(channel, data):
    q_scope_msg = lcmt_scope.decode(data)
    q = np.array(q_scope_msg.value)
    plant.SetPositions(context_plant, q)
    visualizer.Publish(context_vis)
    q_u = q[q_sim.get_q_u_indices_into_q()]
    X_WB = RigidTransform(Quaternion(q_u[:4]), q_u[4:])
    meshcat.SetTransform('plant/box/box/frame', X_WB)
    time.sleep(0.03)


lc = lcm.LCM()
subscription = lc.subscribe(kQEstimatedChannelName, draw_q)
subscription.set_queue_capacity(1)

try:
    while True:
        lc.handle()

except KeyboardInterrupt:
    pass




