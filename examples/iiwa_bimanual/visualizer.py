import time

import numpy as np

from pydrake.all import (LeafSystem, Meshcat, DrakeLcm, RigidTransform,
                         BasicVector, StartMeshcat, DiagramBuilder,
                         Simulator, LcmInterfaceSystem, LcmSubscriberSystem,
                         LcmScopeSystem, MeshcatVisualizerCpp, Quaternion)

from drake import lcmt_scope

from manipulation.meshcat_cpp_utils import AddMeshcatTriad

from qsim.parser import QuasistaticParser

from control.systems_utils import wait_for_msg, add_triad
from control.drake_sim import add_mbp_scene_graph, load_ref_trajectories

from iiwa_bimanual_setup import q_model_path
from state_estimator import kQEstimatedChannelName

import lcm

parser = QuasistaticParser(q_model_path)
q_sim = parser.make_simulator_cpp()

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
q_knots_ref, _, _ = load_ref_trajectories("hand_trj.pkl", 0.1, q_sim)
q_u_goal = q_knots_ref[-1, q_sim.get_q_u_indices_into_q()]
X_WG = RigidTransform(Quaternion(q_u_goal[:4]), q_u_goal[4:])
add_triad(meshcat, name='frame', prefix="goal", length=0.4, radius=0.03,
          opacity=0.5)
meshcat.SetTransform("goal", X_WG)

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




