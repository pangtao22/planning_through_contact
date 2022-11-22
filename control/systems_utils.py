from typing import Callable

import numpy as np
from pydrake.all import (
    LeafSystem,
    AbstractValue,
    BasicVector,
    DrakeLcm,
    DiagramBuilder,
    LcmSubscriberSystem,
    LcmInterfaceSystem,
    Simulator,
)
from drake import lcmt_scope
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.geometry import Meshcat, Rgba, Cylinder
from pydrake.math import RigidTransform


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file."""
    from graphviz import Source

    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def wait_for_msg(channel_name: str, lcm_type, is_message_good: Callable):
    d_lcm = DrakeLcm()

    builder = DiagramBuilder()

    sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=channel_name, lcm_type=lcm_type, lcm=d_lcm
        )
    )
    builder.AddSystem(LcmInterfaceSystem(d_lcm))
    diag = builder.Build()
    sim = Simulator(diag)

    print(f"Waiting for first msg on {channel_name}...")
    while True:
        n_msgs = d_lcm.HandleSubscriptions(10)
        if n_msgs == 0:
            continue

        sim.reset_context(diag.CreateDefaultContext())
        sim.AdvanceTo(1e-1)
        msg = sub.get_output_port(0).Eval(
            sub.GetMyContextFromRoot(sim.get_context())
        )
        if is_message_good(msg):
            break
    print("Message received!")
    return msg


class QReceiver(LeafSystem):
    def __init__(self, n_q: int):
        super().__init__()
        self.set_name("q_receiver")
        self.n_q = n_q
        self.input_port = self.DeclareAbstractInputPort(
            "lcmt_scope_msg", AbstractValue.Make(lcmt_scope)
        )
        self.output_port = self.DeclareVectorOutputPort(
            "q", BasicVector(n_q), self.calc_q
        )

    def calc_q(self, context, output):
        msg = self.input_port.Eval(context)
        assert msg.size == self.n_q
        output.SetFromVector(msg.value)


def add_triad(
    vis: Meshcat, name: str, prefix: str, length=1.0, radius=0.04, opacity=1.0
):
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    Args:
        vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        prefix: (string) name of the node in the meshcat tree to which this
            triad is added.
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array(
        [[length / 2, 0, 0], [0, length / 2, 0], [0, 0, length / 2]]
    )

    axes_name = ["x", "y", "z"]
    axes_color = [
        Rgba(1, 0, 0, opacity),
        Rgba(0, 1, 0, opacity),
        Rgba(0, 0, 1, opacity),
    ]
    rotation_axes = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

    for i in range(3):
        path = f"{prefix}/{name}/{axes_name[i]}"
        vis.SetObject(path, Cylinder(radius, length), axes_color[i])
        X = RigidTransform(AngleAxis(np.pi / 2, rotation_axes[i]), delta_xyz[i])
        vis.SetTransform(path, X)
