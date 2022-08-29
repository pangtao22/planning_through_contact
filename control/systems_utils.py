from typing import Callable

from pydrake.all import (LeafSystem, AbstractValue, BasicVector, DrakeLcm,
                         DiagramBuilder, LcmSubscriberSystem,
                         LcmInterfaceSystem, Simulator)
from drake import lcmt_scope


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def wait_for_msg(channel_name: str,  lcm_type, is_message_good: Callable):
    d_lcm = DrakeLcm()

    builder = DiagramBuilder()

    sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=channel_name,
            lcm_type=lcm_type,
            lcm=d_lcm))
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
            sub.GetMyContextFromRoot(sim.get_context()))
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
            "lcmt_scope_msg",
            AbstractValue.Make(lcmt_scope))
        self.output_port = self.DeclareVectorOutputPort(
            "q", BasicVector(n_q), self.calc_q)

    def calc_q(self, context, output):
        msg = self.input_port.Eval(context)
        assert msg.size == self.n_q
        output.SetFromVector(msg.value)
