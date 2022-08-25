from pydrake.all import LeafSystem, AbstractValue, BasicVector
from drake import lcmt_scope


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


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
