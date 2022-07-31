import numpy as np

from pydrake.all import LeafSystem, AbstractValue, PortDataType, BasicVector

from qsim.simulator import QuasistaticSimulator
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend


class ControllerSystem(LeafSystem):
    def __init__(self, control_period: float,
                 x0_nominal: np.ndarray,
                 q_parser: QuasistaticParser):
        super().__init__()
        self.set_name("allegro_controller")
        # Periodic state update
        self.control_period = control_period
        self.DeclarePeriodicDiscreteUpdate(control_period)
        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(x0_nominal)

        self.q_sim = q_parser.make_simulator_cpp()
        self.plant = self.q_sim.get_plant()

        self.x_nominal_input_port = self.DeclareInputPort(
            "x_nominal",
            PortDataType.kVectorValued,
            self.plant.num_positions())

        self.u_nominal_input_port = self.DeclareInputPort(
            "u_nominal",
            PortDataType.kVectorValued,
            self.q_sim.num_actuated_dofs())

        self.position_cmd_ports = {}
        # Make sure that the positions vector of the plant is q = [q_u, q_a].
        assert self.q_sim.get_q_u_indices_into_q()[0] == 0
        model_to_indices_map = self.q_sim.get_position_indices()

        for model in self.q_sim.get_actuated_models():
            nq = self.plant.num_positions(model)
            name = self.plant.GetModelInstanceName(model)

            def calc_output(context, output, model=model):
                output.SetFromVector(
                context.get_discrete_state().value()[
                    model_to_indices_map[model]])

            self.position_cmd_output_ports[model] = (
                self.DeclareVectorOutputPort(
                    f'{name}_cmd', BasicVector(nq), calc_output))

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        super().DoCalcDiscreteVariableUpdates(context, events, discrete_state)

        x_nominal = self.x_nominal_input_port.Eval(context)
        u_nominal = self.u_nominal_input_port.Eval(context)

        x_nominal[self.q_sim.get_q_a_indices_into_q()] = u_nominal
        discrete_state.set_value(x_nominal)






