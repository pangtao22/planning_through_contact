import copy
import numpy as np

from pydrake.all import (LeafSystem, AbstractValue, PortDataType, BasicVector,
                         GurobiSolver)
import pydrake.solvers.mathematicalprogram as mp

from qsim_cpp import ForwardDynamicsMode, GradientMode
from qsim.parser import QuasistaticParser


class Controller:
    def __init__(self, q_parser: QuasistaticParser,
                 control_period: float):
        self.q_sim = q_parser.make_simulator_cpp()
        self.solver = GurobiSolver()

        # TODO: do not hardcode these parameters. They need to be consistent
        #  with the trajectory optimizer that generates these trajectories.
        p = copy.deepcopy(q_parser.q_sim_params)
        p.h = control_period
        p.forward_mode = ForwardDynamicsMode.kLogIcecream
        p.gradient_mode = GradientMode.kBOnly
        p.log_barrier_weight = 5000
        self.sim_params = p

        self.Qu = 1e2 * np.diag([10, 10, 10, 10, 1, 1, 1.])
        self.R = np.diag(1e-1 * np.ones(self.q_sim.num_actuated_dofs()))

    def calc_linearization(self,
                           q_nominal: np.ndarray,
                           u_nominal: np.ndarray):
        idx_q_u_into_q = self.q_sim.get_q_u_indices_into_q()
        q_next_nominal = self.q_sim.calc_dynamics(
            q_nominal, u_nominal, self.sim_params)
        B = self.q_sim.get_Dq_nextDqa_cmd()
        n_u = self.q_sim.num_unactuated_dofs()
        Au = np.eye(n_u)
        Bu = B[idx_q_u_into_q, :]
        cu = (q_next_nominal[idx_q_u_into_q] - q_nominal[idx_q_u_into_q]
              - Bu @ u_nominal)

        return Au, Bu, cu

    def calc_u(self, q_nominal: np.ndarray, u_nominal: np.ndarray,
               q: np.ndarray):
        idx_q_u_into_q = self.q_sim.get_q_u_indices_into_q()
        q_u_nominal = q_nominal[idx_q_u_into_q]
        q_u = q[idx_q_u_into_q]
        q_a = q[self.q_sim.get_q_a_indices_into_q()]
        Au, Bu, cu = self.calc_linearization(q_nominal, u_nominal)

        n_u = len(idx_q_u_into_q)
        n_a = self.q_sim.num_actuated_dofs()

        prog = mp.MathematicalProgram()
        q_u_next = prog.NewContinuousVariables(n_u, "q_u_+")
        u = prog.NewContinuousVariables(n_a, "u")

        # TODO: q_u_ref should be q_u_nominal_+?
        prog.AddQuadraticErrorCost(self.Qu, q_u_nominal, q_u_next)
        prog.AddQuadraticErrorCost(self.R, u_nominal, u)
        prog.AddLinearEqualityConstraint(
            np.hstack([-np.eye(n_u), Bu]), -(q_u + cu),
            np.hstack([q_u_next, u]))

        result = self.solver.Solve(prog)
        if not result.is_success():
            raise RuntimeError("QP controller failed.")

        return result.GetSolution(u)


class ControllerSystem(LeafSystem):
    def __init__(self, control_period: float,
                 x0_nominal: np.ndarray,
                 q_parser: QuasistaticParser,
                 closed_loop: bool):
        super().__init__()
        self.set_name("allegro_controller")
        # Periodic state update
        self.control_period = control_period
        self.closed_loop = closed_loop
        self.DeclarePeriodicDiscreteUpdate(control_period)
        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(x0_nominal)

        self.controller = Controller(
            q_parser=q_parser, control_period=control_period)
        self.q_sim = self.controller.q_sim
        self.plant = self.q_sim.get_plant()

        self.q_ref_input_port = self.DeclareInputPort(
            "q_ref",
            PortDataType.kVectorValued,
            self.plant.num_positions())

        self.u_ref_input_port = self.DeclareInputPort(
            "u_ref",
            PortDataType.kVectorValued,
            self.q_sim.num_actuated_dofs())

        self.q_input_port = self.DeclareInputPort(
            "q", PortDataType.kVectorValued, self.plant.num_positions())

        self.position_cmd_output_ports = {}
        model_to_indices_map = self.q_sim.get_position_indices()

        for model in self.q_sim.get_actuated_models():
            nq = self.plant.num_positions(model)
            name = self.plant.GetModelInstanceName(model)

            def calc_output(context, output):
                output.SetFromVector(
                context.get_discrete_state().value()[
                    model_to_indices_map[model]])

            self.position_cmd_output_ports[model] = (
                self.DeclareVectorOutputPort(
                    f'{name}_cmd', BasicVector(nq), calc_output))

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        super().DoCalcDiscreteVariableUpdates(context, events, discrete_state)
        q_nominal = self.q_ref_input_port.Eval(context)
        u_nominal = self.u_ref_input_port.Eval(context)
        q = self.q_input_port.Eval(context)

        if self.closed_loop:
            u = self.controller.calc_u(
                q_nominal=q_nominal, u_nominal=u_nominal, q=q)
        else:
            u = u_nominal

        q_nominal[self.q_sim.get_q_a_indices_into_q()] = u
        discrete_state.set_value(q_nominal)
