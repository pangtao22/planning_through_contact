import time
from typing import Tuple
import copy
import numpy as np

from pydrake.all import (LeafSystem, AbstractValue, PortDataType, BasicVector,
                         GurobiSolver, DiagramBuilder, TrajectorySource,
                         PiecewisePolynomial)
import pydrake.solvers.mathematicalprogram as mp
from drake import lcmt_allegro_status, lcmt_allegro_command

from qsim_cpp import (ForwardDynamicsMode, GradientMode,
                      QuasistaticSimulatorCpp, QuasistaticSimParameters)


class ControllerParams:
    def __init__(self,
                 forward_mode: ForwardDynamicsMode,
                 gradient_mode: GradientMode,
                 log_barrier_weight: float,
                 Qu: np.ndarray,
                 R: np.ndarray,
                 control_period: float):
        self.forward_mode = forward_mode
        self.gradient_mode = gradient_mode
        self.log_barrier_weight = log_barrier_weight
        self.Qu = Qu
        self.R = R
        self.control_period = control_period


class Controller:
    def __init__(self, q_sim: QuasistaticSimulatorCpp,
                 controller_params: ControllerParams):
        self.q_sim = q_sim
        self.solver = GurobiSolver()

        # TODO: do not hardcode these parameters. They need to be consistent
        #  with the trajectory optimizer that generates these trajectories.
        p = copy.deepcopy(q_sim.get_sim_params())
        p.h = controller_params.control_period
        p.forward_mode = controller_params.forward_mode
        p.gradient_mode = controller_params.gradient_mode
        p.log_barrier_weight = controller_params.log_barrier_weight
        self.sim_params = p

        self.Qu = controller_params.Qu
        self.R = controller_params.R

        # joint limits
        self.lower_limits, self.upper_limits = self.get_joint_limits_vec()

    def get_joint_limits_vec(self):
        joint_limits = self.q_sim.get_actuated_joint_limits()
        n_qa = self.q_sim.num_actuated_dofs()
        model_to_idx_map = self.q_sim.get_position_indices()

        lower_limits = np.zeros(n_qa)
        upper_limits = np.zeros(n_qa)
        for model in self.q_sim.get_actuated_models():
            indices = model_to_idx_map[model]
            lower_limits[indices] = joint_limits[model]["lower"]
            upper_limits[indices] = joint_limits[model]["upper"]

        return lower_limits, upper_limits

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

        # joint limits
        prog.AddBoundingBoxConstraint(self.lower_limits, self.upper_limits, u)

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
    def __init__(self,
                 x0_nominal: np.ndarray,
                 q_sim: QuasistaticSimulatorCpp,
                 controller_params: ControllerParams,
                 closed_loop: bool):
        super().__init__()
        self.set_name("quasistatic_controller")
        # Periodic state update
        self.control_period = controller_params.control_period
        self.closed_loop = closed_loop
        self.DeclarePeriodicDiscreteUpdate(self.control_period)
        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(x0_nominal)

        self.controller = Controller(
            q_sim=q_sim, controller_params=controller_params)
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

            def calc_output(context, output, model=model):
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


def add_controller_system_to_diagram(
        builder: DiagramBuilder,
        t_knots: np.ndarray,
        u_knots_ref: np.ndarray,
        q_knots_ref: np.ndarray,
        controller_params: ControllerParams,
        q_sim: QuasistaticSimulatorCpp,
        closed_loop: bool) -> Tuple[ControllerSystem, PiecewisePolynomial,
                                    PiecewisePolynomial]:
    """
    Adds the following three system to the diagram, and makes the following
     two connections.
    |trj_src_q| ---> |                  |
                     | ControllerSystem |
    |trj_src_u| ---> |                  |
    """
    # Create trajectory sources.
    u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)
    q_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_ref.T)
    trj_src_u = TrajectorySource(u_ref_trj)
    trj_src_q = TrajectorySource(q_ref_trj)
    trj_src_u.set_name("u_src")
    trj_src_q.set_name("q_src")

    # Allegro controller system.
    ctrller_allegro = ControllerSystem(controller_params=controller_params,
                                       x0_nominal=q_knots_ref[0],
                                       q_sim=q_sim,
                                       closed_loop=closed_loop)
    builder.AddSystem(trj_src_u)
    builder.AddSystem(trj_src_q)
    builder.AddSystem(ctrller_allegro)

    # Make connections.
    builder.Connect(trj_src_q.get_output_port(),
                    ctrller_allegro.q_ref_input_port)
    builder.Connect(trj_src_u.get_output_port(),
                    ctrller_allegro.u_ref_input_port)

    return ctrller_allegro, q_ref_trj, u_ref_trj
