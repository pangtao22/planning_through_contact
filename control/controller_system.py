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
                      QuasistaticSimulatorCpp)


class Controller:
    def __init__(self, q_sim: QuasistaticSimulatorCpp,
                 control_period: float):
        self.q_sim = q_sim
        self.solver = GurobiSolver()

        # TODO: do not hardcode these parameters. They need to be consistent
        #  with the trajectory optimizer that generates these trajectories.
        p = copy.deepcopy(q_sim.get_sim_params())
        p.h = control_period
        p.forward_mode = ForwardDynamicsMode.kLogIcecream
        p.gradient_mode = GradientMode.kBOnly
        p.log_barrier_weight = 5000
        self.sim_params = p

        self.Qu = 1e2 * np.diag([10, 10, 10, 10, 1, 1, 1.])
        self.R = np.diag(1e-1 * np.ones(self.q_sim.num_actuated_dofs()))

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
    def __init__(self, control_period: float,
                 x0_nominal: np.ndarray,
                 q_sim: QuasistaticSimulatorCpp,
                 closed_loop: bool):
        super().__init__()
        self.set_name("quasistatic_controller")
        # Periodic state update
        self.control_period = control_period
        self.closed_loop = closed_loop
        self.DeclarePeriodicDiscreteUpdate(control_period)
        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(x0_nominal)

        self.controller = Controller(
            q_sim=q_sim, control_period=control_period)
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


class CommandVec2LcmSystem(LeafSystem):
    def __init__(self, q_sim: QuasistaticSimulatorCpp):
        super().__init__()
        self.set_name("command_vec_to_lcm")
        self.q_sim = q_sim
        self.q_cmd_input_port = self.DeclareInputPort(
            "q_a_cmd",
            PortDataType.kVectorValued,
            self.q_sim.num_actuated_dofs())

        self.status_input_port = self.DeclareAbstractInputPort(
            "allegro_status",
            AbstractValue.Make(lcmt_allegro_status()))

        self.cmd_output_port = self.DeclareAbstractOutputPort(
            "allegro_cmd", lambda: AbstractValue.Make(lcmt_allegro_command()),
            self.copy_allegro_cmd_out)

    def copy_allegro_cmd_out(self, context, output):
        msg = output.get_value()
        allegro_staus_msg = self.status_input_port.Eval(context)
        q_a_cmd = self.q_cmd_input_port.Eval(context)
        msg.utime = allegro_staus_msg.utime
        msg.num_joints = len(q_a_cmd)
        msg.joint_position = q_a_cmd


def add_controller_system_to_diagram(
        builder: DiagramBuilder,
        t_knots: np.ndarray,
        u_knots_ref: np.ndarray,
        q_knots_ref: np.ndarray,
        h_ctrl: float,
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
    ctrller_allegro = ControllerSystem(control_period=h_ctrl,
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
