import copy
import numpy as np

from pydrake.all import LeafSystem, PortDataType, BasicVector, GurobiSolver, eq
from pydrake.all import MathematicalProgram as mp

from qsim_cpp import ForwardDynamicsMode, GradientMode, QuasistaticSimulatorCpp


class ControllerParams:
    def __init__(
        self,
        forward_mode: ForwardDynamicsMode,
        gradient_mode: GradientMode,
        log_barrier_weight: float,
        Qu: np.ndarray,
        R: np.ndarray,
        control_period: float,
        joint_limit_padding: float = 0.0,
    ):
        self.forward_mode = forward_mode
        self.gradient_mode = gradient_mode
        self.log_barrier_weight = log_barrier_weight
        self.Qu = Qu
        self.R = R
        self.control_period = control_period
        self.joint_limit_padding = joint_limit_padding


class Controller:
    def __init__(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        q_sim: QuasistaticSimulatorCpp,
        controller_params: ControllerParams,
    ):
        self.q_sim = q_sim
        self.plant = q_sim.get_plant()
        self.solver = GurobiSolver()
        self.n_q = self.plant.num_positions()

        # q_nominal is NOT a placeholder with 1 vector only.
        if q_nominal.shape[1] > 1:
            self.q_nominal = q_nominal
            self.u_nominal = u_nominal
            self.q_segment_lengths = np.linalg.norm(
                q_nominal[1:] - q_nominal[:-1], axis=1
            )
            self.q_segment_lengths_cumsum_normalized = np.cumsum(
                self.q_segment_lengths
            ) / np.sum(self.q_segment_lengths)

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
        self.lower_limits, self.upper_limits = self.get_joint_limits_vec(
            controller_params.joint_limit_padding
        )

        self.u_prev = None

    def get_joint_limits_vec(self, padding: float):
        """
        Padding \in [0, 1]. (1 - padding) of the joint limits are used.
        """
        joint_limits = self.q_sim.get_actuated_joint_limits()
        n_q = self.plant.num_positions()
        model_to_idx_map = self.q_sim.get_position_indices()

        lower_limits = np.zeros(n_q)
        upper_limits = np.zeros(n_q)
        for model in self.q_sim.get_actuated_models():
            indices = model_to_idx_map[model]
            lower_original = joint_limits[model]["lower"]
            upper_original = joint_limits[model]["upper"]
            joint_midpoint = (lower_original + upper_original) / 2
            joint_range = (upper_original - lower_original) * (1 - padding)
            lower_limits[indices] = joint_midpoint - joint_range / 2
            upper_limits[indices] = joint_midpoint + joint_range / 2

        indices_q_a_into_q = self.q_sim.get_q_a_indices_into_q()
        lower_limits = lower_limits[indices_q_a_into_q]
        upper_limits = upper_limits[indices_q_a_into_q]

        return lower_limits, upper_limits

    def find_closest_on_nominal_path(self, q: np.ndarray):
        t, indices = self.calc_t_and_indices_for_q(q)
        q_t, u_t = self.calc_q_and_u_on_arc(t, indices)
        return q_t, u_t, t, indices

    def calc_arc_length(self, t, indices):
        idx0, idx1 = indices
        q_t = t * self.q_nominal[idx0] + (1 - t) * self.q_nominal[idx1]
        l = np.sum(self.q_segment_lengths[:idx0])
        l += np.linalg.norm(q_t - self.q_nominal[idx0])

        return l / np.sum(self.q_segment_lengths)

    def calc_q_and_u_from_arc_length(self, s: float):
        idx = 0
        n_segments = len(self.q_segment_lengths_cumsum_normalized)
        while (
            idx < n_segments
            and self.q_segment_lengths_cumsum_normalized[idx] < s
        ):
            idx += 1
        if idx == n_segments:
            return self.q_nominal[-1], self.u_nominal[-1]

        if idx > 0:
            fraction = s - self.q_segment_lengths_cumsum_normalized[idx - 1]
        else:
            fraction = s

        l = np.linalg.norm(self.q_nominal[idx + 1] - self.q_nominal[idx])
        t = fraction / l  # 0 and 1.
        return self.calc_q_and_u_on_arc(t, (idx, idx + 1))

    def calc_t_and_indices_for_q(self, q):
        distances = np.linalg.norm(q - self.q_nominal, axis=1)
        distances_and_indices = sorted(
            [(d, i) for i, d in enumerate(distances)]
        )
        indices_closest = sorted(
            [distances_and_indices[0][1], distances_and_indices[1][1]]
        )

        prog = mp.MathematicalProgram()
        t = prog.NewContinuousVariables(1, "t")[0]
        q_t = prog.NewContinuousVariables(self.n_q, "q_t")
        q0 = self.q_nominal[indices_closest[0]]
        q1 = self.q_nominal[indices_closest[1]]

        prog.AddBoundingBoxConstraint(0, 1, t)
        prog.AddLinearConstraint(eq(t * q0 + (1 - t) * q1, q_t))
        prog.AddQuadraticErrorCost(np.eye(self.n_q), q, q_t)

        result = self.solver.Solve(prog)

        t_value = result.GetSolution(t)

        return t_value, indices_closest

    def calc_q_and_u_on_arc(self, t: float, indices):
        """
        t \in [0, 1] describes the convex combination of two points:
         t * self.q_nominal[indices[0]] + (1 - t) * self.q_nominal[indices[1]].
        delta_t > 0.
        """
        q0 = self.q_nominal[indices[0]]
        q1 = self.q_nominal[indices[1]]

        u0 = self.u_nominal[indices[0]]
        u1 = self.u_nominal[indices[1]]

        q_t = t * q0 + (1 - t) * q1
        u_t = t * u0 + (1 - t) * u1

        return q_t, u_t

    def calc_linearization(self, q_nominal: np.ndarray, u_nominal: np.ndarray):
        idx_q_u_into_q = self.q_sim.get_q_u_indices_into_q()
        q_next_nominal = self.q_sim.calc_dynamics(
            q_nominal, u_nominal, self.sim_params
        )
        B = self.q_sim.get_Dq_nextDqa_cmd()
        n_u = self.q_sim.num_unactuated_dofs()
        Au = np.eye(n_u)
        Bu = B[idx_q_u_into_q, :]
        cu = (
            q_next_nominal[idx_q_u_into_q]
            - q_nominal[idx_q_u_into_q]
            - Bu @ u_nominal
        )

        return Au, Bu, cu

    def calc_u(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        q: np.ndarray,
        q_goal: np.ndarray,
        u_goal: np.ndarray,
    ):
        idx_q_u_into_q = self.q_sim.get_q_u_indices_into_q()
        q_u_goal = q_goal[idx_q_u_into_q]
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
        if self.u_prev is not None:
            prog.AddQuadraticErrorCost(self.R * 0.5, self.u_prev, u)
        else:
            prog.AddQuadraticErrorCost(self.R * 0.5, q_a, u)

        # TODO: q_u_ref should be q_u_nominal_+?
        prog.AddQuadraticErrorCost(self.Qu, q_u_goal, q_u_next)
        prog.AddQuadraticErrorCost(self.R, u_goal, u)
        prog.AddLinearEqualityConstraint(
            np.hstack([-np.eye(n_u), Bu]), -(q_u + cu), np.hstack([q_u_next, u])
        )

        result = self.solver.Solve(prog)
        if not result.is_success():
            raise RuntimeError("QP controller failed.")
        u_value = result.GetSolution(u)
        self.u_prev = u_value
        return u_value


class ControllerSystem(LeafSystem):
    def __init__(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        q_sim_mbp: QuasistaticSimulatorCpp,
        q_sim_q_control: QuasistaticSimulatorCpp,
        controller_params: ControllerParams,
        closed_loop: bool,
    ):
        super().__init__()
        self.q_sim = q_sim_mbp
        self.plant = self.q_sim.get_plant()

        self.set_name("quasistatic_controller")
        # Periodic state update
        self.control_period = controller_params.control_period
        self.closed_loop = closed_loop
        self.DeclarePeriodicDiscreteUpdateNoHandler(self.control_period)

        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(BasicVector(self.plant.num_positions()))
        self.controller = Controller(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            q_sim=q_sim_q_control,
            controller_params=controller_params,
        )

        self.q_ref_input_port = self.DeclareInputPort(
            "q_ref",
            PortDataType.kVectorValued,
            q_sim_q_control.get_plant().num_positions(),
        )

        self.u_ref_input_port = self.DeclareInputPort(
            "u_ref",
            PortDataType.kVectorValued,
            q_sim_q_control.num_actuated_dofs(),
        )

        self.q_input_port = self.DeclareInputPort(
            "q", PortDataType.kVectorValued, self.plant.num_positions()
        )

        self.position_cmd_output_ports = {}
        model_to_indices_map = self.q_sim.get_position_indices()

        for model in self.q_sim.get_actuated_models():
            nq = self.plant.num_positions(model)
            name = self.plant.GetModelInstanceName(model)

            def calc_output(context, output, model=model):
                output.SetFromVector(
                    context.get_discrete_state().value()[
                        model_to_indices_map[model]
                    ]
                )

            self.position_cmd_output_ports[
                model
            ] = self.DeclareVectorOutputPort(
                f"{name}_cmd", BasicVector(nq), calc_output
            )

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        super().DoCalcDiscreteVariableUpdates(context, events, discrete_state)
        q_goal = self.q_ref_input_port.Eval(context)
        u_goal = self.u_ref_input_port.Eval(context)

        if self.closed_loop:
            # TODO: The "control" we were trying to do is not useful.
            #  Consider removal.
            q = self.q_input_port.Eval(context)
            (
                q_nominal,
                u_nominal,
                t_value,
                indices_closest,
            ) = self.controller.find_closest_on_nominal_path(q)
            # q_goal, u_goal = self.calc_along_arc(t_value, indices_closest, 0.05)

            u = self.controller.calc_u(
                q_nominal=q_nominal,
                u_nominal=u_nominal,
                q=q,
                q_goal=q_goal,
                u_goal=u_goal,
            )
        else:
            q_nominal = q_goal
            u = u_goal

        q_nominal[self.q_sim.get_q_a_indices_into_q()] = u
        discrete_state.set_value(q_nominal)
