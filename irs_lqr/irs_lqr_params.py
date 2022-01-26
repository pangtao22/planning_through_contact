import enum


class IrsLqrGradientMode(enum.Enum):
    # Supports "first_order", "exact", "zero_order_B", "zero_order_AB"
    kFirst = enum.auto()
    kExact = enum.auto()
    kZeroB = enum.auto()
    kZeroAb = enum.auto()


class IrsLqrParallelizationMode(enum.Enum):
    kNone = enum.auto()

    kZmq = enum.auto()
    # sends sampled du to the workers, instead of
    # having the works do the sampling, so that the result is deterministic.
    kZmqDebug = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrj
    kCppBundledB = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrjDirect
    kCppBundledBDirect = enum.auto()

    # calls BatchQuasistaticSimulator::CalcDynamicsParallel, which is the
    # backend of BatchQuasistaticSimulator::CalcBundledBTrj.
    # This mode allows sampling in python and thus comparison between the
    # ZMQ-based implementation.
    kCppDebug = enum.auto()


class IrsLqrQuasistaticParameters:
    def __init__(self):
        # Necessary arguments defining optimal control problem.
        self.Q_dict = None
        self.Qd_dict = None
        self.R_dict = None
        self.x0 = None
        self.x_trj_d = None
        self.u_trj_0 = None
        self.T = None

        # Optional arguments defining bounds.
        self.x_bounds_abs = None
        self.u_bounds_abs = None
        self.x_bounds_rel = None
        self.u_bounds_rel = None

        # Necessary arguments related to sampling.
        self.sampling = None
        self.std_u_initial = None
        self.num_samples = 100

        # Arguments related to various options.
        self.decouple_AB = True
        self.solver_name = "gurobi"
        self.publish_every_iteration = False
        self.gradient_mode = IrsLqrGradientMode.kFirst
        self.parallel_mode = IrsLqrParallelizationMode.kZmq
