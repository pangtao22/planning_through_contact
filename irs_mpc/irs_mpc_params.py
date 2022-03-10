import enum


class BundleMode(enum.Enum):
    # Supports "first_order", "exact", "zero_order_B", "zero_order_AB"
    # This is also used in IrsRrt to decide which smoothing scehme to use.
    kFirst = enum.auto()
    kExact = enum.auto()
    kZeroB = enum.auto()
    kZeroAB = enum.auto()
    kFirstAnalytic = enum.auto()


class ParallelizationMode(enum.Enum):
    # This should never be used in practice...
    kNone = enum.auto()

    kZmq = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrj
    kCppBundledB = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrjDirect
    kCppBundledBDirect = enum.auto()

    # sends sampled du to a single-threaded implementation of bundled
    # gradient computation, which is least likely to have mistakes...?
    kDebug = enum.auto()

    # calls BatchQuasistaticSimulator::CalcDynamicsParallel, which is the
    # backend of BatchQuasistaticSimulator::CalcBundledBTrj.
    # This mode allows sampling in python and thus comparison between the
    # ZMQ-based implementation.
    kCppDebug = enum.auto()

    # TODO (pang) is this really necessary? It seems like comparing batch cpp
    #  against serial python is sufficient?
    # sends sampled du to the workers, instead of
    # having the works do the sampling, so that the result is deterministic.
    kZmqDebug = enum.auto()


class IrsMpcQuasistaticParameters:
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
        # calc_std_u is a function with two inputs (std_u_initial, iteration)
        self.calc_std_u = None
        self.std_u_initial = None
        self.num_samples = 100

        # Arguments related to various options.
        self.decouple_AB = True
        self.solver_name = "gurobi"
        self.publish_every_iteration = False
        self.bundle_mode = BundleMode.kFirst
        self.parallel_mode = ParallelizationMode.kCppBundledB
