import enum


class BundleMode(enum.Enum):
    # This is also used in IrsRrtParams to decide which smoothing scheme to use.
    kFirstRandomized = enum.auto()
    kFirstExact = enum.auto()
    kFirstAnalytic = enum.auto()

    # These have not been updated in a while and we are no longer sure if
    # they behave...
    kZeroB = enum.auto()
    kZeroAB = enum.auto()


class ParallelizationMode(enum.Enum):
    # This should never be used in practice...
    kNone = enum.auto()

    kZmq = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrj
    kCppBundledB = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledBTrjDirect
    kCppBundledBDirect = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledABTrj
    kCppBundledAB = enum.auto()

    # calls BatchQuasistaticSimulator::CalcBundledABTrjDirect
    kCppBundledABDirect = enum.auto()

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

        # Arguments related to sampling, i.e. BundleMode.kFirst.
        # calc_std_u is a function with two inputs (std_u_initial, iteration)
        self.calc_std_u = None
        self.std_u_initial = None
        self.num_samples = 100

        # Arguments for analytic smoothing, i.e. BundleMode.kFirstAnalytic.
        self.log_barrier_weight_initial = None
        self.log_barrier_weight_multiplier = None

        # Arguments related to various options.
        self.decouple_AB = True
        self.solver_name = "gurobi"
        self.publish_every_iteration = False
        self.bundle_mode = BundleMode.kFirstRandomized
        self.parallel_mode = ParallelizationMode.kCppBundledB
