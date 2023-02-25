import enum

import numpy as np

from qsim_cpp import ForwardDynamicsMode


class SmoothingMode(enum.Enum):
    # use exact gradient.
    kNonePyramid = enum.auto()
    kNoneIcecream = enum.auto()
    # use randomized gradient.
    k1RandomizedPyramid = enum.auto()
    k1RandomizedIcecream = enum.auto()
    # use analytic gradient.
    k1AnalyticPyramid = enum.auto()
    k1AnalyticIcecream = enum.auto()
    # TODO: consider supporting zero-order smoothing modes.
    k0Pyramid = enum.auto()
    k0Icecream = enum.auto()


class BundleMode(enum.Enum):
    """
    This is kept around for backward compatibility with earlier versions of
    pickled RRT trees. It should NOT be used anywhere in IrsMpc.
    """

    # This is also used in IrsRrtParams to decide which smoothing scheme to use.
    kFirstRandomized = enum.auto()
    kFirstExact = enum.auto()
    kFirstAnalytic = enum.auto()

    # These have not been updated in a while and we are no longer sure if
    # they behave...
    kZeroB = enum.auto()
    kZeroAB = enum.auto()


kSmoothingMode2ForwardDynamicsModeMap = {
    SmoothingMode.kNonePyramid: ForwardDynamicsMode.kQpMp,
    SmoothingMode.kNoneIcecream: ForwardDynamicsMode.kSocpMp,
    SmoothingMode.k1RandomizedPyramid: ForwardDynamicsMode.kQpMp,
    SmoothingMode.k1RandomizedIcecream: ForwardDynamicsMode.kSocpMp,
    SmoothingMode.k1AnalyticPyramid: ForwardDynamicsMode.kLogPyramidMy,
    SmoothingMode.k1AnalyticIcecream: ForwardDynamicsMode.kLogIcecream,
    SmoothingMode.k0Pyramid: ForwardDynamicsMode.kQpMp,
    SmoothingMode.k0Icecream: ForwardDynamicsMode.kSocpMp,
}

kNoSmoothingModes = {SmoothingMode.kNonePyramid, SmoothingMode.kNoneIcecream}

k0RandomizedSmoothingModes = {
    SmoothingMode.k0Pyramid,
    SmoothingMode.k0Icecream,
}

k1RandomizedSmoothingModes = {
    SmoothingMode.k1RandomizedPyramid,
    SmoothingMode.k1RandomizedIcecream,
}

kAnalyticSmoothingModes = {
    SmoothingMode.k1AnalyticPyramid,
    SmoothingMode.k1AnalyticIcecream,
}


class IrsMpcQuasistaticParameters:
    def __init__(self):
        self.h = np.nan
        # Necessary arguments defining optimal control problem.
        self.Q_dict = None
        self.Qd_dict = None
        self.R_dict = None

        # Optional arguments defining bounds.
        self.x_bounds_abs = None
        self.u_bounds_abs = None
        self.x_bounds_rel = None
        self.u_bounds_rel = None

        # Smoothing.
        self.smoothing_mode = SmoothingMode.k1AnalyticIcecream
        self.use_A = False

        """
        When self.rollout_forward_dynamics_mode is None, the c obtained from 
         computing the smoothed A and B is used as c.
        
        When not None, the c is computed separately using the 
        forward mode specified in self.rollout_forward_dynamics_mode.
        """
        self.rollout_forward_dynamics_mode = None

        # Arguments for randomized smoothing.
        # calc_std_u is a function with two inputs (std_u_initial, iteration)
        self.calc_std_u = None
        self.std_u_initial = None
        self.n_samples_randomized = 100

        # Arguments for analytic smoothing.
        # calc_log_barrier_weight is a function with two inputs
        #  (log_barrier_weight_initial, iteration)
        self.log_barrier_weight_initial = None
        self.calc_log_barrier_weight = None

        self.enforce_joint_limits = False

        # backward compatibility
        self.bundle_mode = BundleMode.kFirstRandomized
