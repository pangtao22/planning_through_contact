import enum

import numpy as np

from qsim_cpp import ForwardDynamicsMode


class SmoothingMode(enum.Enum):
    # use exact gradient.
    kNonePyramid = enum.auto()
    kNoneIcecream = enum.auto()
    # use randomized gradient.
    kFirstRandomizedPyramid = enum.auto()
    kFirstRandomizedIcecream = enum.auto()
    # use analytic gradient.
    kFirstAnalyticPyramid = enum.auto()
    kFirstAnalyticIcecream = enum.auto()
    # TODO: consider supporting zero-order smoothing modes.


kSmoothingMode2ForwardDynamicsModeMap = {
    SmoothingMode.kNonePyramid: ForwardDynamicsMode.kQpMp,
    SmoothingMode.kNoneIcecream: ForwardDynamicsMode.kSocpMp,
    SmoothingMode.kFirstRandomizedPyramid: ForwardDynamicsMode.kQpMp,
    SmoothingMode.kFirstRandomizedIcecream: ForwardDynamicsMode.kSocpMp,
    SmoothingMode.kFirstAnalyticPyramid: ForwardDynamicsMode.kLogPyramidMy,
    SmoothingMode.kFirstAnalyticIcecream: ForwardDynamicsMode.kLogIcecream}

RandomizedSmoothingModes = {
    SmoothingMode.kFirstRandomizedPyramid,
    SmoothingMode.kFirstRandomizedIcecream}

AnalyticSmoothingModes = {
    SmoothingMode.kFirstAnalyticPyramid,
    SmoothingMode.kFirstAnalyticIcecream}


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
        self.smoothing_mode = SmoothingMode.kFirstAnalyticIcecream
        self.use_A = False

        '''
        When self.rollout_forward_dynamics_mode is None, the c obtained from 
         computing the smoothed A and B is used as c.
        
        When not None, the c is computed separately using the 
        forward mode specified in self.rollout_forward_dynamics_mode.
        '''
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


