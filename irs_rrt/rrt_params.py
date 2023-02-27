import numpy as np
from irs_mpc2.irs_mpc_params import SmoothingMode


class RrtParams:
    """
    Base tree parameters class. Only "yaml'-able parameters should be stored
    in the parameters.
    """

    def __init__(self):
        self.max_size = 100
        self.goal = None  # q_goal.
        self.root_node = None
        self.goal_as_subgoal_prob = 0.5
        self.termination_tolerance = 0.1
        # TODO: rewire is not properly supported. Bad things can happen if it
        #  is set to True. We need to either remove or support this.
        self.rewire = False
        self.stepsize = 0.1
        # TODO (pang): the QuasistaticSimulator does not support joint limits
        #  yet. Therefore the distance metric does not reflect being close
        #  to joint limits. As a result, clipping the commanded positions
        #  using the joint limits without a distance metric that supports
        #  joint limits will lead to inefficient exploration.
        #  enforce_robot_joint_limits should be False in order to reproduce
        #  the results in the TR-O paper. But it should be set to True for
        #  hardware demos, such as the iiwa_bimanual example.
        self.enforce_robot_joint_limits = False


class IrsRrtParams(RrtParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__()
        # Options for computing bundled dynamics.
        self.h = 0.1
        self.n_samples = 100
        self.q_model_path = q_model_path
        self.std_u = 0.1

        self.smoothing_mode = None

        # When self.bundle_mode == BundleMode.kFirstAnalytic,
        #  this log_barrier_weight is used in
        #  ReachableSet.calc_bundled_Bc_analytic.
        self.log_barrier_weight_for_bundling = 100

        # State-space limits for sampling, provided as a bounding box.
        # During tree expansion, samples that go outside of this limit will be
        # rejected, implicitly enforcing the constraint.
        self.joint_limits = joint_limits

        # Regularization for computing inverse of covariance matrices.
        # NOTE(terry-suh): Note that if the covariance matrix is singular,
        # then the Mahalanobis distance metric is infinity. One interpretation
        # of the regularization term is to cap the infinity distance to some
        # value that scales with inverse of regularization.
        self.regularization = 1e-6

        # Stepsize.
        # TODO(terry-suh): the selection of this parameter should be automated.
        self.stepsize = 0.3

        # When set to True, this field ensures that Gurobi and Mosek are not
        # used anywhere.
        self.use_free_solvers = False


class IrsRrtProjectionParams(IrsRrtParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__(q_model_path, joint_limits)
        # Subgoals further away from any node in the tree than
        # distance_threshold will be rejected.
        self.distance_threshold = np.inf
        self.grasp_prob = 0.2
