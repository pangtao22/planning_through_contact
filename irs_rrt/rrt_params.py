from irs_mpc.irs_mpc_params import BundleMode


class RrtParams:
    """
    Base tree parameters class. Only "yaml'-able parameters should be stored
    in the parameters.
    """
    def __init__(self):
        self.max_size = 100
        self.goal = None  # q_goal.
        self.root_node = None
        self.subgoal_prob = 0.5
        self.termination_tolerance = 0.1
        self.rewire = False


class IrsRrtParams(RrtParams):
    def __init__(self, q_model_path, joint_limits):
        super().__init__()
        # Options for computing bundled dynamics.
        self.h = 0.1
        self.n_samples = 100
        self.q_model_path = q_model_path
        self.std_u = 0.1

        # kFirst and kExact are supported.
        self.bundle_mode = BundleMode.kFirst

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

        # Supports local, local_u, global, global_u
        self.distance_metric = "global"
        self.global_metric = None  # only used when distance_metric is "global".

