import enum


class IrsLqrGradientMode(enum.Enum):
    # Supports "first_order", "exact", "zero_order_B", "zero_order_AB"
    kFirst = enum.auto()
    kExact = enum.auto()
    kZeroB = enum.auto()
    kZeroAb = enum.auto()


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
        self.use_zmq_workers = True
        self.gradient_mode = IrsLqrGradientMode.kFirst
        self.solver_name = "gurobi"
        self.publish_every_iteration = False
