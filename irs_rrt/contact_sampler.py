from sys import implementation
import numpy as np
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import BundleMode

class ContactSampler:
    def __init__(self, q_dynamics: QuasistaticDynamics, n_samples: int):
        """
        Base class for sampling contact.
        """
        self.q_dynamics = q_dynamics
        self.q_sim = q_dynamics.q_sim_py
        self.n_samples = n_samples

    def sample_contact(self, q_u):
        """
        Given a q_u, return a state vector that corresponds to a grasp
        configuration.
        """
        raise NotImplementedError("This method is virtual.")
