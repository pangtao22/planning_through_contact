from irs_mpc.quasistatic_dynamics import QuasistaticDynamics


class ContactSampler:
    def __init__(self, q_dynamics: QuasistaticDynamics):
        """
        Base class for sampling contact.
        """
        self.q_dynamics = q_dynamics
        self.q_sim = q_dynamics.q_sim

    def sample_contact(self, q_u):
        """
        Given a q_u, return a state vector that corresponds to a grasp
        configuration.
        """
        raise NotImplementedError("This method is virtual.")
