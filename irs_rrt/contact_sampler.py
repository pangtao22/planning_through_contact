from qsim_cpp import QuasistaticSimulatorCpp


class ContactSampler:
    def __init__(self, q_sim: QuasistaticSimulatorCpp):
        """
        Base class for sampling contact.
        """
        self.q_sim = q_sim

    def sample_contact(self, q_u):
        """
        Given a q_u, return a state vector that corresponds to a grasp
        configuration.
        """
        raise NotImplementedError("This method is virtual.")
