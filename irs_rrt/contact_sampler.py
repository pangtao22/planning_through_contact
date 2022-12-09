from qsim_cpp import QuasistaticSimulatorCpp
from qsim.simulator import QuasistaticSimulator


class ContactSampler:
    def __init__(
        self, q_sim: QuasistaticSimulatorCpp, q_sim_py: QuasistaticSimulator
    ):
        """
        Base class for sampling contact.
        """
        self.q_sim = q_sim
        # TODO: q_sim_py is needed only because it has an internal
        #  visualizer. Consider supporting visualization in
        #  QuasistaticSimulatorCpp.
        self.q_sim_py = q_sim_py

    def sample_contact(self, q):
        """
        Given a q, return a state vector that corresponds to a grasp
        configuration.
        """
        raise NotImplementedError("This method is virtual.")
