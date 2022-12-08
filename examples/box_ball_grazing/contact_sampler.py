import numpy as np

from qsim_cpp import QuasistaticSimulatorCpp
from qsim.simulator import QuasistaticSimulator

from irs_rrt.contact_sampler import ContactSampler

from box_ball_setup import robot_name, object_name


class BoxBallContactSampler(ContactSampler):
    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
    ):
        super().__init__(q_sim, q_sim_py)
        plant = q_sim.get_plant()
        self.idx_a = plant.GetModelInstanceByName(robot_name)
        self.idx_u = plant.GetModelInstanceByName(object_name)

    def sample_contact(self, q: np.ndarray):
        """
        Everything is in the yz plane.
        Box is square and has edge length 1m.
        The center of the box is at (-0.6, 0), and is constrained to slide
        along the y-axis.
        Ball has radius 0.1m.
        """
        l = 1.0  # edge length of the
        r = 0.1  # radius of ball
        y_u = q[self.q_sim.get_q_u_indices_into_q()][0]
        d = l - 2 * r
        y_a = y_u - d / 2 + np.random.rand() * d
        z_a = 0
        q_dict = {self.idx_a: np.array([y_a, z_a]), self.idx_u: np.array([y_u])}

        return self.q_sim.get_q_vec_from_dict(q_dict)
