import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

import cProfile

from pydrake.all import PiecewisePolynomial

from qsim_cpp import QuasistaticSimulatorCpp
from qsim.simulator import QuasistaticSimulator

from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsRrtParams
from irs_rrt.contact_sampler import ContactSampler

from planar_pushing_setup import *


class PlanarPushingContactSampler(ContactSampler):
    def __init__(self, 
        q_sim: QuasistaticSimulatorCpp, 
        q_sim_py: QuasistaticSimulator):
        super().__init__(q_sim=q_sim, q_sim_py=q_sim_py)
        """
        This class samples contact for the planar system.
        """
        self.cw = 0.6  # cspace width
        self.q_sim = q_sim

    def sample_contact(self, q):

        q_u = q[self.q_sim.get_q_u_indices_into_q()]

        # 1. Sample a contact from the surface of the box.
        side = np.random.randint(4)
        pos = self.cw * 2.0 * (np.random.rand() - 0.5)

        if side == 0:  # left side.
            contact = np.array([-self.cw, pos, 1])
        if side == 1:  # right side.
            contact = np.array([self.cw, pos, 1])
        if side == 2:  # top side.
            contact = np.array([pos, self.cw, 1])
        if side == 3:  # top side.
            contact = np.array([pos, -self.cw, 1])

        # Apply a transformation as defined by q_u.

        theta = q_u[2]
        X_WB = np.array(
            [
                [np.cos(theta), -np.sin(theta), q_u[0]],
                [np.sin(theta), np.cos(theta), q_u[1]],
                [0, 0, 1],
            ]
        )

        q_WB = X_WB.dot(contact)
        return np.hstack([q_u, q_WB[0:2]])
