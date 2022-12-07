import copy
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    PiecewisePolynomial,
    RotationMatrix,
    AngleAxis,
    Quaternion,
    RigidTransform,
    ModelInstanceIndex,
)
from pydrake.math import RollPitchYaw

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp, GradientMode, ForwardDynamicsMode

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_rrt.contact_sampler import ContactSampler

from iiwa_bimanual_setup import *


#%%
def calc_q(a, q_start, q_end):
    """
    0 <= a <= 1
    """
    return q_start + a * (q_end - q_start)


class ContactSamplerBimanualPlanar:
    def __init__(self):
        q_parser = QuasistaticParser(q_model_path_planar)
        self.q_sim = q_parser.make_simulator_cpp()
        self.plant = self.q_sim.get_plant()

        self.idx_a_l = self.plant.GetModelInstanceByName(iiwa_l_name)
        self.idx_a_r = self.plant.GetModelInstanceByName(iiwa_r_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)

    def has_collisions(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        # this also updates query_object.
        self.q_sim.update_mbp_positions(q_dict)
        return self.q_sim.get_query_object().HasCollisions()

    def find_contact_bisection(
        self,
        idx_a: ModelInstanceIndex,
        q_a_start: np.ndarray,
        q_a_end: np.ndarray,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        tol: float = 1e-2,
    ):
        """
        q_a_start must be collision-free.
        q_a_end must have collision.
        The returned q_dict[idx_a] is NOT (but close to be) in collision.
        """
        start = 0
        end = 1.0
        while end - start > tol:
            mid = (start + end) / 2
            q_dict[idx_a] = calc_q(mid, q_a_start, q_a_end)
            if self.has_collisions(q_dict):
                end = mid
            else:
                start = mid

        q_dict[idx_a] = calc_q(start, q_a_start, q_a_end)

        return q_dict

    def find_contact_linear(
        self,
        idx_a: ModelInstanceIndex,
        q_a_start: np.ndarray,
        q_a_end: np.ndarray,
        q_dict: Dict[ModelInstanceIndex, np.ndarray],
        n_samples: int = 11,
    ):
        """
        The returned q_dict[idx_a] is in collision.
        """

        found_contact = False
        for end in np.linspace(0, 1, n_samples):
            q_a_r = calc_q(end, q_a_start, q_a_end)
            q_dict[idx_a] = q_a_r
            if self.has_collisions(q_dict):
                found_contact = True
                break

        if end == 0.0 or not found_contact:
            raise RuntimeError

        return q_dict

    def sample_enveloping_contact(
        self,
        q_u: np.ndarray,
        idx_a: ModelInstanceIndex,
        joint0_range: List[float],
        joint1_range: List[float],
        joint2_range: List[float],
    ):
        q_dict = {
            self.idx_a_l: np.array([np.pi / 2, 0, 0]),
            self.idx_a_r: np.array([-np.pi / 2, 0, 0]),
            self.idx_u: q_u,
        }

        # First joint.
        n_trials = 0
        found = False
        while n_trials < 20:
            q1_sampled = joint1_range[0] + np.random.rand() * (
                joint1_range[1] - joint1_range[0]
            )
            q2_sampled = joint2_range[0] + np.random.rand() * (
                joint2_range[1] - joint2_range[0]
            )

            q_a_start = np.array([joint0_range[0], q1_sampled, q2_sampled])
            q_a_end = np.array([joint0_range[1], q1_sampled, q2_sampled])

            try:
                q_dict = self.find_contact_linear(
                    idx_a=idx_a,
                    q_a_start=q_a_start,
                    q_a_end=q_a_end,
                    q_dict=q_dict,
                )
                found = True
                break
            except RuntimeError:
                pass
            n_trials += 1

        if not found:
            raise RuntimeError

        q_dict = self.find_contact_bisection(
            idx_a=idx_a,
            q_a_start=q_a_start,
            q_a_end=q_dict[idx_a],
            q_dict=q_dict,
            tol=0.05,
        )

        # Second joint.
        q_a_start = q_dict[idx_a]
        q_a_end = np.copy(q_a_start)
        q_a_end[1] = joint1_range[1]

        q_dict = self.find_contact_bisection(
            idx_a=idx_a, q_a_start=q_a_start, q_a_end=q_a_end, q_dict=q_dict
        )

        return q_dict

    def sample_contact(self, q: np.ndarray):
        q_u = q[self.q_sim.get_q_u_indices_into_q()]

        joint0_range_right = np.array([-np.pi / 2, np.pi / 2])
        joint1_range_right = np.array([np.pi / 2, -np.pi / 2])
        joint2_range_right = np.array([0, np.pi / 2])

        joint0_range_left = np.array([np.pi / 2, -np.pi / 2])
        joint1_range_left = np.array([-np.pi / 2, np.pi / 2])
        joint2_range_left = np.array([0, -np.pi / 2])

        q_a_left_list = []
        q_a_right_list = []

        for _ in range(5):
            q_dict = self.sample_enveloping_contact(
                q_u,
                self.idx_a_r,
                joint0_range_right,
                joint1_range_right,
                joint2_range_right,
            )
            q_a_right_list.append(q_dict[self.idx_a_r])

            q_dict = self.sample_enveloping_contact(
                q_u,
                self.idx_a_l,
                joint0_range_left,
                joint1_range_left,
                joint2_range_left,
            )
            q_a_left_list.append(q_dict[self.idx_a_l])

        n_trials = 0
        while n_trials < 20:
            q_a_left = random.choice(q_a_left_list)
            q_a_right = random.choice(q_a_right_list)
            q_dict = {
                self.idx_a_l: q_a_left,
                self.idx_a_r: q_a_right,
                self.idx_u: q_u,
            }

            if not self.has_collisions(q_dict):
                return self.q_sim.get_q_vec_from_dict(q_dict)
            n_trials += 1

        raise RuntimeError


if __name__ == "__main__":
    contact_sampler = ContactSamplerBimanualPlanar()

    #%%
    q_parser = QuasistaticParser(q_model_path_planar)
    q_sim = q_parser.make_simulator_cpp()
    q_sim_py = q_parser.make_simulator_py(internal_vis=True)
    q_vis = QuasistaticVisualizer(q_sim, q_sim_py)
    plant = q_sim.get_plant()

    #%%
    idx_a_l = plant.GetModelInstanceByName(iiwa_l_name)
    idx_a_r = plant.GetModelInstanceByName(iiwa_r_name)
    idx_u = plant.GetModelInstanceByName(object_name)

    #%%
    q_u = np.array([0.4, 0, np.pi / 6])
    q = np.zeros(plant.num_positions())
    q[q_sim.get_q_u_indices_into_q()] = q_u

    q_vis.draw_configuration(contact_sampler.sample_contact(q))
