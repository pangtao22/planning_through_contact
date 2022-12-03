import copy
from typing import Dict, List
import random
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (PiecewisePolynomial, RotationMatrix, AngleAxis,
                         Quaternion, RigidTransform, ModelInstanceIndex)
from pydrake.math import RollPitchYaw
from pydrake.systems.meshcat_visualizer import AddTriad

from qsim.parser import QuasistaticParser
from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp, GradientMode, ForwardDynamicsMode

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_rrt.contact_sampler import ContactSampler

from iiwa_box_setup import *


#%%

# Default arm pose.
arm_pose0 = np.array([0.0, 1.1570067668968655, 0.0, -1.8186781966930226, 0.0, -0.9762174969632965, 0.0])

# Indices of the joints to be searched over.
kSampleJoints = [1, 2, 3, 4, 5]

# Sampling range
# q_lower = np.array([-np.pi/2, np.pi/8, -np.pi/8, -np.pi,  -np.pi/8, -np.pi/2, 0])
# q_upper = np.array([ np.pi/2, np.pi/2,  np.pi/8, -np.pi/4, np.pi/8,     0,    0])
q_lower = np.array([-0.0, 1.0, -0.2, -2.5, -0.2, -1.0, 0.0])
q_upper = np.array([+0.0, 2.0, +0.2, -0.5, +0.2, -0.8, 0.0])
# Configurations coming from IK
# q_lower = np.array([0.11595558069436812, 1.1498902301725908, 0.09035657128340467, -1.8600193988236655, -0.013333201234995212, -0.9771667696382608, 0.0])
# q_upper = np.array([0.11141419079949524, 1.361483358108273, 0.05245450922732749, -1.2132745063882238, -0.00732818743950596, -0.8808546105085491, 0.0])


class ContactSamplerBox:
    def __init__(self):
        q_parser = QuasistaticParser(q_model_path_no_ground)
        self.q_sim = q_parser.make_simulator_cpp()
        self.plant = self.q_sim.get_plant()

        self.idx_a = self.plant.GetModelInstanceByName(robot_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)

        q_parser_ground = QuasistaticParser(q_model_path)
        self.q_sim_ground = q_parser_ground.make_simulator_cpp(
            has_objects=False)

    def has_collisions(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        # this also updates query_object.
        self.q_sim.update_mbp_positions(q_dict)
        return self.q_sim.get_query_object().HasCollisions()

    def has_collisions_ground(self,
                              q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        self.q_sim_ground.update_mbp_positions(q_dict)
        return self.q_sim_ground.get_query_object().HasCollisions()

    def sample_contact_for_joints(self,
            joint_indices: List[int],
            q_start: np.ndarray,
            q_end: np.ndarray,
            idx_a: ModelInstanceIndex,
            q0_dict: Dict[ModelInstanceIndex, np.ndarray],
            tol: float,
            debug: bool = False):
        # sample a direction in joint space.
        n_joints = len(joint_indices)
        # d = np.random.normal(n_joints) * 0.5 + mean_joints[joint_indices]
        d = np.random.rand(n_joints)
        directions = np.array(
            [1. if q_end[i] - q_start[i] > 0 else -1 for i in joint_indices])
        for i, dir_i in enumerate(directions):
            if d[i] * dir_i < 0:
                d[i] *= -1
        max_size = min(
            abs(q_end[joint_indices] - q_start[joint_indices]) / abs(d))

        # print(abs(q_end[joint_indices] - q_start[joint_indices]))

        def calc_q(p: float):
            """
            p \in [0, 1].
            """
            q = np.copy(q_start)
            q[joint_indices] += d * p * max_size
            return q

        q_dict = copy_q_dict(q0_dict)

        # print(f"d, {d}")
        # find a configuration with contact as end.
        for end in np.linspace(0, 1, 10):
            # print(calc_q(end))
            q_dict[idx_a] = calc_q(end)
            if debug:
                q_vis.draw_configuration(q_sim.get_q_vec_from_dict(q_dict))
                input()
            if self.has_collisions(q_dict):
                break

        if end == 0 or end == 1.:
            raise RuntimeError

        start = 0
        # print(f"start {start}, end {end}")
        while end - start > tol:
            mid = (start + end) / 2
            q_dict[idx_a] = calc_q(mid)
            if self.has_collisions(q_dict):
                end = mid
            else:
                start = mid

        q_dict[idx_a] = calc_q(start)

        if self.has_collisions_ground(q_dict):
            raise RuntimeError

        return q_dict

    def sample_contact_iiwa(self, q_u: np.ndarray):
        q_dict = {self.idx_a: None, self.idx_u: q_u}

        while True:
            try:
                q_start = np.copy(q_lower)
                # if np.random.rand() > 0.5:
                #     q_start = np.copy(arm_pose0)

                q_dict[self.idx_a] = arm_pose0
                q_dict = self.sample_contact_for_joints(
                    kSampleJoints, q_start, q_upper, self.idx_a,
                    q_dict, 1e-3, False)

                break
            except RuntimeError:
                pass

        return q_dict

    def sample_contact(self, q: np.ndarray):
        q_u = q[self.q_sim.get_q_u_indices_into_q()]
        q_a_list = []
        for _ in range(5):
            q_a_list.append(
                self.sample_contact_iiwa(q_u)[self.idx_a])

        while True:
            q_a = random.choice(q_a_list)
            q_dict = {self.idx_a: q_a,
                      self.idx_u: q_u}

            if not self.has_collisions(q_dict):
                break

        return self.q_sim.get_q_vec_from_dict(q_dict)


def copy_q_dict(q0_dict):
    return {model: np.array(q) for model, q in q0_dict.items()}


if __name__ == "__main__":
    contact_sampler = ContactSamplerBox()

    #%%
    q_parser = QuasistaticParser(q_model_path_no_ground)
    q_sim = q_parser.make_simulator_cpp()
    q_sim_py = q_parser.make_simulator_py(internal_vis=True)
    q_vis = QuasistaticVisualizer(q_sim, q_sim_py)
    plant = q_sim.get_plant()

    # Initial box pose [qw, qx, qy, qz, x, y, z].
    box_pose0 = np.array([1, 0, 0, 0, 0.712, 0, 0.089])
    # Stack the configurations.
    q0 = np.hstack((arm_pose0, box_pose0))
    q_vis.draw_configuration(q0)
    input("Showing the initial pose")

    #%% Visualize sampled contacts.
    i = 0
    while True:
        if i%10==0:
            q0[-3] += 0.01
        i = i + 1
        q_dict = contact_sampler.sample_contact(q0)
        q_vis.draw_configuration(q_dict)
        print(q_dict[:7])
        input("Sample?")
