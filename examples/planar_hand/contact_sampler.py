from typing import Dict

import numpy as np
import meshcat

from qsim.simulator import QuasistaticSimulator
from pydrake.all import ModelInstanceIndex


def sample_on_sphere(radius: float, n_samples: int):
    """
    Uniform sampling on a sphere with radius r.
    http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    u = np.random.rand(n_samples, 2)  # uniform samples.
    u_theta = u[:, 0]
    u_phi = u[:, 1]
    theta = u_theta * 2 * np.pi
    phi = np.arccos(1 - 2 * u_phi)

    xyz_samples = np.zeros((n_samples, 3))
    xyz_samples[:, 0] = radius * np.sin(phi) * np.cos(theta)
    xyz_samples[:, 1] = radius * np.sin(phi) * np.sin(theta)
    xyz_samples[:, 2] = radius * np.cos(phi)

    return xyz_samples


class ContactSampler:
    """
    !!!FOR THE TWO-FINGER AND ONE BALL SYSTEM ONLY!!!
    For each model instance with n DOFs,
    joint_limits[model] is an (n, 2) array, where joint_limits[model][i, 0] is the
    lower bound of joint i and joint_limits[model][i, 1] the upper bound.
    """
    def __init__(self, model_u: ModelInstanceIndex,
                 model_a_l: ModelInstanceIndex,
                 model_a_r: ModelInstanceIndex,
                 q_sim: QuasistaticSimulator):
        self.q_sim = q_sim
        self.model_u = model_u
        self.model_a_l = model_a_l
        self.model_a_r = model_a_r
        '''
        Suggested joint limits for left and right fingers:
        Left (reversed)
        - joint 1: [-np.pi/2, np.pi/2]. lb: pointing up, ub: pointing down.
        - joint 2: [-np.pi/2, 0]: lb: pointing inwards, ub: flat.
        Right (normal)
        - joint 1: [-np.pi/2, np.pi/2]. lb: pointing down, ub: pointing up.
        - joint 2: [0, np.pi/2]: lb: flat. ub: pointing inwards.
        '''
        # joint limits
        self.joint_limits = {
            model_u: np.array([[-0.5, 0.5], [0.3, 0.6], [-np.pi, np.pi]]),
            model_a_l: np.array([[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]),
            model_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])}

    def sample_contact(self, q_u: np.ndarray):
        """
        For a given configuration of the ball, q_u, this function finds configurations
        of the fingers such that both fingers (and links?) contact the surface of ball.
        Suggested range of q_u:
            y: [-0.3, 0.3]
            z: [0.3, 0.8]
            theta: [-np.pi, np.pi]
        """
        # Set both fingers to the "down" configuration.
        q_dict = {self.model_u: q_u,
                  self.model_a_l: self.joint_limits[self.model_a_l][:, 1],
                  self.model_a_r: self.joint_limits[self.model_a_r][:, 0]}

        for model_a in [self.model_a_r, self.model_a_l]:
            # For each finger.
            qa = np.zeros(2)
            q_dict[model_a] = qa
            for i_joint in [0, 1]:
                # For each joint.
                '''
                It is assumed that qi_down is always collision-free, and qi_up is in 
                collision.
                For the right finger, 
                    qi_down is initialized to the lower joint limit, qi_up the upper 
                    joint limit.
                For the left finger,
                    qi_down is initialized to the lower joint limit, qi_up the lower 
                    joint limit.
                '''
                if model_a == self.model_a_r:
                    qi_down = self.joint_limits[model_a][i_joint, 0]
                    qi_up = self.joint_limits[model_a][i_joint, 1]
                else:
                    qi_down = self.joint_limits[model_a][i_joint, 1]
                    qi_up = self.joint_limits[model_a][i_joint, 0]

                while True:
                    qi_middle = (qi_down + qi_up) / 2
                    qa[i_joint] = qi_middle
                    if abs(qi_up - qi_down) < 1e-3:
                        # We need to return a collision-free configuration.
                        qa[i_joint] = qi_down
                        break

                    if self.has_collisions(q_dict):
                        qi_up = qi_middle
                    else:
                        qi_down = qi_middle

        return q_dict

    def has_collisions(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        # this also updates query_object.
        self.q_sim.update_mbp_positions(q_dict)
        return self.q_sim.query_object.HasCollisions()
