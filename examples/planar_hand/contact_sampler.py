import logging
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import ModelInstanceIndex

from qsim_cpp import QuasistaticSimulatorCpp
from qsim.simulator import QuasistaticSimulator
from irs_rrt.contact_sampler import ContactSampler

from .planar_hand_setup import robot_l_name, robot_r_name, object_name


class PlanarHandContactSampler(ContactSampler):
    """
    !!!FOR THE TWO-FINGER AND ONE BALL SYSTEM ONLY!!!
    For each model instance with n DOFs,
    joint_limits[model] is an (n, 2) array, where joint_limits[model][i, 0] is
     the lower bound of joint i and joint_limits[model][i, 1] the upper bound.
    """

    def __init__(
        self,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
        pinch_prob: float,
    ):
        super().__init__(q_sim, q_sim_py)

        self.pinch_prob = pinch_prob
        n2i_map = self.q_sim.get_model_instance_name_to_index_map()
        self.model_u = n2i_map[object_name]
        self.model_a_l = n2i_map[robot_l_name]
        self.model_a_r = n2i_map[robot_r_name]
        """
        Suggested joint limits for left and right fingers:
        Left (reversed)
        - joint 1: [-np.pi/2, np.pi/2]. lb: pointing up, ub: pointing down.
        - joint 2: [-np.pi/2, 0]: lb: pointing inwards, ub: flat.
        Right (normal)
        - joint 1: [-np.pi/2, np.pi/2]. lb: pointing down, ub: pointing up.
        - joint 2: [0, np.pi/2]: lb: flat. ub: pointing inwards.
        """
        # joint limits
        self.joint_limits = {
            self.model_u: np.array([[-0.3, 0.3], [0.3, 0.5], [-np.pi, np.pi]]),
            self.model_a_l: np.array(
                [[-np.pi / 2, np.pi / 2], [-np.pi / 2, 0]]
            ),
            self.model_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]]),
        }

        # some constants for solving finger IK.
        self.r_u = 0.25
        self.r_a = 0.05
        self.r = self.r_u + self.r_a + 0.001
        self.l1 = 6 * self.r_a  # length of the first link.
        self.l2 = 4 * self.r_a  # length of the second link.
        self.p_WBr = np.array([0, 0.1, 0])  # right base
        self.p_WBl = np.array([0, -0.1, 0])  # left base.

    def calc_enveloping_grasp(self, q_u: np.ndarray):
        """
        For a given configuration of the ball, q_u, this function finds
        configurations of the fingers such that both fingers (and links?)
        contact the surface of ball.
        Suggested range of q_u:
            y: [-0.3, 0.3]
            z: [0.3, 0.8]
            theta: [-np.pi, np.pi]
        """
        # Set both fingers to the "down" configuration.
        q_dict = {
            self.model_u: q_u,
            self.model_a_l: self.joint_limits[self.model_a_l][:, 1],
            self.model_a_r: self.joint_limits[self.model_a_r][:, 0],
        }

        for model_a in [self.model_a_r, self.model_a_l]:
            # For each finger.
            qa = np.zeros(2)
            q_dict[model_a] = qa
            for i_joint in [0, 1]:
                # For each joint.
                """
                It is assumed that qi_down is always collision-free, and qi_up
                is in collision.
                For the right finger,
                    qi_down is initialized to the lower joint limit, qi_up the
                    upper joint limit.
                For the left finger,
                    qi_down is initialized to the lower joint limit, qi_up the
                    lower joint limit.
                """
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
        return self.q_sim.get_query_object().HasCollisions()

    def sample_contact_points_in_workspace(
        self, p_WB: np.ndarray, p_WO: np.ndarray, n_samples: int, arm: str
    ):
        """
        p_WB: (3,): world frame coordinates of the origin of the robot's base
            link.
        p_WO: (3,): world frame coordinates of the center of the object.
        """
        a = self.r
        c = self.l1 + self.l2
        b = np.linalg.norm(p_WO - p_WB)
        if a + c < b:
            # a, b, c do not form the three edges of a triangle.
            return None
        gamma = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

        dp = p_WB - p_WO
        alpha = np.arctan2(dp[2], dp[1])

        if arm == "left":
            lb = alpha - gamma
            ub = min(alpha + gamma, -np.pi / 2)
        elif arm == "right":
            lb = max(alpha - gamma, -np.pi / 2)
            ub = alpha + gamma
        else:
            raise RuntimeError(f"undefined arm {arm}")

        theta = np.random.rand(n_samples) * (ub - lb) + lb
        p_WC = np.tile(p_WO, (n_samples, 1))
        p_WC[:, 1] += np.cos(theta) * self.r
        p_WC[:, 2] += np.sin(theta) * self.r

        return p_WC

    def plot_sampled_points(
        self, p_WO: np.ndarray, p_WCl: np.ndarray, p_WCr: np.ndarray
    ):
        """
        p_WO: object origin coordinates in world frame.
        p_WCl: contact points of the left arm
        """
        co = plt.Circle(p_WO[1:], self.r, color="r", alpha=0.1)
        cl = plt.Circle(self.p_WBl[1:], self.l1 + self.l2, color="g", alpha=0.1)
        cr = plt.Circle(self.p_WBr[1:], self.l1 + self.l2, color="b", alpha=0.1)
        plt.scatter(p_WCl[:, 1], p_WCl[:, 2], color="g")
        plt.scatter(p_WCr[:, 1], p_WCr[:, 2], color="b")
        plt.gca().add_patch(cl)
        plt.gca().add_patch(cr)
        plt.gca().add_patch(co)
        plt.axis("equal")
        plt.show()

    def sample_pinch_grasp(
        self, q_u: np.ndarray, n_samples: int, show_debug_plot=False
    ):
        # Sample antipodal grasps with 10 random angles between -np.pi / 4
        # and np.pi / 4.
        p_WO = np.array([0, q_u[0], q_u[1]])
        p_WCr = self.sample_contact_points_in_workspace(
            p_WB=self.p_WBr, p_WO=p_WO, n_samples=n_samples, arm="right"
        )
        p_WCl = self.sample_contact_points_in_workspace(
            p_WB=self.p_WBl, p_WO=p_WO, n_samples=n_samples, arm="left"
        )

        if p_WCl is None or p_WCr is None:
            raise RuntimeError("Left or right robot cannot reach the sphere.")

        if show_debug_plot:
            self.plot_sampled_points(p_WO, p_WCl, p_WCr)

        q_a_l_batch = self.solve_left_arm_ik((p_WCl - self.p_WBl)[:, 1:])
        q_a_l_batch[:, 0] = -(np.pi - q_a_l_batch[:, 0])
        q_a_r_batch = self.solve_right_arm_ik((p_WCr - self.p_WBr)[:, 1:])

        # get rid of joint angles in collision or out of joint limits.
        q_dict = {self.model_u: q_u}
        # left arm.
        lb = self.joint_limits[self.model_a_l][:, 0]
        ub = self.joint_limits[self.model_a_l][:, 1]
        q_dict[self.model_a_r] = np.zeros(2)
        q_a_l_valid = []
        for q_a_l in q_a_l_batch:
            q_dict[self.model_a_l] = q_a_l
            if (
                np.all(q_a_l >= lb)
                and np.all(q_a_l <= ub)
                and not self.has_collisions(q_dict)
            ):
                q_a_l_valid.append(q_a_l)

        # right arm.
        lb = self.joint_limits[self.model_a_r][:, 0]
        ub = self.joint_limits[self.model_a_r][:, 1]
        q_dict[self.model_a_l] = np.zeros(2)
        q_a_r_valid = []
        for q_a_r in q_a_r_batch:
            q_dict[self.model_a_r] = q_a_r
            if (
                np.all(q_a_r >= lb)
                and np.all(q_a_r <= ub)
                and not self.has_collisions(q_dict)
            ):
                q_a_r_valid.append(q_a_r)

        if len(q_a_l_valid) * len(q_a_r_valid) == 0:
            raise RuntimeError("No valid pinch grasp is found.")
        # TODO: mix and match good samples (even with the sample from
        #  enveloping grasp) to generate more diverse grasps.
        return [
            {self.model_u: q_u, self.model_a_l: q_a_l, self.model_a_r: q_a_r}
            for q_a_l, q_a_r in zip(q_a_l_valid, q_a_r_valid)
        ]

    def solve_right_arm_ik(self, yz_targets):
        """
        "elbow right"
        """
        theta = np.zeros_like(yz_targets)

        y = yz_targets[:, 0]
        z = yz_targets[:, 1]
        l1 = self.l1
        l2 = self.l2
        gamma = np.arctan2(z, y)

        theta[:, 1] = np.arccos(
            (y**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        )
        beta = np.arctan2(
            l2 * np.sin(theta[:, 1]), l1 + l2 * np.cos(theta[:, 1])
        )
        theta[:, 0] = gamma - beta

        return theta

    def solve_left_arm_ik(self, yz_targets):
        """
        "elbow left"
        """
        theta = np.zeros_like(yz_targets)

        y = yz_targets[:, 0]
        z = yz_targets[:, 1]
        l1 = self.l1
        l2 = self.l2
        gamma = np.arctan2(z, y)

        theta[:, 1] = np.arccos(
            (y**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
        )
        beta = np.arctan2(
            l2 * np.sin(theta[:, 1]), l1 + l2 * np.cos(theta[:, 1])
        )
        theta[:, 0] = gamma + beta
        theta[:, 1] *= -1

        return theta

    def cointoss_for_grasp(self):
        return 1 if np.random.rand() > self.pinch_prob else 0

    def sample_contact(self, q: np.ndarray):
        """
        Given a q, sample a grasp using the contact sampler.
        """
        q_u = q[self.q_sim.get_q_u_indices_into_q()]
        pinch_grasp = self.cointoss_for_grasp()
        if pinch_grasp:
            try:
                q_dict = self.sample_pinch_grasp(q_u, n_samples=50)[0]
            except RuntimeError as err:
                logging.warning(err)
                # In this case, the quality of the enveloping grasp is
                # expected to be quite bad, i.e. degenerate
                # reachability ellipsoid. This node will be considered
                # distance from other subgoals and not being connected to,
                # i.e. becoming a "minor" node.
                q_dict = self.calc_enveloping_grasp(q_u)
        else:
            q_dict = self.calc_enveloping_grasp(q_u)

        return self.q_sim.get_q_vec_from_dict(q_dict)


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
