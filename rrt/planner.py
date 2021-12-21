from typing import Dict
import numpy as np

from pydrake.all import ModelInstanceIndex
from qsim.simulator import QuasistaticSimulator


class ConfigurationSpace:
    def __init__(self, model_u: ModelInstanceIndex,
                 model_a_l: ModelInstanceIndex,
                 model_a_r: ModelInstanceIndex,
                 q_sim: QuasistaticSimulator):
        """
        !!!FOR THE TWO-FINGER AND ONE BALL SYSTEM ONLY!!!
        For each model instance with n DOFs,
        joint_limits[model] is an (n, 2) array, where joint_limits[model][i, 0] is the
        lower bound of joint i and joint_limits[model][i, 1] the upper bound.
        """
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

                    if self.has_collision(q_dict):
                        qi_up = qi_middle
                    else:
                        qi_down = qi_middle

        return q_dict

    def sample_near(self, q: Dict[ModelInstanceIndex, np.ndarray], r: float):
        """
        Sample a configuration of the system, with q_u being close to q[model_u] and
            q_a collision-free.
        """
        model_u = self.model_u

        qu = q[model_u]
        lb_u = np.maximum(self.joint_limits[model_u][:, 0], qu - r)
        ub_u = np.minimum(self.joint_limits[model_u][:, 1], qu + r)

        q_dict = {}
        while True:
            for model, bounds in self.joint_limits.items():
                n = len(bounds)
                if model == model_u:
                    lb = lb_u
                    ub = ub_u
                else:
                    lb = bounds[:, 0]
                    ub = bounds[:, 1]
                q_dict[model] = np.random.rand(n) * (ub - lb) + lb

            if not self.has_collision(q_dict):
                break
        return q_dict

    def sample(self):
        """
        returns a collision-free configuration for self.qsim.plant.
        """
        q_dict = {}
        while True:
            for model, bounds in self.joint_limits.items():
                n = len(bounds)
                lb = bounds[:, 0]
                ub = bounds[:, 1]
                q_model = np.random.rand(n) * (ub - lb) + lb
                q_dict[model] = q_model

            if not self.has_collision(q_dict):
                break
        return q_dict

    def dist(self, q1: Dict[ModelInstanceIndex, np.ndarray],
             q2: Dict[ModelInstanceIndex, np.ndarray]):
        d = 0
        for model in q1.keys():
            dq = q1[model] - q2[model]
            d += np.sqrt((dq**2).sum())
        return d

    def has_collision(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        self.q_sim.update_mbp_positions(q_dict)  # this also updates query_object.
        return self.q_sim.query_object.HasCollisions()


class TreeNode:
    def __init__(self, q, parent):
        self.q = q
        self.parent = parent
        self.children = []


class RRT:
    class RRT:
        """
        RRT Tree.
        """
    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_node(self, parent_node: TreeNode,
                 q_child: Dict[ModelInstanceIndex, np.ndarray]):
        child_node = TreeNode(q_child, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, q_target: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Finds the nearest node in the tree by distance to q_target in the
             configuration space.
        Args:
            q_target: dictionary of arrays representing a configuration of the MBP.
        Returns:
            closest: TreeNode. the closest node in the configuration space
                to q_target
            distance: float. distance from q_target to closest.q
        """

        def recur(node, depth=0):
            closest, distance = node, self.cspace.dist(node.q, q_target)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth+1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance
        return recur(self.root)[0]

