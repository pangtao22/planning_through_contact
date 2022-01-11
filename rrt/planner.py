import copy
from typing import Dict
import numpy as np

from pydrake.all import ModelInstanceIndex, MathematicalProgram
import pydrake.symbolic as sym
import pydrake.solvers.mathematicalprogram as mp
from qsim.simulator import QuasistaticSimulator
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from networkx import DiGraph
from rrt.utils import pca_gaussian
import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import plotly.graph_objects as go
import plotly.io as pio


class ConfigurationSpace:
    def __init__(self, model_u: ModelInstanceIndex,
                 model_a_l: ModelInstanceIndex, model_a_r: ModelInstanceIndex,
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
            model_a_r: np.array([[-np.pi / 2, np.pi / 2], [0, np.pi / 2]])
        }
        # angle limit of antipodal grasp from horizontal
        self.grasp_angle_limit = np.array([-np.pi / 4, np.pi / 4])
        self.epsilon_joint_limit = 0.2
        # planar hand parameters
        self.r_u = 0.25
        self.r_a = 0.05
        self.l = self.r_u + self.r_a + 0.001
        self.a1 = 6 * self.r_a
        self.a2 = 4 * self.r_a
        self.right_base_pos = 0.1
        self.left_base_pos = -0.1

    def reachable_sets(self, x0, u0, q_dynamics, n_samples=2000, radius=0.2):
        du = np.random.rand(n_samples, 4) * radius * 2 - radius
        qu_samples = np.zeros((n_samples, 3))
        qa_l_samples = np.zeros((n_samples, 2))
        qa_r_samples = np.zeros((n_samples, 2))

        def save_x(x: np.ndarray):
            q_dict = q_dynamics.get_q_dict_from_x(x)
            qu_samples[i] = q_dict[self.model_u]
            qa_l_samples[i] = q_dict[self.model_a_l]
            qa_r_samples[i] = q_dict[self.model_a_r]

        for i in tqdm.tqdm(range(n_samples)):
            u = u0 + du[i]
            x_1 = q_dynamics.dynamics(x0, u)
            save_x(x_1)

        return qu_samples

    def sample_enveloping_grasp(self, q_u: np.ndarray):
        """
        For a given configuration of the ball, q_u, this function finds configurations
        of the fingers such that both fingers (and links?) contact the surface of ball.

        Suggested range of q_u:
            y: [-0.3, 0.3]
            z: [0.3, 0.8]
            theta: [-np.pi, np.pi]
        """
        # Set both fingers to the "down" configuration.
        q_dict = {
            self.model_u: q_u,
            self.model_a_l: self.joint_limits[self.model_a_l][:, 1],
            self.model_a_r: self.joint_limits[self.model_a_r][:, 0]
        }

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
                    qi_down is initialized to the upper joint limit, qi_up the
                    lower joint limit.
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

    def sample_pinch_grasp(self, q_u: np.ndarray):
        """
        For a given configuration of the ball, q_u, this function finds configurations
        of the fingers such that only finger tips contact the surface of ball.

        Suggested range of q_u:
            y: [-0.3, 0.3]
            z: [0.3, 0.8]
            theta: [-np.pi, np.pi]
        """
        # Set both fingers to the "down" configuration.
        q_dict = {
            self.model_u: q_u,
            self.model_a_l: self.joint_limits[self.model_a_l][:, 1],
            self.model_a_r: self.joint_limits[self.model_a_r][:, 0]
        }
        y = q_u[0]
        z = q_u[1]

        # Sample antipodal grasps with 15 random angles
        for _ in range(15):
            g = np.random.rand() * (self.grasp_angle_limit[
                1] - self.grasp_angle_limit[0]) + \
                self.grasp_angle_limit[0]

            right_ee_goal = np.array([
                y + np.cos(g) * self.l - self.right_base_pos,
                z + np.sin(g) * self.l
            ])
            q_dict[self.model_a_r] = self.ik_calc_joint_angle_mp(
                right_ee_goal, self.joint_limits[self.model_a_r])

            left_ee_goal = np.array([
                -y + np.cos(g) * self.l + self.left_base_pos,
                -z + np.sin(g) * self.l
            ])
            q_dict[self.model_a_l] = self.ik_calc_joint_angle_mp(
                left_ee_goal, self.joint_limits[self.model_a_l], side="left")
            if not self.has_collision(q_dict):
                return q_dict

        # If cannot find antipodal pinch grasp, decrease the angle
        # between the two contact points
        t = 0
        while True:
            ratio = np.clip(12 - 9 * t / 49, 2, 12)
            g1 = -np.random.rand() * np.pi / ratio
            g2 = np.random.rand() * np.pi / ratio
            right_ee_goal = np.array([
                y + np.cos(g1) * self.l - self.right_base_pos,
                z + np.sin(g1) * self.l
            ])
            q_dict[self.model_a_r] = self.ik_calc_joint_angle_mp(
                right_ee_goal, self.joint_limits[self.model_a_r])

            left_ee_goal = np.array([
                -y + np.cos(g2) * self.l + self.left_base_pos,
                -z + np.sin(g2) * self.l
            ])
            q_dict[self.model_a_l] = self.ik_calc_joint_angle_mp(
                left_ee_goal, self.joint_limits[self.model_a_l], side="left")
            t += 1
            if not self.has_collision(q_dict):
                return q_dict

    def ik_calc_joint_angle_mp(self, goal, joint_limits, side="right"):
        """
        Inverse Kinematics given the tip pose. Intial guess decides the upward or
        downward solution.
        """
        if side == "right":
            q_initial_guess = joint_limits[:, 1]
        else:
            q_initial_guess = joint_limits[:, 0]
        prog = MathematicalProgram()

        q = prog.NewContinuousVariables(2)
        prog.AddConstraint(self.a1 * sym.cos(q[0]) +
                           self.a2 * sym.cos(q[0] + q[1]) == goal[0])
        prog.AddConstraint(self.a1 * sym.sin(q[0]) +
                           self.a2 * sym.sin(q[0] + q[1]) == goal[1])
        prog.AddBoundingBoxConstraint(joint_limits[:, 0], joint_limits[:, 1],
                                      q)

        prog.SetInitialGuess(q, q_initial_guess)
        result = mp.Solve(prog)
        q_value = result.GetSolution(q)
        return q_value

    def check_in_contact(self, q_a, q_u):
        x_a = self.a1 * np.cos(q_a[0]) + self.a2 * np.cos(q_a[0] + q_a[1])
        y_a = self.a1 * np.sin(q_a[0]) + self.a2 * np.sin(q_a[0] + q_a[1])
        if np.linalg.norm(np.array([x_a, y_a]) - q_u[:2]) < self.l + 0.2:
            return True
        return False

    def visualize_grasp(self, q_dict):
        self.q_sim.update_mbp_positions(q_dict)
        self.q_sim.draw_current_configuration(draw_forces=False)

    def close_to_joint_limits(self, q_dict):
        # for model in [self.model_a_l, self.model_a_r]:
        #     if (q_dict[model] <= self.joint_limits[model][:,0] + self.epsilon_joint_limit).any():
        #         return True
        #     if (q_dict[model] >= self.joint_limits[model][:,1] - self.epsilon_joint_limit).any():
        #         return True
        # return False

        for model in [self.model_a_l, self.model_a_r]:
            if (q_dict[model] <= self.joint_limits[model][:, 0] -
                    self.epsilon_joint_limit).any():
                return True
            if (q_dict[model] >= self.joint_limits[model][:, 1] +
                    self.epsilon_joint_limit).any():
                return True
        return False

    def regrasp(self, q_dict, q_dynamics):
        q_regrasp = self.sample_pinch_grasp(q_dict[self.model_u])
        x0 = q_dynamics.get_x_from_q_dict(q_dict)
        xf = q_dynamics.get_x_from_q_dict(q_regrasp)
        return q_regrasp, 0, np.array([x0, xf])

    def sample_reachable_near(self,
                              node,
                              rrt,
                              method="gaussian",
                              scale_rad=np.pi,
                              n=3):
        """
        Sample a configuration of the system, with q_u being close to q[model_u] and
            q_a collision-free.
        Sample goal q_u from the reachable Gaussian.
        Sample both enveloping and pinch grasps for the same q_u.
        """
        model_u = self.model_u

        lb_u = self.joint_limits[model_u][:, 0]
        ub_u = self.joint_limits[model_u][:, 1]

        while True:
            if method == "gaussian":
                # The samples in the ellispoid is within 3 standard
                # deviation of the Gaussian
                qu_sample = np.random.multivariate_normal(
                    node.mean, node.cov / 4)
            elif method == "shell":
                qu_sample = node.sample_shell()
            elif method == "explore":
                qu_sample = rrt.sample_near_configuration(node,
                                                          lb_u,
                                                          ub_u,
                                                          num_samples=50)
            # Scale angle back to radian
            qu_sample[2] = np.clip(qu_sample[2] * scale_rad, -np.pi, np.pi)
            # Make sure the ball is not in collision with the hands
            if (qu_sample >= lb_u).all() and (qu_sample <= ub_u).all():
                break

        q_goals = []
        for i in range(n):
            while True:
                q_dict = {}
                q_dict[model_u] = qu_sample
                if i == 0:
                    q_dict = self.sample_enveloping_grasp(q_dict[model_u])
                else:
                    q_dict = self.sample_pinch_grasp(q_dict[model_u])
                if not self.has_collision(q_dict):
                    q_goals.append(q_dict)
                    break

        return q_goals

    def sample(self, t, num_tree_build, n=2):
        """
        returns a random collision-free configuration for self.qsim.plant.
        """
        model_u = self.model_u
        u_limit = self.joint_limits[model_u]
        q_dict = {
            model_u:
            np.random.rand(3) * (u_limit[:, 1] - u_limit[:, 0]) + u_limit[:, 0]
        }

        q_goals = []
        for i in range(n):
            while True:
                if i == 0:
                    q_dict = self.sample_enveloping_grasp(q_dict[model_u])
                else:
                    q_dict = self.sample_pinch_grasp(q_dict[model_u])
                if not self.has_collision(q_dict):
                    q_goals.append(q_dict)
                    break
        return q_goals

    def has_collision(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        self.q_sim.update_mbp_positions(
            q_dict)  # this also updates query_object.
        return self.q_sim.query_object.HasCollisions()


class TreeNode:
    def __init__(self,
                 q,
                 q_dynamics,
                 cspace,
                 index=1,
                 value=0,
                 q_goal=None,
                 x_waypoints=None,
                 calc_reachable=True):
        self.q = q
        self.index = index
        limits = cspace.joint_limits[cspace.model_u]
        qu = q[cspace.model_u]
        self.in_contact = False
        if calc_reachable:
            x0 = q_dynamics.get_x_from_q_dict(q)
            u0 = q_dynamics.get_u_from_q_cmd_dict(q)
            qu_samples = cspace.reachable_sets(x0, u0, q_dynamics)
            qu_mean, sigma, cov, Vh = pca_gaussian(qu_samples, scale_rad=2, r=0.35)
            # Check if hands are in contact with the ball
            # if not np.isclose(qu_samples, qu_samples[0]).all():
            # Check if ball is within the configuration limit
            if (qu >= limits[:, 0]).all() and (qu <= limits[:, 1]).all():
                if cspace.check_in_contact(
                        q[cspace.model_a_r],
                        qu - np.array([cspace.right_base_pos, 0, 0])
                ) and cspace.check_in_contact(
                        q[cspace.model_a_l],
                        -qu + np.array([cspace.left_base_pos, 0, 0])):
                # if np.linalg.det(cov)>0:
                    self.in_contact = True
            # Handle covariance singularity/numerical instability
            # Happens when the reachable set is small,e.g. the ball is too
            # far away (out of reach), fingers not in contact with the ball
            while np.linalg.det(cov) <= 0:
                cov += 1e-8
            self.mean = qu_mean
            self.sigma = sigma
            self.cov = cov
            self.vol = np.linalg.det(cov)
            self.Vh = Vh
            self.value = value
        self.q_goal = q_goal
        self.x_waypoints = x_waypoints

    def gaussian_pdf(self, x):
        return np.exp(-(x - self.mean).T @ np.linalg.inv(self.cov / 4)
                      @ (x - self.mean)) / (2 * np.pi)**(
                          len(self.sigma) / 2) / np.linalg.det(self.cov)**0.5

    def sample_shell(self, num_samples=1, scale=0.8):
        """
        Sample on a shell of ellipsoid to encourage exploration
        """
        k = np.random.rand(num_samples, 3)
        theta = k[:, 0] * 2 * np.pi
        phi = np.arccos(1 - 2 * k[:, 1])
        ratio = (1 - scale) * k[:, 2] + scale

        x = self.sigma[0] * ratio * np.sin(phi) * np.cos(theta)
        y = self.sigma[1] * ratio * np.sin(phi) * np.sin(theta)
        z = self.sigma[2] * ratio * np.cos(phi)

        p = self.Vh.T @ np.vstack((x, y, z)) + self.mean[:, None]
        p = np.squeeze(p)

        return p

    def visualize_sample(self, p):
        vis = meshcat.Visualizer()
        vis["ellipsoid"].set_object(
            g.Mesh(g.Ellipsoid(self.sigma),
                   g.MeshLambertMaterial(color=0x008000, opacity=0.3)))
        R = np.zeros((4, 4))
        R[:3, :3] = self.Vh.T
        R[3, 3] = 1
        vis["ellipsoid"].set_transform(tf.translation_matrix(self.mean).dot(R))
        vis["sample"].set_object(
            g.Points(g.PointsGeometry(p.reshape((3, p.shape[1]))),
                     g.PointsMaterial(size=0.01)))


class RRT(DiGraph):
    """
    RRT Tree.
    """

    def __init__(self, root: TreeNode, cspace: ConfigurationSpace,
                 q_dynamics: QuasistaticDynamics):
        DiGraph.__init__(self)
        self.root = root  # root TreeNode
        self.add_node(root)
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path
        self.q_dynamics = q_dynamics

    def add_tree_node(self, parent_node: TreeNode, q: Dict[ModelInstanceIndex,
                                                           np.ndarray], cost,
                      q_goal: Dict[ModelInstanceIndex,
                                   np.ndarray], x_waypoints):
        self.size += 1
        child_node = TreeNode(q, self.q_dynamics, self.cspace, self.size,
                              cost + parent_node.value, q_goal, x_waypoints)
        self.add_node(child_node)
        self.add_edge(parent_node, child_node)
        return child_node

    def sample_node(self, mode="random"):
        """
        Sample node from which to build RRT
        mode: "random": randomly pick a node on the tree;
        "explore": sample existing nodes in the tree weighted by ellipsoid
        volume;
        "new": sample more frequently newly added nodes.
        Only sample nodes where hands are in contact with the ball
        """

        if mode == "random":
            while True:
                node_sample = np.random.choice(list(self.nodes), 1)[0]
                if node_sample.in_contact:
                    return node_sample
        elif mode == "explore":
            vol_list = []
            node_list = list(self.nodes)
            for node in node_list:
                if node.in_contact:
                    vol_list.append(node.vol)
                else:
                    vol_list.append(0)
            vol_list = np.array(vol_list)
            prob = vol_list / np.sum(vol_list)
            ind = np.argmax(np.random.multinomial(1, prob))
            return node_list[ind]
        elif mode == "new":
            while True:
                prob = np.tanh((np.arange(self.size) + 1) / self.size * 2)
                prob /= np.sum(prob)
                ind = np.argmax(np.random.multinomial(1, prob))
                if list(self.nodes)[ind].in_contact:
                    return list(self.nodes)[ind]

    def sample_near_configuration(self, qi, lb_u, ub_u, num_samples=10):
        qjs = qi.sample_shell(num_samples=num_samples, scale=1)
        p = np.inf

        for j in range(num_samples):
            pj = 0
            qj = qjs[:, j]
            if (qj >= lb_u).all() and (qj <= ub_u).all():
                for node in list(self.nodes):
                    pj += node.gaussian_pdf(qj) * node.vol
                if pj < p:
                    p = pj
                    qu = qj
        return qu

    def rewire(self, qi, irs_lqr_q, T, num_iters, k=5):
        if k > self.size:
            k = self.size
        p_list = self.calc_near_nodes_prob(qi.q[self.cspace.model_u])

        ind = list(np.argpartition(p_list, -k)[-k:])
        # Remove the node itself from the parent candidate list
        ind.remove(qi.index-1)

        parent_node = list(self.predecessors(qi))[0]
        value = qi.value
        x_waypoints = qi.x_waypoints
        # Disconnect the wire from the original parent
        self.remove_edge(parent_node, qi)

        xd = self.q_dynamics.get_x_from_q_dict(qi.q)
        for i in ind:
            node = list(self.nodes)[i]
            u0 = self.q_dynamics.get_u_from_q_cmd_dict(node.q)
            irs_lqr_q.initialize_problem(x0=self.q_dynamics.get_x_from_q_dict(
                node.q),
                x_trj_d=np.tile(xd, (T + 1, 1)),
                u_trj_0=np.tile(u0, (T, 1)))

            irs_lqr_q.iterate(num_iters)

            new_value = node.value + irs_lqr_q.cost_best
            if new_value < value:
                value = new_value
                x_waypoints = irs_lqr_q.x_trj_best
                parent_node = node

        self.add_edge(parent_node, qi)
        qi.value = value
        qi.x_waypoints = x_waypoints

    def calc_near_nodes_prob(self, xi, scale_rad=2):
        p_list = []

        for node in list(self.nodes):
            x = copy.deepcopy(xi)
            x[-1] = x[-1]/scale_rad
            p_list.append(node.gaussian_pdf(x))

        return np.array(p_list)

    def get_nearest_node(self, xi):
        p_list = self.calc_near_nodes_prob(xi)
        return list(self.nodes)[np.argmax(p_list)]

    def l2_dist(self, q1, q2):
        xu1 = q1.q[self.cspace.model_u]
        xu2 = q2.q[self.cspace.model_u]
        xal1 = q1.q[self.cspace.model_a_l]
        xal2 = q2.q[self.cspace.model_a_l]
        xar1 = q1.q[self.cspace.model_a_r]
        xar2 = q2.q[self.cspace.model_a_r]
        return np.linalg.norm(xu1 - xu2)

    def visualize_meshcat(self, groupby="object"):
        """
        groupby: visualization groupby index or object type
        """
        vis = meshcat.Visualizer()
        model_u = self.cspace.model_u
        node_p = self.root.q[model_u].reshape((3, 1))
        vis['root'].set_object(
            g.Points(
                g.PointsGeometry(node_p,
                                 color=np.array([1.0, 1.0, 0.0]).reshape(
                                     (3, 1))), g.PointsMaterial(size=0.01)))

        # RRT tree traversal
        for node in list(self.nodes):
            node_p = node.q[model_u].reshape((3, 1))
            i = 0
            for child in list(self.successors(node)):
                index = child.index
                child_p = child.q[model_u].reshape((3, 1))
                goal = child.q_goal[model_u].reshape((3, 1))
                vertices = np.hstack((node_p, child_p)).astype(np.float32)
                goal_reach = np.hstack((goal, child_p)).astype(np.float32)
                if groupby == "index":
                    vis["{}/node".format(index)].set_object(
                        g.Points(g.PointsGeometry(child_p),
                                 g.PointsMaterial(size=0.01)))
                    vis["{}/goal".format(index)].set_object(
                        g.Points(
                            g.PointsGeometry(goal,
                                             color=np.array([0.0, 0.0,
                                                             1.0]).reshape(
                                                                 (3, 1))),
                            g.PointsMaterial(size=0.01)))
                    vis["{}/line".format(index)].set_object(
                        g.LineSegments(g.PointsGeometry(vertices)))
                    vis["{}/line_goal".format(index)].set_object(
                        g.LineSegments(
                            g.PointsGeometry(goal_reach,
                                             color=np.array([0.0, 0.0,
                                                             1.0]).reshape(
                                                                 (3, 1)))))
                    vis["{}/ellipsoid".format(index)].set_object(
                        g.Mesh(
                            g.Ellipsoid(child.sigma),
                            g.MeshLambertMaterial(color=0x008000,
                                                  opacity=0.3)))
                    R = np.zeros((4, 4))
                    R[:3, :3] = child.Vh.T
                    R[3, 3] = 1
                    vis["{}/ellipsoid".format(index)].set_transform(
                        tf.translation_matrix(child.mean).dot(R))
                elif groupby == "object":
                    vis["node/{}".format(index)].set_object(
                        g.Points(g.PointsGeometry(child_p),
                                 g.PointsMaterial(size=0.01)))
                    vis["goal/{}".format(index)].set_object(
                        g.Points(
                            g.PointsGeometry(goal,
                                             color=np.array([0.0, 0.0,
                                                             1.0]).reshape(
                                                                 (3, 1))),
                            g.PointsMaterial(size=0.01)))
                    vis["line/{}".format(index)].set_object(
                        g.LineSegments(g.PointsGeometry(vertices)))
                    vis["line_goal/{}".format(index)].set_object(
                        g.LineSegments(
                            g.PointsGeometry(goal_reach,
                                             color=np.array([0.0, 0.0,
                                                             1.0]).reshape(
                                                                 (3, 1)))))
                    vis["ellipsoid/{}".format(index)].set_object(
                        g.Mesh(
                            g.Ellipsoid(child.sigma),
                            g.MeshLambertMaterial(color=0x008000,
                                                  opacity=0.3)))
                    R = np.zeros((4, 4))
                    R[:3, :3] = child.Vh.T
                    R[3, 3] = 1
                    vis["ellipsoid/{}".format(index)].set_transform(
                        tf.translation_matrix(child.mean).dot(R))
                i += 1

    def visualize(self, model_u):
        pio.renderers.default = "browser"  # see plotly charts in pycharm.
        data = []
        node_p = self.root.q[model_u].reshape((3, 1))
        data.append(
            go.Scatter3d(x=node_p[0],
                         y=node_p[1],
                         z=node_p[2],
                         mode="markers",
                         marker=dict(color="red", size=5,
                                     sizemode='diameter')))

        # RRT tree traversal
        for node in list(self.nodes):
            node_p = node.q[model_u].reshape((3, 1))
            for child in list(self.successors(node)):
                child_p = child.q[model_u].reshape((3, 1))
                goal = child.q_goal[model_u].reshape((3, 1))
                # Tree Node
                data.append(
                    go.Scatter3d(x=child_p[0],
                                 y=child_p[1],
                                 z=child_p[2],
                                 mode="markers",
                                 marker=dict(color="black",
                                             size=5,
                                             sizemode='diameter')))
                # Originally generated goal configuration
                data.append(
                    go.Scatter3d(x=goal[0],
                                 y=goal[1],
                                 z=goal[2],
                                 mode="markers",
                                 marker=dict(color='blue',
                                             size=5,
                                             sizemode='diameter')))
                data.append(
                    go.Scatter3d(x=[child_p[0], goal[0]],
                                 y=[child_p[1], goal[1]],
                                 z=[child_p[2], goal[2]],
                                 marker=dict(size=0),
                                 line=dict(color='blue', dash="dash",
                                           width=2)))
                # TRee Vertices
                data.append(
                    go.Scatter3d(x=[child_p[0], node_p[0]],
                                 y=[child_p[1], node_p[1]],
                                 z=[child_p[2], node_p[2]],
                                 line=dict(color="black", dash="dash",
                                           width=2)))

        fig = go.Figure(data=data)
        fig.show()
