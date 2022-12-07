from typing import Dict
import random
import numpy as np
import time

from pydrake.all import (
    InverseKinematics,
    ModelInstanceIndex,
    Quaternion,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SolutionResult,
    SnoptSolver,
)

from qsim.parser import QuasistaticParser

from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from iiwa_box_setup import *

# %%
# Select the edges to be sampled.
# Edges 0, 1, 2, and 3 represent the front (facing -x), right (facing -y),
# back (facing +x) and left (facing +y) edges in the nominal pose.
EDGES_FOR_SAMPLING = [0, 1, 2, 3]

# Define the arm pose to be used as the initial guess for the IK.
q_arm0 = np.array([0.0, 1.157, 0.0, -1.819, 0.0, -0.976, 0.0])

# Define box properties.
BOX_SIZE = np.array([0.276, 0.198, 0.176])
box_h_size = BOX_SIZE / 2.0
# The height of the horizontal cross section to be sampled.
Z_LEVEL = box_h_size[2] - 5e-2
# Corners of the horizontal cross section.
c1 = np.array([-box_h_size[0], +box_h_size[1], Z_LEVEL])
c2 = np.array([-box_h_size[0], -box_h_size[1], Z_LEVEL])
c3 = np.array([+box_h_size[0], -box_h_size[1], Z_LEVEL])
c4 = np.array([+box_h_size[0], +box_h_size[1], Z_LEVEL])
# The vertices of the horizontal edges of the box.
edges = []
edges.append([c1, c2])
edges.append([c2, c3])
edges.append([c3, c4])
edges.append([c4, c1])
# Nominal surface normals for each edge.
edge_normals = []
edge_normals.append(RollPitchYaw(0.0, np.pi / 2, 0.0))
edge_normals.append(RollPitchYaw(0.0, np.pi / 2, np.pi / 2))
edge_normals.append(RollPitchYaw(0.0, np.pi / 2, np.pi))
edge_normals.append(RollPitchYaw(0.0, np.pi / 2, -np.pi / 2))

# Flag for reporting IK stats.
PRINT_IK_STATS = False


class ContactSamplerBoxIK:
    """Methods for sampling joint configurations for iiwa that contacts
    desired side faces of the box along the surface normal using a nonlinear
    optimization-based inverse kinematics solver."""

    def __init__(self):
        # Create a world w/o a floor.
        q_parser = QuasistaticParser(q_model_path_no_ground)
        self.q_sim = q_parser.make_simulator_cpp()
        self.plant = self.q_sim.get_plant()

        # Get the model and joint indices of the robot and the box.
        self.idx_a = self.plant.GetModelInstanceByName(robot_name)
        self.idx_u = self.plant.GetModelInstanceByName(object_name)
        self.q_a_idx = self.q_sim.get_q_a_indices_into_q()
        self.q_u_idx = self.q_sim.get_q_u_indices_into_q()

        # Get world, end-effector, and box frames.
        self.frame_W = self.plant.world_frame()
        self.frame_E = self.plant.GetFrameByName("iiwa_link_7")
        self.frame_B = self.plant.GetFrameByName("box")

        # Create a SNOPT solver for the inverse kinematics problem.
        self.solver = SnoptSolver()

    def sample_along_box_edge(self, edge_id: int, q_box: np.ndarray):
        """Sample a point along the given edge."""
        # Uniformly sample a pose along the edge.
        sample_p = (
            edges[edge_id][0]
            + (edges[edge_id][1] - edges[edge_id][0]) * np.random.rand()
        )
        # Get the nominal surface normal direction.
        edge_normal_rpy = edge_normals[edge_id]
        # Normalize the quaternion in the box pose.
        # Represent the orientation of the box as a rotation matrix.
        box_r = RollPitchYaw(0, 0, q_box[2]).ToRotationMatrix().matrix()
        # Apply the box's transformation to the sampled pose.
        sample_p = box_r @ sample_p + np.array([q_box[0], q_box[1], 0])
        # Rotate the surface normal per box's orientation.
        sample_r = box_r @ edge_normal_rpy.ToRotationMatrix().matrix()
        # Return the resulting target end-effector pose.
        return RigidTransform(RotationMatrix(sample_r), sample_p)

    def solve_ik(self, X_WE: RigidTransform, q_box: np.ndarray):
        """Solve the IK problem given the target end-effector pose, X_WE,
        and the object pose, q_box, to be provided as an initial guess."""
        # Create an inverse kinematics solver.
        ik = InverseKinematics(self.plant)

        # Add pose constraints based on the target.
        ik.AddPositionConstraint(
            frameB=self.frame_E,
            p_BQ=np.array([0.0, 0.0, 5e-2]),
            frameA=self.frame_W,
            p_AQ_lower=X_WE.translation() - 1e-3,
            p_AQ_upper=X_WE.translation() + 1e-3,
        )
        ik.AddOrientationConstraint(
            frameAbar=self.frame_W,
            R_AbarA=X_WE.rotation(),
            frameBbar=self.frame_E,
            R_BbarB=RotationMatrix(),
            theta_bound=1e-1,
        )

        # Set the initial guess for the arm.
        ik.prog().SetInitialGuess(ik.q()[self.q_a_idx], q_arm0)

        # Set the initial guess for the box pose.
        ik.prog().SetInitialGuess(ik.q()[self.q_u_idx], q_box)

        # Solve the program.
        result = self.solver.Solve(ik.prog())

        # Return the status and the solution.
        q_sol = result.GetSolution()
        success = False
        if result.get_solution_result() == SolutionResult.kSolutionFound:
            success = True

        return success, q_sol

    def has_collisions(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        """Check whether the given configuration has any collisions."""
        # This also updates query_object.
        self.q_sim.update_mbp_positions(q_dict)
        return self.q_sim.get_query_object().HasCollisions()

    def sample_contact(self, q: np.ndarray):
        """Sample a contact configuration for a randomly-selected edge
        using the IK solver. If a feasible configuration cannot be found
        within 10 attempts, the edge is probably unreachable, so sample
        a new edge."""
        # Get the box pose from the configuration.
        q_box = q[self.q_u_idx]
        # Specify maximum number of attempts per edge.
        attempt = 0
        max_attempt_per_edge = 10
        # Keep sampling until a configuration in contact is reached.
        while True:
            # Count the number of attempts.
            if attempt == 0 or attempt > max_attempt_per_edge:
                # Select a random edge to be sampled.
                edge_id = random.choice(EDGES_FOR_SAMPLING)
                if PRINT_IK_STATS:
                    print(f"\tSampling edge {edge_id}")
                # Reset the attempt counter in case it exceeded the
                # maximum number of attempts.
                attempt = 0
            elif PRINT_IK_STATS:
                print(f"Retrying #{attempt} for edge {edge_id}")
            attempt = attempt + 1

            # Sample a point along the edges.
            X_WE = self.sample_along_box_edge(edge_id=edge_id, q_box=q_box)

            # Solve the IK problem.
            t_ik_start = time.time()
            success, q_sol = self.solve_ik(X_WE=X_WE, q_box=q_box)
            if PRINT_IK_STATS:
                print(
                    f"\tIK solver {'succeeded' if success else 'failed'}"
                    f" and took {time.time() - t_ik_start} s"
                )

            # Put the solution into a qsim dictionary.
            q_dict = {self.idx_a: q_sol[self.q_a_idx], self.idx_u: q_box}

            # Return the solution if the IK succeeded and in contact.
            if success and self.has_collisions(q_dict):
                break

        return self.q_sim.get_q_vec_from_dict(q_dict)


# Test the contact sampler.
if __name__ == "__main__":
    contact_sampler = ContactSamplerBoxIK()

    # %%
    # Create a visualizer for the plant.
    q_parser = QuasistaticParser(q_model_path_no_ground)
    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
    q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
    plant = q_sim.get_plant()

    idx_a = plant.GetModelInstanceByName(robot_name)
    idx_u = plant.GetModelInstanceByName(object_name)

    # Initial box pose.
    q_box0 = np.hstack([0.7, 0, 0])
    # Stack the arm and box configurations.
    q0 = q_sim.get_q_vec_from_dict({idx_a: q_arm0, idx_u: q_box0})
    # Plot the initial configuration
    q_vis.draw_configuration(q0)
    # input("Ready?")

    # %%
    # Sample and visualize contact configurations.
    i = 0
    while True:
        if i % 5 == 0:
            q0[-3] += 0.01
        i = i + 1

        # Sample a new configuration.
        q_sample = contact_sampler.sample_contact(q=q0)

        q_vis.draw_configuration(q_sample)
        input("Sample?")
        # time.sleep(0.1)
