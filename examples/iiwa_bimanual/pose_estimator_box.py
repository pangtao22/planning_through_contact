import numpy as np

from pydrake.all import RotationMatrix, RigidTransform, Quaternion
from optitrack import optitrack_frame_t

from control.systems_utils import wait_for_msg
from control.low_pass_filter_SE3 import LowPassFilterSe3

kObjName = "box"
kBaseName = "base"
kInchM = 0.0254


def is_optitrack_message_good(msg: optitrack_frame_t):
    if msg.num_marker_sets == 0:
        return False

    non_empty_set_names = [kObjName, kBaseName]
    is_good = {name: False for name in non_empty_set_names}
    for marker_set in msg.marker_sets:
        if marker_set.name in non_empty_set_names:
            for p in marker_set.xyz:
                if np.linalg.norm(p) < 1e-3:
                    return False
            is_good[marker_set.name] = True

    return all(is_good.values())


def get_marker_set_points(marker_set_name: str, msg: optitrack_frame_t):
    """
    Returns None if msg does not have a marker_set with name marker_set_name.
    Otherwise, returns ((n, 3) marker coordinates, index of marker set.)
    """
    for i, marker_set in enumerate(msg.marker_sets):
        if marker_set.name == marker_set_name:
            return np.array(marker_set.xyz), i


class PoseEstimatorBase:
    def __init__(self, initial_msg: optitrack_frame_t):
        self.X_WL = self.calc_X_WL(initial_msg)

    @staticmethod
    def calc_X_WL(initial_msg: optitrack_frame_t):
        """
        The center of the markers in right_base is 15/16 inches above the
        table surface.
        """
        # right_base points in Lab frame.
        p_b_L, _ = get_marker_set_points(kBaseName, initial_msg)
        # R: right base frame.
        p_Wo_L = np.mean(p_b_L, axis=0)
        p_Wo_L[2] -= kInchM / 16 * 15
        X_LW = RigidTransform(p_Wo_L)
        return X_LW.inverse()

    @staticmethod
    def get_X_LF_from_msg(
        optitrack_msg: optitrack_frame_t, idx_rigid_body: int
    ):
        """
        F is the frame of the rigid body indexed by idx_rigid_body in
         optitrack_msg.
        """
        q = optitrack_msg.rigid_bodies[idx_rigid_body].quat
        return RigidTransform(
            Quaternion(q[3], q[0], q[1], q[2]),
            optitrack_msg.rigid_bodies[idx_rigid_body].xyz,
        )

    @staticmethod
    def get_q_and_p_from_msg(
        optitrack_msg: optitrack_frame_t, idx_rigid_body: int
    ):
        """
        F is the frame of the rigid body indexed by idx_rigid_body in
         optitrack_msg.
        """
        q = optitrack_msg.rigid_bodies[idx_rigid_body].quat
        q = np.array([q[3], q[0], q[1], q[2]])
        p = np.array(optitrack_msg.rigid_bodies[idx_rigid_body].xyz)
        return q, p

    @staticmethod
    def get_index_from_optitrack_msg(name: str, msg: optitrack_frame_t):
        for i, marker_set in enumerate(msg.marker_sets):
            if marker_set.name == name:
                return i

    def calc_X_WB(self, optitrack_msg: optitrack_frame_t) -> RigidTransform:
        raise NotImplementedError
        pass


class BoxPoseEstimator(PoseEstimatorBase):
    def __init__(self, initial_msg: optitrack_frame_t):
        super().__init__(initial_msg)
        self.box_idx = self.get_index_from_optitrack_msg(kObjName, initial_msg)
        self.X_B0B = self.calc_X_B0B(initial_msg)
        self.lpf_SE3 = LowPassFilterSe3(h=0.005, w_cutoff=4 * 2 * np.pi)

    def calc_X_B0B(self, initial_msg: optitrack_frame_t):
        # We marked the initial pose of the object on the table with tape.
        # This function assumes that the box is at that pose.
        # B0 is the frame defined by optitrack.
        # B is the frame used by the planner.

        X_LB0 = self.get_X_LF_from_msg(initial_msg, self.box_idx)
        X_WB0 = self.X_WL.multiply(X_LB0)
        X_WB = RigidTransform([0.55, 0, 0.315])
        return X_WB0.inverse().multiply(X_WB)

    def calc_X_WB(self, optitrack_msg: optitrack_frame_t):
        X_LB0 = self.get_X_LF_from_msg(optitrack_msg, self.box_idx)
        X_WB = self.X_WL.multiply(X_LB0.multiply(self.X_B0B))
        self.lpf_SE3.update(X_WB)
        return self.lpf_SE3.get_current_state()


if __name__ == "__main__":
    initial_msg = wait_for_msg(
        channel_name="OPTITRACK_FRAMES",
        lcm_type=optitrack_frame_t,
        is_message_good=is_optitrack_message_good,
    )
    bpe = BoxPoseEstimator(initial_msg)
