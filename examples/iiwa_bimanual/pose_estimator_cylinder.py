import numpy as np

from pydrake.all import (RotationMatrix, RigidTransform, Quaternion)
from optitrack import optitrack_frame_t

from control.systems_utils import wait_for_msg
from control.low_pass_filter_SE3 import LowPassFilterSe3

from pose_estimator_box import (kInchM, is_optitrack_message_good,
                                PoseEstimatorBase, get_marker_set_points)

kObjName = "cylinder"


class CylinderPoseEstimator(PoseEstimatorBase):
    def __init__(self, initial_msg: optitrack_frame_t):
        super().__init__(initial_msg)
        self.obj_id = self.get_index_from_optitrack_msg(kObjName, initial_msg)
        self.X_B0B = self.calc_X_B0B(initial_msg)
        self.lpf_SE3 = LowPassFilterSe3(h=0.005, w_cutoff=4 * 2 * np.pi)

    def calc_X_B0B(self, initial_msg: optitrack_frame_t):
        X_LB0 = self.get_X_LF_from_msg(initial_msg, self.obj_id)

        # Two markers are lower than the three that form an equilateral
        # triangle whose center is the center of the cylinder.
        points, _ = get_marker_set_points(kObjName, initial_msg)
        z_indices = sorted([(z, i) for i, z in enumerate(points[:, 2])])
        indices = [z_indices[i][1] for i in range(2, 5)]
        p_LBo = np.mean(points[indices], axis=0)
        p_LBo[2] -= 0.25
        X_LB = RigidTransform(p_LBo)
        return X_LB0.inverse().multiply(X_LB)

    def calc_X_WB(self, optitrack_msg: optitrack_frame_t):
        X_LB0 = self.get_X_LF_from_msg(optitrack_msg, self.obj_id)
        X_WB = self.X_WL.multiply(X_LB0.multiply(self.X_B0B))
        self.lpf_SE3.update(X_WB)
        return self.lpf_SE3.get_current_state()


if __name__ == "__main__":
    initial_msg = wait_for_msg(channel_name="OPTITRACK_FRAMES",
                               lcm_type=optitrack_frame_t,
                               is_message_good=is_optitrack_message_good)
    bpe = CylinderPoseEstimator(initial_msg)
