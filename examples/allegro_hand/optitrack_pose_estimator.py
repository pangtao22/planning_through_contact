import numpy as np

from pydrake.all import RotationMatrix, RigidTransform, Quaternion
from optitrack import optitrack_frame_t
from robotics_utilities.primitives.low_pass_filter import LowPassFilter

kAllegroPalmName = "allegro_palm"  # 3 markers on the palm.
kBallName = "ball"
kMarkerRadius = 0.00635
kMarkerCenterToPalm = 0.0254 / 16 * 7


def is_optitrack_message_good(msg: optitrack_frame_t):
    if msg.num_marker_sets == 0:
        return False

    non_empty_set_names = [kAllegroPalmName, kBallName]
    is_good = {name: False for name in non_empty_set_names}
    for marker_set in msg.marker_sets:
        if marker_set.name in non_empty_set_names:
            for p in marker_set.xyz:
                if np.linalg.norm(p) < 1e-3:
                    return False
            is_good[marker_set.name] = True

    is_good_all = True
    for good in is_good.values():
        is_good_all = is_good_all and good

    return is_good_all


def get_marker_set_points(marker_set_name: str, msg: optitrack_frame_t):
    """
    Returns None if msg does not have a marker_set with name marker_set_name.
    Otherwise, returns ((n, 3) marker coordinates, index of marker set.)
    """
    for i, marker_set in enumerate(msg.marker_sets):
        if marker_set.name == marker_set_name:
            return np.array(marker_set.xyz), i


class OptitrackPoseEstimator:
    def __init__(self, optitrack_msg: optitrack_frame_t):
        self.X_WL, self.p_palm_W = self.calc_X_WL(optitrack_msg)
        self.X_WP = self.calc_X_WP(optitrack_msg, self.X_WL)

        # B0: frame of the ball defined in optitrack.
        # B: frame of the ball at the ball's center.
        self.X_B0B = self.calc_X_B0B(optitrack_msg)

        # TODO: maybe not hard-code the filtering frequencies?
        h = 0.02
        w_cutoff = 2 * 2 * np.pi
        self.p_WBo_lpf = LowPassFilter(dimension=3, h=h, w_cutoff=w_cutoff)
        self.q_WB_lpf = LowPassFilter(dimension=4, h=h, w_cutoff=w_cutoff)
        self.update_X_WB(optitrack_msg)

    @staticmethod
    def get_X_LB_from_msg(
        optitrack_msg: optitrack_frame_t, idx_rigid_body: int
    ):
        q = optitrack_msg.rigid_bodies[idx_rigid_body].quat
        return RigidTransform(
            Quaternion(q[3], q[0], q[1], q[2]),
            optitrack_msg.rigid_bodies[idx_rigid_body].xyz,
        )

    @staticmethod
    def calc_X_WP(optitrack_msg: optitrack_frame_t, X_WL: RigidTransform):
        _, idx_p = get_marker_set_points(kAllegroPalmName, optitrack_msg)
        X_LP = OptitrackPoseEstimator.get_X_LB_from_msg(optitrack_msg, idx_p)
        return X_WL.multiply(X_LP)

    @staticmethod
    def get_palm_marker_indices(p_palm_L: np.ndarray):
        """
        p_palm_L (3, 3) array, where p_palm_L[i] is the xyz coordinates of
         palm marker i in L frame.
        W is the world frame of the hand-ball MBP.
        A----------B--> y_W
          |
          |
          C
          |
          V
         x_W
        The three palm markers are indicated by (A, B, C).
        """
        idx_C = np.argmax(p_palm_L[:, 0])
        idx_B = np.argmax(p_palm_L[:, 1])
        idx_all = [0, 1, 2]
        idx_all.remove(idx_B)
        idx_all.remove(idx_C)
        idx_A = idx_all[0]
        return idx_A, idx_B, idx_C

    @staticmethod
    def calc_z_LW(p_palm_L: np.ndarray):
        nz_LW = np.cross(p_palm_L[1] - p_palm_L[0], p_palm_L[2] - p_palm_L[0])
        nz_LW /= np.linalg.norm(nz_LW)
        if nz_LW[2] < 0:
            nz_LW *= -1

        return nz_LW

    @staticmethod
    def calc_X_B0B(optitrack_msg: optitrack_frame_t):
        p_palm_L, _ = get_marker_set_points(kAllegroPalmName, optitrack_msg)
        _, idx_rigid_body_ball = get_marker_set_points(kBallName, optitrack_msg)
        idx_A, idx_B, idx_C = OptitrackPoseEstimator.get_palm_marker_indices(
            p_palm_L
        )
        X_LB0 = OptitrackPoseEstimator.get_X_LB_from_msg(
            optitrack_msg, idx_rigid_body_ball
        )

        R = 0.061  # sphere radius
        y = np.linalg.norm(p_palm_L[idx_C] - p_palm_L[idx_A])
        c = np.sqrt((R + kMarkerRadius) ** 2 - (R - kMarkerCenterToPalm) ** 2)
        e = np.sqrt(c**2 - (y / 2) ** 2)

        n_AC = p_palm_L[idx_C] - p_palm_L[idx_A]
        n_AC /= np.linalg.norm(n_AC)
        nz_LW = OptitrackPoseEstimator.calc_z_LW(p_palm_L)

        n_e = np.cross(nz_LW, n_AC)
        # Sc: sphere center.
        p_Sc_L = (p_palm_L[idx_A] + p_palm_L[idx_C]) / 2 + n_e * e
        p_Sc_L += nz_LW * (R - kMarkerCenterToPalm)

        X_LB = RigidTransform(p_Sc_L)

        return X_LB0.inverse().multiply(X_LB)

    @staticmethod
    def calc_X_WL(optitrack_msg: optitrack_frame_t):
        """
        The axes of the Lab frame (L) should be almost aligned with those of W.
        """

        p_palm_L, _ = get_marker_set_points(kAllegroPalmName, optitrack_msg)
        idx_A, idx_B, idx_C = OptitrackPoseEstimator.get_palm_marker_indices(
            p_palm_L
        )

        # z-axis
        nz_LW = OptitrackPoseEstimator.calc_z_LW(p_palm_L)

        # y-axis
        ny_LW = p_palm_L[idx_B] - p_palm_L[idx_A]
        ny_LW /= np.linalg.norm(ny_LW)
        if ny_LW[1] < 0:
            ny_LW *= -1
        # x-axis
        nx_LW = np.cross(ny_LW, nz_LW)

        R_LW = RotationMatrix(np.vstack([nx_LW, ny_LW, nz_LW]).T)

        # origin.
        y0_LW = (p_palm_L[idx_B, 1] + p_palm_L[idx_A, 1]) / 2
        # the center of the markers are 7 / 16 inches above the surface.
        z0_LW = np.mean(p_palm_L[:, 2]) - kMarkerCenterToPalm - 0.0098

        x0_LW = np.mean(p_palm_L[[idx_A, idx_B], 0]) + 0.0859

        X_LW = RigidTransform(R_LW, [x0_LW, y0_LW, z0_LW])
        X_WL = X_LW.inverse()
        p_palm_W = X_WL.multiply(p_palm_L.T).T

        return X_WL, p_palm_W

    def update_X_WB(self, optitrack_msg: optitrack_frame_t):
        _, idx = get_marker_set_points(kBallName, optitrack_msg)
        X_LB0 = self.get_X_LB_from_msg(optitrack_msg, idx)
        X_WB0 = self.X_WL.multiply(X_LB0)
        X_WB = X_WB0.multiply(self.X_B0B)

        p_WBo = X_WB.translation()
        q_WB = X_WB.rotation().ToQuaternion().wxyz()

        self.p_WBo_lpf.update(p_WBo)
        self.q_WB_lpf.update(q_WB)
        self.q_WB_lpf.x /= np.linalg.norm(self.q_WB_lpf.x)

    def get_X_WB(self):
        """
        Get filtered X_WB.
        """
        return RigidTransform(
            Quaternion(self.q_WB_lpf.get_current_state()),
            self.p_WBo_lpf.get_current_state(),
        )

    def get_p_ball_surface_W_and_X_WB(self, optitrack_msg: optitrack_frame_t):
        """
        returns (p_ball_surface_W (4, 3), X_WB)
        """
        p_ball_surface_L, idx = get_marker_set_points(kBallName, optitrack_msg)
        p_ball_surface_W = self.X_WL.multiply(p_ball_surface_L.T).T

        return p_ball_surface_W, self.get_X_WB()

    def reset_ball_orientation(self, optitrack_msg: optitrack_frame_t):
        p_ball_surface_L, idx = get_marker_set_points(kBallName, optitrack_msg)
        X_LB0 = self.get_X_LB_from_msg(optitrack_msg, idx)
        X_WB0 = self.X_WL.multiply(X_LB0)

        R_WB0 = X_WB0.rotation()
        R_WB = RotationMatrix()
        self.X_B0B.set_rotation(R_WB0.inverse().multiply(R_WB))
