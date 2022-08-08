import numpy as np

from pydrake.all import (RotationMatrix, RigidTransform, )
from optitrack import optitrack_frame_t
from estimate_sphere_center import estimate_center_and_r, estimate_center

kAllegroPalmName = "allegro_palm"  # 3 markers on the palm.
kAllegroBackName = "allegro_back"  # 3 markers on the back.
kBallName = "ball"
kMarkerRadius = 0.00635


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


def get_marker_set_points(marker_set_name: str,
                          msg: optitrack_frame_t):
    """
    Returns None if msg does not have a marker_set with name marker_set_name.
    """
    for marker_set in msg.marker_sets:
        if marker_set.name == marker_set_name:
            return np.array(marker_set.xyz)


class OptitrackPoseEstimator:
    def __init__(self, optitrack_msg: optitrack_frame_t):
        self.X_WL, self.p_palm_W = self.calc_X_LW(optitrack_msg)
        self.delta_surface_to_center = \
            self.calc_ball_markers_relative_to_center(optitrack_msg)

    @staticmethod
    def calc_X_LW(optitrack_msg: optitrack_frame_t):
        """
        W is the world frame of the hand-ball MBP.
        *----------*--> y_W
          |
          |
          *
          |
          V
         x_W
        The three palm markers are indicated by stars.
        The axes of the Lab frame (L) should be almost aligned with those of W.
        """
        p_palm_L = get_marker_set_points(kAllegroPalmName, optitrack_msg)

        idx_palm_back = np.argmax(p_palm_L[:, 0])
        idx_palm_front = [0, 1, 2]
        idx_palm_front.remove(idx_palm_back)
        # z-axis
        nz_LW = np.cross(p_palm_L[1] - p_palm_L[0], p_palm_L[2] - p_palm_L[0])
        nz_LW /= np.linalg.norm(nz_LW)

        # y-axis
        ny_LW = p_palm_L[idx_palm_front[0]] - p_palm_L[idx_palm_front[1]]
        ny_LW /= np.linalg.norm(ny_LW)
        if ny_LW[1] < 0:
            ny_LW *= -1
        # x-axis
        nx_LW = np.cross(ny_LW, nz_LW)

        R_LW = RotationMatrix(np.vstack([nx_LW, ny_LW, nz_LW]).T)

        # origin.
        y0_LW = (p_palm_L[idx_palm_front[0], 1]
                 + p_palm_L[idx_palm_front[1], 1]) / 2
        # the center of the markers are 7 / 16 inches above the surface.
        z0_LW = np.mean(p_palm_L[:, 2]) - 0.0254 / 16 * 7 - 0.0098

        x0_LW = np.mean(p_palm_L[idx_palm_front, 0]) + 0.0859

        X_LW = RigidTransform(R_LW, [x0_LW, y0_LW, z0_LW])
        X_WL = X_LW.inverse()
        p_palm_W = X_WL.multiply(p_palm_L.T).T

        return X_WL, p_palm_W

    @staticmethod
    def calc_ball_markers_relative_to_center(
            optitrack_msg: optitrack_frame_t):
        """
        Assuming that there are three markers on the x, y, and z axis of the
        ball. Check my phone for the correct starting pose for this function
        to work.
        """
        p_ball_surface_L = get_marker_set_points(kBallName, optitrack_msg)

        idx_x = np.argmin(p_ball_surface_L[:, 0])
        idx_y = np.argmin(p_ball_surface_L[:, 1])
        idx_z = np.argmax(p_ball_surface_L[:, 2])

        p_x_L = p_ball_surface_L[idx_x]
        p_y_L = p_ball_surface_L[idx_y]
        p_z_L = p_ball_surface_L[idx_z]
        p_c_L = (p_x_L + p_y_L + p_z_L) / 3

        d_xy = np.linalg.norm(p_x_L - p_y_L)
        d_yz = np.linalg.norm(p_y_L - p_z_L)
        d_zx = np.linalg.norm(p_z_L - p_x_L)
        r_ball = (d_xy + d_yz + d_zx) / 3 / np.sqrt(2)
        v_xz = p_z_L - p_x_L
        v_zy = p_y_L - p_z_L
        n = np.cross(v_xz, v_zy)
        n /= np.linalg.norm(n)
        p_ball_center_L = p_c_L + n * r_ball / np.sqrt(3)

        return p_ball_surface_L - p_ball_center_L

    def calc_ball_center_W(self, p_ball_surface_L: np.ndarray):
        p_ball_center_L = np.mean(
            p_ball_surface_L - self.delta_surface_to_center, axis=0)
        return self.X_WL.multiply(p_ball_center_L)


