import numpy as np

from pydrake.all import RigidTransform, Quaternion

from robotics_utilities.primitives.low_pass_filter import LowPassFilter


class LowPassFilterSe3:
    def __init__(self, h: float, w_cutoff: float):
        # position filter.
        self.p_lpf = LowPassFilter(dimension=3, h=h, w_cutoff=w_cutoff)
        # quaternion filter.
        self.q_lpf = LowPassFilter(dimension=4, h=h, w_cutoff=w_cutoff)

    def update(self, X: RigidTransform):
        p_new = X.translation()
        q_new = X.rotation().ToQuaternion().wxyz()
        self.p_lpf.update(p_new)
        self.q_lpf.update(q_new)
        self.q_lpf.x /= np.linalg.norm(self.q_lpf.x)

    def get_current_state(self):
        return RigidTransform(
            Quaternion(self.q_lpf.get_current_state()),
            self.p_lpf.get_current_state(),
        )
