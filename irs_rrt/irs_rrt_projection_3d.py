import numpy as np
from irs_rrt.rrt_params import IrsRrtProjectionParams
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix


class IrsRrtProjection3D(IrsRrtProjection):
    def __init__(self, params: IrsRrtProjectionParams, contact_sampler):
        super().__init__(params, contact_sampler)
        self.irs_rrt_3d = IrsRrt3D(params)

    def sample_subgoal(self):
        # Sample translation
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

        rpy = RollPitchYaw(subgoal[self.irs_rrt_3d.quat_ind][0:3])
        subgoal[self.irs_rrt_3d.quat_ind] = rpy.ToQuaternion().wxyz()

        return subgoal
