import numpy as np

from pydrake.math import RollPitchYaw
from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_traj_opt import IrsRrtTrajOpt
from irs_rrt.rrt_base import Node

from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from .rrt_params import IrsRrtTrajOptParams
from .contact_sampler import ContactSampler
from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix


class IrsRrtTrajOpt3D(IrsRrtTrajOpt):
    def __init__(self, rrt_params: IrsRrtTrajOptParams,
                 mpc_params: IrsMpcQuasistaticParameters,
                 contact_sampler: ContactSampler):
        super().__init__(rrt_params, mpc_params, contact_sampler)
        self.irs_rrt_3d = IrsRrt3D(rrt_params)
    
    def sample_subgoal(self):
        # Sample translation
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal

        rpy = RollPitchYaw(subgoal[self.irs_rrt_3d.quat_ind][0:3])
        subgoal[self.irs_rrt_3d.quat_ind] = rpy.ToQuaternion().wxyz()

        return subgoal
