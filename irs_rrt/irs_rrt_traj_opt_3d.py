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
        self.quat_ind = self.irs_rrt_3d.quat_ind
    
    def sample_subgoal(self):
        # subgoal = np.random.rand(self.q_dynamics.dim_x)
        # subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal
        #
        # yaw = - np.random.rand() * np.pi
        # subgoal[self.irs_rrt_3d.quat_ind] = (
        #     RollPitchYaw([0, 0, yaw]).ToQuaternion().wxyz())
        #
        # return subgoal
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal

        rpy = RollPitchYaw(subgoal[self.quat_ind][0:3])
        subgoal[self.quat_ind] = Quaternion(
            RotationMatrix(rpy).matrix()).wxyz()
        return subgoal
        # return self.irs_rrt_3d.sample_subgoal()
