import numpy as np

from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.rrt_params import IrsRrtProjectionParams
from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.contact_sampler import ContactSampler


class IrsRrtProjection3D(IrsRrtProjection):
    def __init__(
        self,
        rrt_params: IrsRrtProjectionParams,
        contact_sampler: ContactSampler,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
    ):
        super().__init__(rrt_params, contact_sampler, q_sim, q_sim_py)
        self.irs_rrt_3d = IrsRrt3D(rrt_params, self.q_sim, q_sim_py)

    def sample_subgoal(self):
        # Sample translation
        subgoal = np.random.rand(self.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

        rpy = RollPitchYaw(subgoal[self.irs_rrt_3d.quat_ind][0:3])
        subgoal[self.irs_rrt_3d.quat_ind] = rpy.ToQuaternion().wxyz()

        return subgoal
