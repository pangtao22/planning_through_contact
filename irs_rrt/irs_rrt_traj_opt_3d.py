import numpy as np

from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_traj_opt import IrsRrtTrajOpt
from irs_rrt.rrt_base import Node

from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from .rrt_params import IrsRrtTrajOptParams
from .contact_sampler import ContactSampler


class IrsRrtTrajOpt3D(IrsRrtTrajOpt):
    def __init__(self, rrt_params: IrsRrtTrajOptParams,
                 mpc_params: IrsMpcQuasistaticParameters,
                 contact_sampler: ContactSampler):
        super().__init__(rrt_params, mpc_params, contact_sampler)
        self.irs_rrt_3d = IrsRrt3D(rrt_params)
    
    def sample_subgoal(self):
        return self.irs_rrt_3d.sample_subgoal()


