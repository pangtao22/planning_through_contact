import numpy as np

from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_traj_opt import IrsRrtTrajOpt
from irs_rrt.rrt_base import Node

from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from .rrt_params import IrsRrtTrajOptParams
from .contact_sampler import ContactSampler

class IrsRrtTrajOpt3D(IrsRrt3D, IrsRrtTrajOpt):
    def __init__(self, rrt_params: IrsRrtTrajOptParams,
                 mpc_params: IrsMpcQuasistaticParameters,
                 contact_sampler: ContactSampler):
        IrsRrt3D.__init__(self, rrt_params)
        IrsRrtTrajOpt.__init__(self, rrt_params, mpc_params, contact_sampler)
        # super(IrsRrtTrajOpt3D, self).__init__(rrt_params, mpc_params, contact_sampler)

    def extend_towards_q(self, parent_node: Node, q: np.array):
        return IrsRrtTrajOpt.extend_towards_q(self, parent_node, q)

