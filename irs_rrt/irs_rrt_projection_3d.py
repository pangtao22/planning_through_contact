import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import pickle

from irs_rrt.rrt_base import Node, Edge, Rrt, RrtParams
from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode, IrsEdge
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.irs_rrt_3d import IrsRrt3D
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from pydrake.all import RollPitchYaw, Quaternion, RotationMatrix


class IrsRrtProjection3D(IrsRrtProjection):
    def __init__(self, params: IrsRrtParams, contact_sampler):
        super().__init__(params, contact_sampler)
        self.irs_rrt_3d = IrsRrt3D(params)

    def sample_subgoal(self):
        # Sample translation
        subgoal = np.random.rand(self.q_dynamics.dim_x)
        subgoal = self.x_lb + (self.x_ub - self.x_lb) * subgoal

        rpy = RollPitchYaw(subgoal[self.irs_rrt_3d.quat_ind][0:3])
        subgoal[self.irs_rrt_3d.quat_ind] = Quaternion(
            RotationMatrix(rpy).matrix()).wxyz()

        return subgoal
