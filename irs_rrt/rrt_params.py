from dataclasses import dataclass
from typing import Dict

import numpy as np
from pydrake.all import ModelInstanceIndex

from irs_mpc2.irs_mpc_params import SmoothingMode


@dataclass
class Node:
    """
    Base node class. Owns all the attributes and methods related to a
    single node. Add to nx.Digraph() using G.add_node(1, node=Node())
    """

    q: np.ndarray = None
    value: float = np.nan
    id: int = np.nan
    # To extend the tree, a subgoal is sampled first, and then a new node
    # that is "as close as possible" to the subgoal is added to the tree.
    # This field stores the subgoal associated with the new node.
    subgoal: np.ndarray = None

    def __init__(self, q):
        self.q = q  # np.array of states.


@dataclass
class Edge:
    """
    Base edge class. Owns all the attributes and methods related to an edge.
    Add to nx.Digraph() using G.add_edge(1, 2, edge=Edge())
    """

    parent: Node = None
    child: Node = None
    cost: float = np.nan


@dataclass
class RrtParams:
    """
    Base tree parameters class. Only "yaml'-able parameters should be stored
    in the parameters.
    """

    max_size: int = 100
    goal: np.ndarray = None
    root_node: Node = None
    goal_as_subgoal_prob: float = 0.5
    termination_tolerance: float = 0.1
    # TODO: rewire is not properly supported. Bad things can happen if it
    #  is set to True. We need to either remove or support this.
    rewire: bool = False
    stepsize: bool = 0.1
    # TODO (pang): the QuasistaticSimulator does not support joint limits
    #  yet. Therefore the distance metric does not reflect being close
    #  to joint limits. As a result, clipping the commanded positions
    #  using the joint limits without a distance metric that supports
    #  joint limits will lead to inefficient exploration.
    #  enforce_robot_joint_limits should be False in order to reproduce
    #  the results in the TR-O paper. But it should be set to True for
    #  hardware demos, such as the iiwa_bimanual example.
    enforce_robot_joint_limits: bool = False


@dataclass
class IrsRrtParams(RrtParams):
    h: float = 0.1
    n_samples: int = 100
    std_u: float = 0.1

    # Options for computing bundled dynamics.
    smoothing_mode: SmoothingMode = None

    # When self.bundle_mode == BundleMode.kFirstAnalytic,
    #  this log_barrier_weight is used in
    #  ReachableSet.calc_bundled_Bc_analytic.
    log_barrier_weight_for_bundling: float = 100

    # Regularization for computing inverse of covariance matrices.
    # NOTE(terry-suh): Note that if the covariance matrix is singular,
    # then the Mahalanobis distance metric is infinity. One interpretation
    # of the regularization term is to cap the infinity distance to some
    # value that scales with inverse of regularization.
    regularization: float = 1e-6

    # Stepsize.
    # TODO(terry-suh): the selection of this parameter should be automated.
    stepsize: float = 0.3

    # When set to True, this field ensures that Gurobi and Mosek are not
    # used anywhere.
    use_free_solvers: float = False

    q_model_path: str = ""
    joint_limits: Dict[ModelInstanceIndex, np.ndarray] = None


@dataclass
class IrsRrtProjectionParams(IrsRrtParams):
    # Subgoals further away from any node in the tree than
    # distance_threshold will be rejected.
    distance_threshold: float = np.inf
    grasp_prob: float = 0.2
