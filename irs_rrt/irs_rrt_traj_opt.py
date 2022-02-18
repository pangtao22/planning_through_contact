import numpy as np
from tqdm import tqdm

from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic

from .irs_rrt import IrsRrt, IrsNode, IrsEdge
from .rrt_params import IrsRrtTrajOptParams
from .reachable_set import ReachableSet
from .contact_sampler import ContactSampler


class IrsRrtTrajOpt(IrsRrt):
    def __init__(self, rrt_params: IrsRrtTrajOptParams,
                 mpc_params: IrsMpcQuasistaticParameters,
                 contact_sampler: ContactSampler):
        super().__init__(params=rrt_params)
        # A QuasistaticDynamics object is constructed in IrsRrt.
        self.idx_q_u_indo_x = self.q_dynamics.get_q_u_indices_into_x()
        self.idx_q_a_into_x = self.q_dynamics.get_q_a_indices_into_x()

        # IrsMpc for traj-opt.
        self.irs_mpc = IrsMpcQuasistatic(q_dynamics=self.q_dynamics,
                                         params=mpc_params)
        self.mpc_params = mpc_params
        self.reachable_set = ReachableSet(
            q_dynamics=self.q_dynamics, params=rrt_params,
            q_dynamics_p=self.irs_mpc.q_dynamics_parallel)
        self.contact_sampler = contact_sampler

    def extend_towards_q(self, parent_node: IrsNode, q: np.array):
        q0 = parent_node.q
        T = self.mpc_params.T
        q_trj_d = np.tile(q, (T + 1, 1))
        u_trj_0 = np.tile(q0[self.q_dynamics.get_q_a_indices_into_x()], (T, 1))
        self.irs_mpc.initialize_problem(x0=q0, x_trj_d=q_trj_d, u_trj_0=u_trj_0)
        self.irs_mpc.iterate(
            10, cost_Qu_f_threshold=self.params.termination_tolerance)
        self.irs_mpc.plot_costs()

        child_node = IrsNode(self.irs_mpc.x_trj_best[-1])
        child_node.subgoal = q
        # teleport finger to a grasping pose.
        q = child_node.q
        q_u = q[self.idx_q_u_indo_x]
        q_grasp = self.contact_sampler.sample_contact(q_u)
        q_a = q_grasp[self.idx_q_a_into_x]
        child_node.q[self.idx_q_a_into_x] = q_a

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = self.irs_mpc.cost_best
        edge.trj = self.irs_mpc.package_solution()

        return child_node, edge

    def select_closest_node(self, subgoal: np.array,
                            d_threshold: float = np.inf):
        """
        Given a subgoal, this function finds the node that is closest from the
         subgoal.
        None is returned if the distances of all nodes are greater than
         d_treshold.
        """
        d_batch = self.calc_distance_batch(subgoal)
        i_min = np.argmin(d_batch)
        if d_batch[i_min] < d_threshold:
            selected_node = self.get_node_from_id(i_min)
        else:
            selected_node = None
        return selected_node

    def iterate(self):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        while self.size < self.params.max_size:
            # 1. Sample a subgoal.
            if self.cointoss_for_goal():
                subgoal = self.params.goal
            else:
                subgoal = self.sample_subgoal()

            # 2. Sample closest node to subgoal
            parent_node = self.select_closest_node(
                subgoal, d_threshold=self.params.distance_threshold)
            if parent_node is None:
                continue
            # update progress only if a valid parent_node is chosen.
            pbar.update(1)

            # 3. Extend to subgoal.
            child_node, edge = self.extend(parent_node, subgoal)

            # 4. Attempt to rewire a candidate child node.
            if self.params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node)

            # 5. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_node_close_to_goal(child_node):
                print("FOUND A PATH TO GOAL!!!!!")
                break

        pbar.close()

    def is_node_close_to_goal(self, node: IrsNode):
        d = self.irs_mpc.calc_Q_cost(
            models_list=self.q_dynamics.models_unactuated,
            x_dict=self.q_dynamics.get_q_dict_from_x(node.q),
            xd_dict=self.q_dynamics.get_q_dict_from_x(self.params.goal),
            Q_dict=self.mpc_params.Qd_dict)

        return d < self.params.termination_tolerance
