import numpy as np
from pydrake.solvers.gurobi import GurobiSolver
import pydrake.solvers.mathematicalprogram as mp


from irs_rrt.irs_rrt import IrsRrtParams, IrsRrt, IrsNode, IrsEdge
from irs_rrt.rrt_base import Node
from tqdm import tqdm


class IrsRrtProjection(IrsRrt):
    def __init__(self, params: IrsRrtParams, contact_sampler):
        self.contact_sampler = contact_sampler
        super().__init__(params)
        self.solver = GurobiSolver()

    def select_closest_node(self, subgoal: np.array,
                            d_threshold: float = np.inf,
                            print_distance: bool = False):
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
            if print_distance:
                print("closest distance to subgoal", d_batch[selected_node.id])
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

            # 3. Extend to subgoal.
            try:
                child_node, edge = self.extend(parent_node, subgoal)
            except RuntimeError:
                continue

            # 4. Attempt to rewire a candidate child node.
            if self.params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node)

            # 5. Register the new node to the graph.
            try:
                # Drawing every new node in meshcat seems to slow down
                #  tree building by quite a bit.
                self.add_node(child_node, draw_node=self.size % 50 == 0)
            except RuntimeError as e:
                print(e)
                continue
            pbar.update(1)

            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            if self.is_close_to_goal():
                self.goal_node_idx = child_node.id
                print("FOUND A PATH TO GOAL!!!!!")
                break

        pbar.close()

    def calc_du_star_towards_q_qp(self, parent_node: Node, q: np.ndarray):
        prog = mp.MathematicalProgram()
        n_a = self.q_dynamics.dim_u
        du = prog.NewContinuousVariables(n_a)
        idx_obj = self.q_sim.get_q_u_indices_into_q()
        idx_robot = self.q_sim.get_q_a_indices_into_q()
        q_a_lb = self.q_lb[idx_robot]
        q_a_ub = self.q_ub[idx_robot]
        B_obj = parent_node.Bhat[idx_obj, :]

        # We need |A * x - b| ^2 + epsilon * |x|^2, but MathematicalProgram
        # requires every term in the quadratic cost to be PD.
        Q = B_obj.T @ B_obj + 1e-2 * np.eye(n_a)
        b = (q - parent_node.chat)[idx_obj]
        b_combined = - B_obj.T @ b
        prog.AddQuadraticCost(Q, b_combined, du)
        prog.AddBoundingBoxConstraint(-self.params.stepsize,
                                      self.params.stepsize,
                                      du)
        prog.AddBoundingBoxConstraint(q_a_lb - parent_node.ubar,
                                      q_a_ub - parent_node.ubar,
                                      du)

        result = self.solver.Solve(prog)
        if not result.is_success():
            raise RuntimeError

        du_star = result.GetSolution(du)
        return du_star

    def calc_du_star_towards_q_lstsq(self, parent_node: Node, q: np.ndarray):
        # Compute least-squares solution.
        # NOTE(terry-suh): it is important to only do this on the submatrix
        # of B that has to do with u.

        idx_obj = self.q_sim.get_q_u_indices_into_q()

        du_star = np.linalg.lstsq(
            parent_node.Bhat[idx_obj, :],
            (q - parent_node.chat)[idx_obj],
            rcond=None)[0]

        # Normalize least-squares solution.
        du_norm = np.linalg.norm(du_star)
        step_size = min(du_norm, self.params.stepsize)
        du_star = du_star / du_norm
        u_star = parent_node.ubar + step_size * du_star

        idx_robot = self.q_sim.get_q_a_indices_into_q()
        q_a_lb = self.q_lb[idx_robot]
        q_a_ub = self.q_ub[idx_robot]

        u_star = np.clip(u_star, q_a_lb, q_a_ub)
        return u_star - parent_node.ubar

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node, 
        """
        regrasp = (np.random.rand() < self.params.grasp_prob)

        if regrasp:
            x_next = self.contact_sampler.sample_contact(parent_node.q)
        else:
            du_star = self.calc_du_star_towards_q_lstsq(parent_node, q)
            u_star = parent_node.ubar + du_star
            x_next = self.q_dynamics.dynamics(parent_node.q, u_star)

        cost = 0.0

        child_node = IrsNode(x_next)
        child_node.subgoal = q

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = cost

        if regrasp:
            edge.du = np.nan
            edge.u = np.nan
        else:
            edge.du = du_star
            edge.u = u_star

        return child_node, edge
