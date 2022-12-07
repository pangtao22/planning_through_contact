import numpy as np
from pydrake.all import RollPitchYaw, Quaternion, AngleAxis
from control.controller_system import *

kIndices3Into7 = [0, 3, 5]
kQIiwa0 = np.array([0, np.pi / 2, np.pi / 2, 0, 0, 0, np.pi / 4 * 3])
kObjZ = 0.25  # z-height of the cylinder.

iiwa_l_name = "iiwa_left"
iiwa_r_name = "iiwa_right"
object_name = "box"


class IiwaBimanualPlanarControllerSystem(ControllerSystem):
    def __init__(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        q_sim_2d: QuasistaticSimulatorCpp,
        q_sim_3d: QuasistaticSimulatorCpp,
        controller_params: ControllerParams,
        closed_loop: bool,
    ):
        super().__init__(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            q_sim_mbp=q_sim_3d,
            q_sim_q_control=q_sim_2d,
            controller_params=controller_params,
            closed_loop=closed_loop,
        )
        self.q_sim_2d = q_sim_2d
        plant_2d = q_sim_2d.get_plant()
        plant_3d = q_sim_3d.get_plant()
        self.idx_a_l_2d = plant_2d.GetModelInstanceByName(iiwa_l_name)
        self.idx_a_r_2d = plant_2d.GetModelInstanceByName(iiwa_r_name)
        self.idx_u_2d = plant_2d.GetModelInstanceByName(object_name)
        self.indices_map_2d = self.q_sim_2d.get_position_indices()

        self.idx_a_l_3d = plant_3d.GetModelInstanceByName(iiwa_l_name)
        self.idx_a_r_3d = plant_3d.GetModelInstanceByName(iiwa_r_name)
        self.idx_u_3d = plant_3d.GetModelInstanceByName(object_name)
        self.indices_map_3d = self.q_sim.get_position_indices()

    def calc_q_2d_from_q_3d(self, q_3d: np.ndarray):
        q_u_3d = q_3d[self.indices_map_3d[self.idx_u_3d]]
        # B: body frame of the un-actuated object.
        Q_WB = Quaternion(q_u_3d[:4] / np.linalg.norm(q_u_3d[:4]))
        p_WBo = q_u_3d[4:]
        q_u_2d = np.zeros(3)
        q_u_2d[:2] = p_WBo[:2]
        yaw_angle = RollPitchYaw(Q_WB).yaw_angle()
        # So that yaw angle \in [- 3 / 2 * M_PI, M_PI / 2].
        if yaw_angle > np.pi / 2:
            yaw_angle -= 2 * np.pi
        q_u_2d[2] = yaw_angle

        q_a_l_2d = q_3d[self.indices_map_3d[self.idx_a_l_3d]][kIndices3Into7]
        q_a_r_2d = q_3d[self.indices_map_3d[self.idx_a_r_3d]][kIndices3Into7]
        q_dict_2d = {
            self.idx_a_l_2d: q_a_l_2d,
            self.idx_a_r_2d: q_a_r_2d,
            self.idx_u_2d: q_u_2d,
        }

        return self.q_sim_2d.get_q_vec_from_dict(q_dict_2d)

    def calc_q_3d_from_q_2d(self, q_2d: np.ndarray):
        q_2d_dict = self.q_sim_2d.get_q_dict_from_vec(q_2d)
        q_a_l_3d = np.copy(kQIiwa0)
        q_a_r_3d = np.copy(kQIiwa0)
        q_a_l_3d[kIndices3Into7] = q_2d_dict[self.idx_a_l_2d]
        q_a_r_3d[kIndices3Into7] = q_2d_dict[self.idx_a_r_2d]
        q_u_2d = q_2d_dict[self.idx_u_2d]
        Q_WB = RollPitchYaw(0, 0, q_u_2d[2]).ToQuaternion()
        q_u_3d = np.zeros(7)
        q_u_3d[:4] = Q_WB.wxyz()
        q_u_3d[4:] = [q_u_2d[0], q_u_2d[1], kObjZ]

        q_3d_dict = {
            self.idx_u_3d: q_u_3d,  # this does not matter!
            self.idx_a_l_3d: q_a_l_3d,
            self.idx_a_r_3d: q_a_r_3d,
        }

        return self.q_sim.get_q_vec_from_dict(q_3d_dict)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        q_goal_2d = self.q_ref_input_port.Eval(context)
        u_goal_2d = self.u_ref_input_port.Eval(context)
        q_3d = self.q_input_port.Eval(context)
        q_2d = self.calc_q_2d_from_q_3d(q_3d)

        if self.closed_loop:
            (
                q_nominal_2d,
                u_nominal_2d,
                t_value,
                indices,
            ) = self.controller.find_closest_on_nominal_path(q_2d)

            s = self.controller.calc_arc_length(t_value, indices)
            (
                q_goal_2d_arc,
                u_goal_2d_arc,
            ) = self.controller.calc_q_and_u_from_arc_length(s + 0.05)
            print(f"s = {s}")
            if np.linalg.norm(q_goal_2d_arc - q_2d) < np.linalg.norm(
                q_goal_2d - q_2d
            ):
                q_goal_2d = q_goal_2d_arc
                u_goal_2d = u_goal_2d_arc
                print("oh no!")

            u_2d = self.controller.calc_u(
                q_nominal=q_nominal_2d,
                u_nominal=u_nominal_2d,
                q=q_2d,
                q_goal=q_goal_2d,
                u_goal=u_goal_2d,
            )
        else:
            u_2d = u_goal_2d

        q_2d[self.q_sim_2d.get_q_a_indices_into_q()] = u_2d
        q_3d_with_cmd = self.calc_q_3d_from_q_2d(q_2d)

        discrete_state.set_value(q_3d_with_cmd)
