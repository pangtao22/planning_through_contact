import os
from typing import Dict, Set, List

from matplotlib import cm
import numpy as np
from pydrake.all import (ModelInstanceIndex, MultibodyPlant)

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from pydrake.all import PiecewisePolynomial, ContactResults, BodyIndex


class QuasistaticVisualizer:
    def __init__(self,
                 q_sim: QuasistaticSimulatorCpp,
                 q_sim_py: QuasistaticSimulator):
        self.q_sim_py = q_sim_py
        self.q_sim = q_sim
        self.plant = self.q_sim.get_plant()
        self.meshcat_vis = self.q_sim_py.viz.vis

        self.body_id_meshcat_name_map = self.get_body_id_to_meshcat_name_map()

        # make sure that q_sim_py and q_sim have the same underlying plant.
        self.check_plants(
            plant_a=self.q_sim.get_plant(),
            plant_b=self.q_sim_py.get_plant(),
            models_all_a=self.q_sim.get_all_models(),
            models_all_b=self.q_sim_py.get_all_models(),
            velocity_indices_a=self.q_sim.get_velocity_indices(),
            velocity_indices_b=self.q_sim.get_velocity_indices())

    def get_body_id_to_meshcat_name_map(self):
        body_id_meshcat_name_map = {}
        prefix = "drake/plant"
        for model in self.q_sim.get_actuated_models():
            body_indices = self.plant.GetBodyIndices(model)
            model_name = self.plant.GetModelInstanceName(model)
            for bi in body_indices:
                body_name = self.plant.get_body(bi).name()
                name = prefix + f'/{model_name}/{body_name}'
                body_id_meshcat_name_map[bi] = name

        return body_id_meshcat_name_map

    @staticmethod
    def check_plants(plant_a: MultibodyPlant, plant_b: MultibodyPlant,
                     models_all_a: Set[ModelInstanceIndex],
                     models_all_b: Set[ModelInstanceIndex],
                     velocity_indices_a: Dict[ModelInstanceIndex, np.ndarray],
                     velocity_indices_b: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Make sure that plant_a and plant_b are identical.
        """
        assert models_all_a == models_all_b
        for model in models_all_a:
            name_a = plant_a.GetModelInstanceName(model)
            name_b = plant_b.GetModelInstanceName(model)
            assert name_a == name_b

            idx_a = velocity_indices_a[model]
            idx_b = velocity_indices_b[model]
            assert idx_a == idx_b

    def publish_trajectory(self, x_knots: np.ndarray, h: float):
        q_dict_knots = [self.q_sim.get_q_dict_from_vec(x) for x in x_knots]
        self.q_sim_py.animate_system_trajectory(h, q_dict_traj=q_dict_knots)

    def normalize_quaternions_in_x(self, x: np.ndarray):
        for model in self.q_sim_py.models_unactuated:
            if not self.q_sim_py.is_3d_floating[model]:
                continue
            indices = self.q_sim_py.position_indices[model]
            q = x[indices[:4]]
            x[indices[:4]] /= np.linalg.norm(q)

    def calc_body_id_to_contact_force_map(self,
                                          contact_results: ContactResults):
        """
        Returns {BodyIndex: (3,) np array}, where the array is a contact force.
        Only Bodies present in both contact_results and 
         body_id_meshcat_name_map().keys() appear in the returned dictionary.
        """
        contact_forces_map = {}

        def update_contact_force_for_body(body_id: BodyIndex, f_W: np.ndarray):
            if body_id in contact_forces_map.keys():
                contact_forces_map[body_id] += f_W
            else:
                contact_forces_map[body_id] = f_W

        for i in range(contact_results.num_point_pair_contacts()):
            ci = contact_results.point_pair_contact_info(i)
            # f_W points from A into B.
            f_W = ci.contact_force()
            f_W_norm = np.linalg.norm(f_W)
            if f_W_norm < 1e-3:
                continue

            b_id_A = ci.bodyA_index()
            b_id_B = ci.bodyB_index()
            is_A_in = b_id_A in self.body_id_meshcat_name_map.keys()
            is_B_in = b_id_B in self.body_id_meshcat_name_map.keys()

            # print(b_id_A, b_id_B, f_W_norm)

            if is_A_in:
                update_contact_force_for_body(b_id_A, -f_W)

            if is_B_in:
                update_contact_force_for_body(b_id_B, f_W)

        # print("---------------------------------------------------------")

        return contact_forces_map

    def calc_contact_forces_knots_map(
            self, contact_results_list: List[ContactResults]):
        """
        Returns {BodyIndex: (n, 3) array}, where n = len(contact_results_list).
        """
        contact_forces_knots_map = {
            body_idx: []
            for body_idx in self.body_id_meshcat_name_map.keys()}

        for t, contact_results in enumerate(contact_results_list):
            contact_forces_map = self.calc_body_id_to_contact_force_map(
                contact_results)
            for body_idx in self.body_id_meshcat_name_map.keys():
                if body_idx in contact_forces_map.keys():
                    contact_forces_knots_map[body_idx].append(
                        contact_forces_map[body_idx])
                else:
                    contact_forces_knots_map[body_idx].append(np.zeros(3))

        for key, val in contact_forces_knots_map.items():
            contact_forces_knots_map[key] = np.array(val)

        return contact_forces_knots_map

    def calc_contact_forces_traj_map(
            self,
            contact_forces_knots_map: Dict[BodyIndex, np.ndarray],
            t_knots: np.ndarray):
        """
        Returns {BodyIndex: PiecewisePolynomial}.
        """
        contact_forces_traj_map = {}
        for key, f_W_knots in contact_forces_knots_map.items():
            contact_forces_traj_map[key] = PiecewisePolynomial.FirstOrderHold(
                t_knots, f_W_knots.T)
        return contact_forces_traj_map

    def calc_contact_force_norm_upper_bound(self,
        cf_knots_map: Dict[BodyIndex, np.ndarray], percentile: float):
        """
        @param cf_knots_map: {BodyIndex: (n, 3) array}, where n is the
         number of knots in the trajectory.
        Forces greater than the bound are capped at the bound when
         visualizing contact force magnitudes using a color map.
        """
        f_W_knot_norms = []
        for cf_knots in cf_knots_map.values():
            f_W_knot_norms.extend(np.linalg.norm(cf_knots, axis=1))

        return np.percentile(f_W_knot_norms, percentile)

    def render_trajectory(self, x_traj_knots: np.ndarray, h: float,
                          folder_path: str, fps: int = 60,
                          contact_result_list: List[ContactResults] = None):
        """
        Saves rendered frames to folder_path.
        """
        n_knots = len(x_traj_knots)
        t_knots = np.arange(n_knots) * h
        x_traj = PiecewisePolynomial.FirstOrderHold(t_knots, x_traj_knots.T)

        cf_traj_map = None
        if contact_result_list:
            assert len(contact_result_list) == len(x_traj_knots)
            cf_knots_map = self.calc_contact_forces_knots_map(
                contact_result_list)
            cf_traj_map = self.calc_contact_forces_traj_map(
                cf_knots_map, t_knots)
            cf_upper_bound = self.calc_contact_force_norm_upper_bound(
                cf_knots_map, 95)

        dt = 1 / fps
        n_frames = int(t_knots[-1] / dt)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for i in range(n_frames):
            t = dt * i
            x = x_traj.value(t).squeeze()
            self.normalize_quaternions_in_x(x)
            self.q_sim_py.update_mbp_positions_from_vector(x)
            self.q_sim_py.draw_current_configuration(False)

            if cf_traj_map:
                cf_map = {key: cf_traj.value(t).squeeze()
                          for key, cf_traj in cf_traj_map.items()}
                for body_id, f_W in cf_map.items():
                    f_W_norm = np.linalg.norm(f_W)
                    color = cm.plasma(f_W_norm / cf_upper_bound)
                    meshcat_name = self.body_id_meshcat_name_map[body_id]
                    self.meshcat_vis[meshcat_name].set_property(
                        "color", color)

            im = self.meshcat_vis.get_image()
            im.save(os.path.join(folder_path, f"{i:04d}.png"), 'png')

