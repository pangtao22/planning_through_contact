import os
from typing import Dict, Set

import numpy as np
from pydrake.all import (ModelInstanceIndex, MultibodyPlant)
from qsim.parser import (QuasistaticParser)
from qsim_cpp import QuasistaticSimulatorCpp

from pydrake.all import PiecewisePolynomial


class QuasistaticVisualizer:
    def __init__(self, q_parser: QuasistaticParser,
                 q_sim: QuasistaticSimulatorCpp):
        self.q_sim_py = q_parser.make_simulator_py(internal_vis=True)
        self.q_sim = q_sim

        # make sure that q_sim_py and q_sim have the same underlying plant.
        self.check_plants(
            plant_a=self.q_sim.get_plant(),
            plant_b=self.q_sim_py.get_plant(),
            models_all_a=self.q_sim.get_all_models(),
            models_all_b=self.q_sim_py.get_all_models(),
            velocity_indices_a=self.q_sim.get_velocity_indices(),
            velocity_indices_b=self.q_sim.get_velocity_indices())

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

    def publish_trajectory(self, x_traj: np.ndarray, h: float):
        q_dict_traj = [self.q_sim.get_q_dict_from_vec(x) for x in x_traj]
        self.q_sim_py.animate_system_trajectory(h, q_dict_traj=q_dict_traj)

    def normalize_quaternions_in_x(self, x: np.ndarray):
        for model in self.q_sim_py.models_unactuated:
            if not self.q_sim_py.is_3d_floating[model]:
                continue
            indices = self.q_sim_py.position_indices[model]
            q = x[indices[:4]]
            x[indices[:4]] /= np.linalg.norm(q)

    def render_trajectory(self, x_traj_knots: np.ndarray, h: float,
                          folder_path: str, fps: int = 60):
        """
        Saves rendered frames to folder_path.
        """
        n_knots = len(x_traj_knots)
        t_knots = np.arange(n_knots) * h
        x_traj = PiecewisePolynomial.FirstOrderHold(t_knots, x_traj_knots.T)

        dt = 1 / fps
        n_frames = int(t_knots[-1] / dt)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for i in range(n_frames):
            x = x_traj.value(dt * i).squeeze()
            self.normalize_quaternions_in_x(x)
            self.q_sim_py.update_mbp_positions_from_vector(x)
            self.q_sim_py.draw_current_configuration(False)
            im = self.q_sim_py.viz.vis.get_image()
            im.save(os.path.join(folder_path, f"{i:04d}.png"), 'png')

