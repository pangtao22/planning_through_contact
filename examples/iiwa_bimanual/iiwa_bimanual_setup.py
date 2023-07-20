import os

import numpy as np

from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw

from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from control.controller_system import ControllerParams

q_model_path = os.path.join(models_dir, "q_sys", "iiwa_bimanual_box.yml")
q_model_path_planar = os.path.join(
    models_dir, "q_sys", "iiwa_planar_bimanual_cylinder.yml"
)
"""
iiwa_bimanual_cylinder.yml describes the same system as 
iiwa_planar_bimanual_box.yml. The former is in 3D, whereas the latter is in 
the xy plane.
"""
q_model_path_cylinder = os.path.join(
    models_dir, "q_sys", "iiwa_bimanual_cylinder.yml"
)

iiwa_l_name = "iiwa_left"
iiwa_r_name = "iiwa_right"
object_name = "box"

controller_params_3d = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu=np.diag([10, 10, 10, 10, 1, 1, 1.0]),
    R=np.diag(np.ones(14)),
    joint_limit_padding=0.05,
)

controller_params_2d = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu=np.diag([1, 1, 1]),
    R=np.diag(1 * np.ones(6)),
    joint_limit_padding=0.05,
)


def calc_z_height(plant: MultibodyPlant):
    """
    returns the height (COM coordinate in world z-axis) of the object,
    when the object is 2D, i.e. having only x, y and theta as its DOFs.
    """
    context_plant = plant.CreateDefaultContext()
    X_WB = plant.CalcRelativeTransform(
        context_plant, plant.world_frame(), plant.GetFrameByName(object_name)
    )
    return X_WB.translation()[2]


def draw_goal_and_object_triads_2d(
    q_vis: QuasistaticVisualizer, plant: MultibodyPlant, q_u_goal: np.ndarray
):
    q_vis.draw_object_triad(length=0.4, radius=0.005, opacity=1, path="box/box")
    q_vis.draw_goal_triad(
        length=0.4,
        radius=0.01,
        opacity=0.7,
        X_WG=RigidTransform(
            RollPitchYaw(0, 0, q_u_goal[2]),
            np.hstack([q_u_goal[:2], [calc_z_height(plant)]]),
        ),
    )
