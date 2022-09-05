import os

import numpy as np

from pydrake.all import MultibodyPlant
from qsim.model_paths import models_dir
from qsim_cpp import (ForwardDynamicsMode, GradientMode)

from control.controller_system import ControllerParams


q_model_path = os.path.join(models_dir, 'q_sys', 'iiwa_bimanual_box.yml')
q_model_path_planar = os.path.join(
    models_dir, 'q_sys', 'iiwa_planar_bimanual_box.yml')

iiwa_l_name = "iiwa_left"
iiwa_r_name = "iiwa_right"
object_name = "box"


controller_params = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu=np.diag([10, 10, 10, 10, 1, 1, 1.]),
    R=np.diag(np.ones(14)),
    joint_limit_padding=0.05)


def calc_z_height(plant: MultibodyPlant):
    """
    returns the height (COM coordinate in world z-axis) of the object,
    when the object is 2D, i.e. having only x, y and theta as its DOFs.
    """
    context_plant = plant.CreateDefaultContext()
    X_WB = plant.CalcRelativeTransform(
        context_plant, plant.world_frame(), plant.GetFrameByName('box'))
    return X_WB.translation()[2]
