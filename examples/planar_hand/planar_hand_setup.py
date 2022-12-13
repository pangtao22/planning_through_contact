import os
import numpy as np

from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode

from control.controller_system import ControllerParams



q_model_path = os.path.join(models_dir, "q_sys", "planar_hand_ball.yml")

# names.
robot_l_name = "arm_left"
robot_r_name = "arm_right"
object_name = "sphere"

# environment
h = 0.1

# data collection.
data_folder = "ptc_data/planar_hand"

# Stabilization.
controller_params = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu= np.diag([2, 2, 3]),
    R=np.diag(0.1 * np.ones(4)),
)
