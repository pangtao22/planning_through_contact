import os
import numpy as np

from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode

from control.controller_system import ControllerParams


q_model_path = os.path.join(models_dir, "q_sys", "allegro_hand_and_sphere.yml")
q_model_path_hardware = os.path.join(
    models_dir, "q_sys", "allegro_hand_and_sphere_hardware.yml"
)
# names.
robot_name = "allegro_hand_right"
object_name = "sphere"

# data collection.
data_folder = "ptc_data/allegro_hand"

# Stabilization.
controller_params = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu=1e2 * np.diag([10, 10, 10, 10, 1, 1, 1.0]),
    R=np.diag(0.1 * np.ones(16)),
)
