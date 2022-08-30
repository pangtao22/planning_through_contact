import os

import numpy as np

from qsim.model_paths import models_dir
from qsim_cpp import (ForwardDynamicsMode, GradientMode)

from control.controller_system import ControllerParams


q_model_path = os.path.join(models_dir, 'q_sys', 'iiwa_bimanual_box.yml')

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
