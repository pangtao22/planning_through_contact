import os
import numpy as np

from irs_mpc2.irs_mpc_params import SmoothingMode
from qsim.model_paths import models_dir
from qsim_cpp import ForwardDynamicsMode, GradientMode

from pydrake.all import RollPitchYaw, RigidTransform

from control.controller_system import ControllerParams

q_model_path = os.path.join(models_dir, "q_sys", "allegro_hand_door_push.yml")

# names.
robot_name = "allegro_hand_right"
object_name = "door"

# environment
h = 0.05

# IrsLqr
num_samples = 100

# data collection.
data_folder = "ptc_data/allegro_hand_door"

# Stabilization.
controller_params = ControllerParams(
    forward_mode=ForwardDynamicsMode.kLogIcecream,
    gradient_mode=GradientMode.kBOnly,
    log_barrier_weight=5000,
    control_period=None,
    Qu=1e2 * np.diag([10, 10]),
    R=np.diag(0.1 * np.ones(19)),
)

def q_u_to_pose(q_u):
    door_angle = q_u[0]
    knob_angle = q_u[1]

    X_dk = RollPitchYaw()

