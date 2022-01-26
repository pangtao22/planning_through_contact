import os

from irs_lqr.irs_lqr_params import IrsLqrGradientMode
from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'planar_hand_ball.yml')

# names.
robot_l_name = "arm_left"
robot_r_name = "arm_right"
object_name = "sphere"

# environment
h = 0.1

# gradient
gradient_mode = IrsLqrGradientMode.kFirst
decouple_AB = True

# workers
use_workers = True
num_iters = 10
num_samples = 50
