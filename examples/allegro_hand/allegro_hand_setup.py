import os

from irs_lqr.irs_lqr_params import IrsLqrGradientMode
from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'allegro_hand_and_sphere.yml')

# names.
robot_name = 'allegro_hand_right'
object_name = 'sphere'

# environment
h = 0.1

# gradient
gradient_mode = IrsLqrGradientMode.kFirst
decouple_AB = True

# workers
use_workers = True
task_stride = 1
num_iters = 10
num_samples = 100
