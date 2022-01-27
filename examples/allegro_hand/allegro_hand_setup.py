import os

from irs_mpc.irs_mpc_params import (BundleMode,
                                    ParallelizationMode)
from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'allegro_hand_and_sphere.yml')

# names.
robot_name = 'allegro_hand_right'
object_name = 'sphere'

# environment
h = 0.1

# gradient computation
gradient_mode = BundleMode.kFirst
parallel_mode = ParallelizationMode.kZmq
decouple_AB = True

# IrsLqr
num_iters = 10
num_samples = 100
