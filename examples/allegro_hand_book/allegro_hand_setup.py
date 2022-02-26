import os

from irs_mpc.irs_mpc_params import (BundleMode,
                                    ParallelizationMode)
from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'allegro_hand_book.yml')

# names.
robot_name = 'allegro_hand_right'
object_name = 'book'

# environment
h = 0.05

# gradient computation
bundle_mode = BundleMode.kFirst
parallel_mode = ParallelizationMode.kCppBundledB
decouple_AB = True

# IrsLqr
num_iters = 0
num_samples = 100
