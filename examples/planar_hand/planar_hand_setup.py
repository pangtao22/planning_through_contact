import os

from irs_mpc.irs_mpc_params import (BundleMode,
                                    ParallelizationMode)

from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'planar_hand_ball.yml')

# names.
robot_l_name = "arm_left"
robot_r_name = "arm_right"
object_name = "sphere"

# environment
h = 0.1

# gradient computation
gradient_mode = BundleMode.kFirst
parallel_mode = ParallelizationMode.kCppBundledB
decouple_AB = True

# IrsLqr
num_iters = 10
num_samples = 100
