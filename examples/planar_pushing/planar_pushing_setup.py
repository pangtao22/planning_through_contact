import os

from irs_mpc.irs_mpc_params import (BundleMode,
                                    ParallelizationMode)

from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, 'q_sys', 'box_pushing.yml')

# names.
robot_name = "hand"
object_name = "box"

# environment
h = 0.1

# gradient computation
bundle_mode = BundleMode.kFirstRandomized
parallel_mode = ParallelizationMode.kCppBundledB
decouple_AB = True

# IrsMpc
num_iters = 10
num_samples = 100
