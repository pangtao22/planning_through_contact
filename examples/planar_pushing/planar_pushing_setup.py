import os

from qsim.model_paths import models_dir

q_model_path = os.path.join(models_dir, "q_sys", "box_pushing.yml")

# names.
robot_name = "hand"
object_name = "box"

# environment
h = 0.1

# IrsMpc
num_iters = 10
num_samples = 100

# data collection.
data_folder = "ptc_data/planar_pushing"
