import os

from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, "q_sys", "planar_hand_ball.yml")

# names.
robot_l_name = "arm_left"
robot_r_name = "arm_right"
object_name = "sphere"

# environment
h = 0.1

# data collection.
data_folder = "ptc_data/planar_hand"
