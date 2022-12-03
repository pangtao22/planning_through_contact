import os

from qsim.model_paths import models_dir


q_model_path = os.path.join(models_dir, "q_sys", "iiwa_box.yml")
q_model_path_no_ground = os.path.join(
    models_dir, "q_sys", "iiwa_box_no_ground.yml"
)

robot_name = "iiwa"
object_name = "box"
