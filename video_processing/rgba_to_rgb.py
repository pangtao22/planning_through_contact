import numpy as np
import cv2
import os
import irs_rrt


frames_folder_prefix = os.path.join(os.path.dirname(irs_rrt.__file__),
                                    '..', '..', "contact_videos")
frames_folder_rgba = os.path.join(frames_folder_prefix, "allegro_rgba_4")
frames_folder_rgb = os.path.join(frames_folder_prefix, "allegro_rgb_4")

files = [os.path.join(
    frames_folder_rgba, p) for p in os.listdir(frames_folder_rgba)]

            
for file in files:
    filename = os.path.split(file)[1]
    if filename.startswith('.'):
        continue
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    alpha_ind = image[:, :, 3] == 0
    image[alpha_ind] = [255, 255, 255, 255]
    new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    cv2.imwrite(os.path.join(frames_folder_rgb, filename), new_image)
    print(filename)