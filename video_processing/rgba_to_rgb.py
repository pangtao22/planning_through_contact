import os

import cv2
import tqdm

import irs_rrt


frames_folder_prefix = os.path.join(
    os.path.dirname(irs_rrt.__file__), "..", "..", "contact_videos"
)
frames_folder_rgba = os.path.join(frames_folder_prefix, "allegro_rgba_0")
frames_folder_rgb = os.path.join(frames_folder_prefix, "allegro_rgb_0")

if not os.path.exists(frames_folder_rgb):
    os.mkdir(frames_folder_rgb)

files = [
    os.path.join(frames_folder_rgba, p) for p in os.listdir(frames_folder_rgba)
]


for file in tqdm.tqdm(files):
    filename = os.path.split(file)[1]
    if filename.startswith("."):
        continue
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    alpha_ind = image[:, :, 3] == 0
    image[alpha_ind] = [255, 255, 255, 255]
    new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    cv2.imwrite(os.path.join(frames_folder_rgb, filename), new_image)
