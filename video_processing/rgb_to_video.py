import numpy as np
import cv2
import os
import subprocess

frames_folder_rgb = "ptc_data/frames_rgb"
videos_folder_rgb = "ptc_data/videos_mp4"

examples = [os.path.join(
    frames_folder_rgb, p) for p in os.listdir(frames_folder_rgb)]
for example in examples:
    example_name = os.path.split(example)[1]
    video_example = os.path.join(videos_folder_rgb, example_name)
    os.mkdir(video_example)

    methods = [os.path.join(
        example, p) for p in os.listdir(example)]
    for method in methods:
        method_name = os.path.split(method)[1]
        video_method = os.path.join(video_example, method_name)
        os.mkdir(video_method)

        trials = [os.path.join(
            method, p) for p in os.listdir(method)]

        for trial in trials:
            trial_name = os.path.split(trial)[1]
            video_trial = os.path.join(video_method, trial_name)

            subprocess.call([
                "ffmpeg", 
                "-r", "30",
                "-i", os.path.join(trial, "%04d.png"),
                "-vcodec", "libx264",
                "-q:v", "1",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-crf", "18", 
                video_trial + ".mp4"])

