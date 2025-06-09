import os
import cv2
import numpy as np


def load_labels_time(root_dir, project, num):
    gt_path = os.path.join(root_dir, project, "ground_truth.txt")
    with open(gt_path, 'r') as file:
        lines = file.readlines()
    labels = lines[0].strip()
    times = lines[2].strip()
    # assert len(labels) == len(times), f" labels ({len(labels)}) " \
    #                                  f" don't match the timestamps(len({times}))"
    labels_array = np.array(labels.split()[:num], dtype=float)
    timestamps = np.array(times.split()[:num], dtype=float)
    return labels_array, timestamps


def load_video(project,video_path):
    video_file = video_path     #VIDEO PATH (e.g. "*.mp4")
    # T = 20 # seconds
    frames = list()
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print()
    print(f"video in {project} FPS: {fs}, Total Frames: {total_frames}")
    print(f"video in {project} frame_width is {frame_width} \n frame_height is {frame_height}")
    #
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            print(f"Failed to read frame {frame_idx}, stopping loading frames...")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    if len(frames) / fs < 20:
        print(f"Warning: Only {len(frames) / fs:.2f} seconds of video loaded, which is less than 20 seconds!")
        raise ValueError(f"Video loaded is too short (less than 20 seconds).")
    else:
        print(f"Successfully loaded {len(frames)} frames ({len(frames) / fs:.2f} seconds) from {project}")
    return frames, fs
    # n_frames = int(fs*T)
    # for i in range(n_frames):
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"Video too short! Could only load {i}/{n_frames} frames")
    #         break
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frames.append(rgb_frame)
    # print(f"Loaded {len(frames)} frames.")
    # return frames, fs