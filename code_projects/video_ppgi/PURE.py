import glob
import json
import os
import cv2
import numpy as np


def read_frames(folder_path):
    """Reads a PURE data file, returns frames(T, H, W, 3) """
    frames = list()
    all_png = sorted(glob.glob(os.path.join(folder_path, '*.png')))

    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)


def read_wave_with_timestamps(data_dir, project):

    bvp_file = os.path.join(data_dir, project, f"{project}.json")
    with open(bvp_file, "r") as f:
        data = json.load(f)
        labels = data["/FullPackage"]
        imgs = data["/Image"]
        # Extract nanosecond timestamp and convert to seconds
        bvp_timestamps = [entry["Timestamp"] / 1e9 for entry in labels]
        waves = [entry["Value"]["waveform"] for entry in labels]
        img_timestamps = [entry["Timestamp"] / 1e9 for entry in imgs]
    assert len(bvp_timestamps) == len(waves), f"length of bvp-timestamps({len(bvp_timestamps)})" \
                                          f" doesn't match length of bvps({len(waves)})"
    return np.asarray(waves), np.asarray(bvp_timestamps), np.asarray(img_timestamps)

# if __name__ == "__main__":
#     folder_path = r"D:\MA_DATA\pure"
#     timestamps, waves, img_timestamps = read_wave_with_timestamps(folder_path, "01-01")
#     print(img_timestamps)
#     print("*********************************************")
#     print(len(img_timestamps))
