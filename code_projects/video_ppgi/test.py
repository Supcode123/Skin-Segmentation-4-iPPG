import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

# with open(r'D:\MA_DATA\video\project1\ground_truth.txt', "r") as f:
#     lines = f.readlines()
#
# ppg_signal = np.array([float(x) for x in lines[0].strip().split()])  # 第一行是 PPG
# mean_gt = np.mean(ppg_signal)
# std_gt = np.std(ppg_signal)
#
# # 打印均值和标准差
# print(f'Mean of ground_truth: {mean_gt}')
# print(f'Standard Deviation of ground_truth: {std_gt}')
#
# # 判断是否标准化（通常均值接近0，标准差接近1）
# if np.abs(mean_gt) < 1e-5 and np.abs(std_gt - 1) < 1e-5:
#     print("ground_truth is likely standardized.")
# else:
#     print("ground_truth is not standardized.")
# timestamps = np.array([float(x) for x in lines[2].strip().split()])  # 第三行是时间戳
# from scipy.interpolate import interp1d
#
# # 生成均匀时间戳
# uniform_timestamps = np.linspace(timestamps[0], timestamps[-1], len(timestamps))
#
# # 插值 PPG 信号
# interp_func = interp1d(timestamps, ppg_signal, kind='linear', fill_value="extrapolate")
# fixed_ppg_signal = interp_func(uniform_timestamps)
#
# # 画对比图
# plt.figure(figsize=(10, 4))
# plt.plot(timestamps, ppg_signal, label="Original PPG", linestyle="dotted")
# plt.plot(uniform_timestamps, fixed_ppg_signal, label="Fixed PPG", linestyle="-")
# plt.legend()
# plt.title("Interpolated PPG Signal")
# plt.xlabel("Time (s)")
# plt.ylabel("PPG Value")
# plt.show()
from video_ppgi.Model import model_load
from video_ppgi.face_parse import segment_skin
from video_ppgi.main import crop_and_resize, overlay
from video_ppgi.roi_extraction import extract_roi

def video_load(load_pth):

    VidObj = cv2.VideoCapture(load_pth)
    fs = VidObj.get(cv2.CAP_PROP_FPS)
    total_frames = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fs}, Total Frames: {total_frames}")


    frames = list()
    while True:
        success, frame = VidObj.read()
        if not success:
            print(f"fails to load frames")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    VidObj.release()
    print(f"length : {len(frames)}")
    return frames


if __name__ == "__main__":
    path = r'D:\MA_DATA\video\project1\vid.avi'
    frames = video_load(path)
    model = model_load()
    cropped_resized_frame = crop_and_resize(frames[:1])
    pred = segment_skin(cropped_resized_frame, model, batch_size=1)
    rois = extract_roi(cropped_resized_frame, pred)
    overlayed_roi = overlay(cropped_resized_frame[0].astype(np.uint8), rois, (0, 255, 0), 0.3)
    plt.figure()
    plt.imshow(overlayed_roi)
    #plt.savefig("overlayed_roi.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()