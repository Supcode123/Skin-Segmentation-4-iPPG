import cv2
import numpy as np
from matplotlib import pyplot as plt


with open(r'D:\MA_DATA\video\project1\ground_truth.txt', "r") as f:
    lines = f.readlines()

ppg_signal = np.array([float(x) for x in lines[0].strip().split()])  # 第一行是 PPG
mean_gt = np.mean(ppg_signal)
std_gt = np.std(ppg_signal)

# 打印均值和标准差
print(f'Mean of ground_truth: {mean_gt}')
print(f'Standard Deviation of ground_truth: {std_gt}')

# 判断是否标准化（通常均值接近0，标准差接近1）
if np.abs(mean_gt) < 1e-5 and np.abs(std_gt - 1) < 1e-5:
    print("ground_truth is likely standardized.")
else:
    print("ground_truth is not standardized.")
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
# VidObj = cv2.VideoCapture(r'D:\MA_DATA\video\project1\vid.avi')
# VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
# fs = VidObj.get(cv2.CAP_PROP_FPS)
# success, frame = VidObj.read()
# frames = list()
# while success:
#     frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
#     frame = np.asarray(frame)
#     frames.append(frame)
#     success, frame = VidObj.read()
# print(f"length : {len(frames)}")
