import numpy as np
import matplotlib.pyplot as plt
from code_projects.video_ppgi.signal_processing import filter_signal


def wave_figure(bvp_signal, ppg_signal, fs):
    """Plotting the PPG signal and instantaneous HR over time"""

    T = 20  # 20 seconds
    num = int(fs * T)  # total number of samples
    bvp_filtered = filter_signal(bvp_signal, fs, cutoff_freqs=[0.6, 3.3])[:num]
    ppg_filtered = filter_signal(ppg_signal, fs, cutoff_freqs=[0.6, 3.3])[:num]

    # normalized to [0,1]
    bvp_normalized = (bvp_filtered - np.min(bvp_filtered)) / (np.max(bvp_filtered) - np.min(bvp_filtered))
    ppg_normalized = (ppg_filtered - np.min(ppg_filtered)) / (np.max(ppg_filtered) - np.min(ppg_filtered))
    bvp_normalized += 1 # y offset of bvp

    time_axis = np.linspace(0, T, num)
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, ppg_normalized, color='blue', label='Ground Truth')
    plt.plot(time_axis, bvp_normalized, color='orange', label='Extracted PPG')
    plt.title('PPG Signal over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.xlim(0, T)
    plt.xticks(np.arange(0, T + 1, 5))
    plt.yticks(np.arange(0, 2.1, 0.5))
    #plt.grid(True)
    #plt.legend()  # 显示图例
    plt.tight_layout()
    plt.savefig("ppg_plot.png", bbox_inches='tight', dpi=300)
    plt.show()
# 保存为PNG
# plt.savefig(r"D:\Skin-Segmentation-4-iPPG\log\pic\ppg_waveform.png", dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(10, 4))
# plt.plot(timestamps, heart_rate, color='red')
# plt.title('Instantaneous Heart Rate over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Heart Rate (BPM)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# if __name__=='__main__':
#     file_path = r'D:\MA_DATA\video\project1\ground_truth.txt'
#
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         ppg_signal = np.array([float(val) for val in lines[0].strip().split()])
#         heart_rate = np.array([float(val) for val in lines[1].strip().split()])
#         timestamps = np.array([float(val) for val in lines[2].strip().split()])