import os
from typing import Tuple, List
from matplotlib import pyplot as plt
import numpy as np
import cv2
from code_projects.video_ppgi.face_parse import segment_skin
from code_projects.video_ppgi.ppgi_signal import extract_bvp_POS
from code_projects.video_ppgi.roi_extraction import extract_roi, apply_masks
from code_projects.video_ppgi.signal_processing import filter_signal, compute_power_spectrum,\
                                                       temporal_filtering
from code_projects.video_ppgi.Model import model_load
from tqdm import tqdm

from code_projects.video_ppgi.metrics import calculate_metric_per_video, calculate_resuls


def get_video_project_dict(root_dir):
    project_dict = {}
    subjects = sorted(os.listdir(root_dir))

    for idx, subject in enumerate(subjects):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            project_dict[idx] = subject

    return project_dict


def load_labels(root_dir, project,num):
    with open(os.path.join(root_dir, project), 'r') as file:
        lines = file.readlines()
    labels = lines[0].strip()
    labels_array = np.array(labels.split(), dtype=float)
    labels_array = labels_array[:num]
    return labels_array


def load_video(project,video_path):
    video_file = video_path     #VIDEO PATH (e.g. "*.mp4")
    T = 30 # seconds

    frames = list()
    cap = cv2.VideoCapture(video_file)
    fs = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"video in {project} frame_width is {frame_width} \n frame_height is {frame_height}")

    n_frames = int(fs*T)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Video too short! Could only load {i}/{n_frames} frames")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    print(f"Loaded {len(frames)} frames.")
    return frames, fs


def crop_and_resize(frames: List[np.ndarray]):
    cropped_and_resized_frame = []
    for frame in frames:
        h, w, _ = frame.shape
        short_edge = min(h, w)
        if w > h:
            start_x = (w - short_edge) // 2
            cropped = frame[:, start_x:start_x + short_edge]
        else:
            start_y = (h - short_edge) // 2
            cropped = frame[start_y:start_y + short_edge, :]

        resized_frame = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
        cropped_and_resized_frame.append(resized_frame)
    return cropped_and_resized_frame


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,


    Returns:
        image_combined: The combined image. np.ndarray

    """
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def standardized_label(label):
    """Z-score standardization for label signal."""
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label


if __name__ == "__main__":
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    SNR_fft_all = list()
    # data_path = r'S:\XDatabase\PPGI\UBFC\DATASET_2\subject1\vid.avi'
    data_dir = r'D:\MA_DATA\video'
    projects = get_video_project_dict(data_dir)
    model = model_load()
    for key in tqdm(projects.keys(), desc="Processing Projects"):
        data_path = os.path.join(data_dir, projects[key], "vid.avi")
        frames, fs = load_video(projects[key],data_path)
        cropped_resized_frame = crop_and_resize(frames)
        pred = segment_skin(cropped_resized_frame, model, batch_size=100)
        rois = extract_roi(cropped_resized_frame, pred)
        # temporal filtering
        filtered_rois = temporal_filtering(rois, k=5)
        print("temporal filtering done!")
        # Estimate a PPGI signal
        rgbt_signal = apply_masks(cropped_resized_frame, filtered_rois)
        bvp_signal = extract_bvp_POS(rgbt_signal, fs).reshape(-1)
        # bvp_filtered = filter_signal(bvp_signal, fs, cutoff_freqs=[0.4, 4])
        ppg_labels = load_labels(data_dir, projects[key], len(frames))

        hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(standardized_label(bvp_signal),
                                                standardized_label(ppg_labels), fs=fs, hr_method='FFT')
        gt_hr_peak_all.append(hr_label_fft)
        predict_hr_peak_all.append(hr_pred_fft)
        SNR_peak_all.append(SNR_fft)
        hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(standardized_label(bvp_signal),
                                                standardized_label(ppg_labels), fs=fs, hr_method='Peak')
        gt_hr_peak_all.append(hr_label_peak)
        predict_hr_peak_all.append(hr_pred_peak)
        SNR_peak_all.append(SNR_peak)

    calculate_resuls(gt_hr_fft_all, predict_hr_fft_all, SNR_fft_all, method="FFT")
    calculate_resuls(gt_hr_peak_all, predict_hr_peak_all, SNR_peak_all, method="Peak")


    # test_frame = frames[:1]



    # x = np.unique(rois[0])
    # overlayed_roi = overlay(cropped_resized_frame[0].astype(np.uint8), rois[0], (0, 255, 0), 0.3)
    # plt.figure()
    # plt.imshow(overlayed_roi)
    # #plt.savefig("overlayed_roi.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.show()


    # Compute the heart rate
    # F, P = compute_power_spectrum(bvp_filtered, fs)
    #
    # HR = F[np.argmax(P)] * 60
    # print(f"Estimated heart rate is {HR:.2f} BPM")
    #
    # plt.figure()
    # plt.plot(F[F <= 4], P[F <= 4])
    # plt.vlines([HR / 60], P.min(), P.max(), 'r', linestyles='dashed')
    # plt.xlabel("f / Hz")
    # plt.ylabel("Power")
    # plt.savefig("heart_rate.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.show()
