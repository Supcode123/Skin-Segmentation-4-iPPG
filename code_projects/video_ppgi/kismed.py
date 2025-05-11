import os
from typing import Tuple
import csv
import cv2
import numpy as np
from tqdm import tqdm
from code_projects.video_ppgi.face_parse import segment_skin, detect_and_crop_faces
from code_projects.video_ppgi.ppgi_signal import extract_bvp_POS
from code_projects.video_ppgi.roi_extraction import extract_roi, apply_masks
from code_projects.video_ppgi.signal_processing import temporal_filtering
from code_projects.video_ppgi.metrics import calculate_metric_per_video, calculate_resuls
from code_projects.video_ppgi.ppg_wave_figure import wave_figure
from code_projects.video_ppgi.UBFC_rPPG import load_video, load_labels_time
from code_projects.video_ppgi.Model import model_load


def get_project_dict(root_dir):
    project_dict = {}
    subjects = sorted(os.listdir(root_dir))  # ["p001", "p002", ...]

    for subject in subjects:
        subject_path = os.path.join(root_dir, subject)  # ./p001
        if os.path.isdir(subject_path):
            tasks = sorted(os.listdir(subject_path))  # ["v01", "v02", ...]
            project_dict[subject] = tasks

    return project_dict


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


def read_bvp_with_timestamps(task_path):

    timestamps = []
    bvp_values = []
    frame_timestamps = []

    with open(os.path.join(task_path,"BVP.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            bvp_values.append(float(row["bvp"]))
    with open(os.path.join(task_path,"frames_timestamps.csv"), "r") as f:
        frame_reader = csv.DictReader(f)
        for row in frame_reader:
            frame_timestamps.append(float(row["timestamp"]))

    return bvp_values, timestamps, frame_timestamps


if __name__ == "__main__":
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    SNR_fft_all = list()

    # data_path = r'S:\XDatabase\PPGI\UBFC\DATASET_2\subject1\vid.avi'

    data_dir = r'D:\MA_DATA\pure'
    projects = get_project_dict(data_dir)
    model = model_load()
    pbar = tqdm(projects.keys(), desc="Processing Projects")
    for key in tqdm(projects.keys(), desc="Processing Projects"):
        tqdm.write(f"... Projects{key} ...")
        for task in tqdm(projects[key], desc="Processing Tasks"):
            task_path = os.path.join(data_dir, key, task)
            frames, fs = load_video(task, os.path.join(task_path, "video_RAW_RGBA.avi") )
            # test_frame = frames[:1]
            cropped_resized_frame = detect_and_crop_faces(frames)
            pred = segment_skin(cropped_resized_frame, model, batch_size=100)
            rois = extract_roi(cropped_resized_frame, pred)
            # temporal filtering
            filtered_rois = temporal_filtering(rois, k=5)
            print("temporal filtering done!")
            # Estimate a PPGI signal
            rgbt_signal = apply_masks(cropped_resized_frame, filtered_rois)
            bvp_signal = extract_bvp_POS(rgbt_signal, fs).reshape(-1)
            # bvp_filtered = filter_signal(bvp_signal, fs, cutoff_freqs=[0.4, 4])

            ppg_labels, ppg_timestamps, img_timestamps = read_bvp_with_timestamps(task_path)
            if key == "p001":
                # output ppg wave
                wave_figure(bvp_signal, ppg_labels, fs, dataset_name="PURE",
                            img_timestamps=img_timestamps, ppg_timestamps=ppg_timestamps)

            hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                            fs_label=60, dataset_name="PURE",
                                                                            img_timestamps=img_timestamps,
                                                                            ppg_timestamps=ppg_timestamps,
                                                                            hr_method='FFT')
            gt_hr_fft_all.append(hr_label_fft)
            predict_hr_fft_all.append(hr_pred_fft)
            SNR_fft_all.append(SNR_fft)

            hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                               fs_label=60, dataset_name="PURE",
                                                                               img_timestamps=img_timestamps,
                                                                               ppg_timestamps=ppg_timestamps,
                                                                               hr_method='Peak')
            gt_hr_peak_all.append(hr_label_peak)
            predict_hr_peak_all.append(hr_pred_peak)
            SNR_peak_all.append(SNR_peak)


    calculate_resuls(gt_hr_fft_all, predict_hr_fft_all, SNR_fft_all, method="FFT")
    calculate_resuls(gt_hr_peak_all, predict_hr_peak_all, SNR_peak_all, method="Peak")