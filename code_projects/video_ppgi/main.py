import os
from typing import Tuple, List
from matplotlib import pyplot as plt
import numpy as np
import cv2
from code_projects.video_ppgi.face_parse import segment_skin, detect_and_crop_faces
from code_projects.video_ppgi.ppgi_signal import extract_bvp_POS
from code_projects.video_ppgi.roi_extraction import extract_roi, apply_masks
from code_projects.video_ppgi.signal_processing import filter_signal, compute_power_spectrum, \
    temporal_filtering
from code_projects.video_ppgi.Model import model_load
from tqdm import tqdm

from code_projects.video_ppgi.metrics import calculate_metric_per_video, calculate_resuls
from code_projects.video_ppgi.ppg_wave_figure import wave_figure
from code_projects.video_ppgi.UBFC_rPPG import load_video, load_labels_time
from code_projects.video_ppgi.PURE import read_frames, read_wave_with_timestamps


def get_project_dict(root_dir):
    project_dict = {}
    subjects = sorted(os.listdir(root_dir))
    for idx, subject in enumerate(subjects):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            project_dict[idx] = subject

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


if __name__ == "__main__":
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    SNR_fft_all = list()

    # data_path = r'S:\XDatabase\PPGI\UBFC\DATASET_2\subject1\vid.avi'
    data = "PURE"  # /  "UBFC"
    data_dir = r'D:\MA_DATA\pure'
    projects = get_project_dict(data_dir)
    model = model_load()

    for key in tqdm(projects.keys(), desc="Processing Projects"):
        if data == "UBFC":
            data_path = os.path.join(data_dir, projects[key], "vid.avi")
            frames, fs = load_video(projects[key], data_path)
        elif data == "PURE":
            data_path = os.path.join(data_dir, projects[key], projects[key])
            frames = read_frames(data_path)
            fs = 30

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
        if data == "UBFC":
            ppg_labels, timestamps = load_labels_time(data_dir, projects[key], len(frames))
            if key == 0:
                # output ppg wave
                wave_figure(bvp_signal, ppg_labels, fs, data)
            hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels,
                                                                            fs=fs, fs_label=fs, hr_method='FFT')
            gt_hr_fft_all.append(hr_label_fft)
            predict_hr_fft_all.append(hr_pred_fft)
            SNR_fft_all.append(SNR_fft)

            hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels,
                                                                               fs=fs, fs_label=fs, hr_method='Peak')
            gt_hr_peak_all.append(hr_label_peak)
            predict_hr_peak_all.append(hr_pred_peak)
            SNR_peak_all.append(SNR_peak)

        elif data == "PURE":
            ppg_labels, ppg_timestamps, img_timestamps = read_wave_with_timestamps(data_dir, projects[key])
            if key == 0:
                # output ppg wave
                wave_figure(bvp_signal, ppg_labels, fs, data, img_timestamps, ppg_timestamps)
            hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                            fs_label=60, dataset_name=data,
                                                                            img_timestamps=img_timestamps,
                                                                            ppg_timestamps=ppg_timestamps,
                                                                            hr_method='FFT')
            gt_hr_fft_all.append(hr_label_fft)
            predict_hr_fft_all.append(hr_pred_fft)
            SNR_fft_all.append(SNR_fft)

            hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                               fs_label=60, dataset_name=data,
                                                                               img_timestamps=img_timestamps,
                                                                               ppg_timestamps=ppg_timestamps,
                                                                               hr_method='Peak')
            gt_hr_peak_all.append(hr_label_peak)
            predict_hr_peak_all.append(hr_pred_peak)
            SNR_peak_all.append(SNR_peak)


    calculate_resuls(gt_hr_fft_all, predict_hr_fft_all, SNR_fft_all, method="FFT")
    calculate_resuls(gt_hr_peak_all, predict_hr_peak_all, SNR_peak_all, method="Peak")

    # #x = np.unique(rois[0])
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
