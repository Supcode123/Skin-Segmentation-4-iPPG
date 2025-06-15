import glob
import os
import csv
import shutil
import time
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from code_projects.tools.video_ppgi.face_parse import segment_skin, detect_and_crop_faces, center_crop_faces
from code_projects.tools.video_ppgi.ppgi_signal import extract_bvp_POS
from code_projects.tools.video_ppgi.roi_extraction import extract_roi, apply_masks

from models.Archi import model_select
from tools.video_ppgi.signal_processing import temporal_filtering
from code_projects.tools.video_ppgi.metrics import calculate_metric_per_video, calculate_resuls
from code_projects.tools.video_ppgi.ppg_wave_figure import wave_figure

from code_projects.utils.before_train import parse_eval_args
from code_projects.tools.video_ppgi.video_output import roi_video_output


def get_project_dict(root_dir):
    project_dict = {}
    subjects = sorted(os.listdir(root_dir))  # ["p001", "p002", ...]

    for subject in subjects:
        subject_path = os.path.join(root_dir, subject)  # ./p001
        if os.path.isdir(subject_path):
            tasks = sorted(os.listdir(subject_path))  # ["v01", "v02", ...]
            project_dict[subject] = tasks

    return project_dict


def read_bvp_with_timestamps(task_path):

    timestamps = []
    bvp_values = []
    frame_timestamps = []

    with open(os.path.join(task_path,"BVP.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            bvp_values.append(float(row["bvp"]))
    with open(os.path.join(task_path,"frames_timestamp.csv"), "r") as f:
        frame_reader = csv.DictReader(f)
        for row in frame_reader:
            frame_timestamps.append(float(row["timestamp"]))

    return np.asarray(bvp_values), np.asarray(timestamps), np.asarray(frame_timestamps)

def read_frames(folder_path):
    """Reads a PURE data file, returns frames(T, H, W, 3) """
    frames = list()
    all_png = sorted(glob.glob(os.path.join(folder_path, '*.png')))

    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)


if __name__ == "__main__":
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    SNR_fft_all = list()

    # data_path = r'S:\XDatabase\PPGI\UBFC\DATASET_2\subject1\vid.avi'
    args = parse_eval_args()
    #data_dir = r'H:\kismed'

    assert os.path.isdir(args.data_path), "Invalid data path."
    assert os.path.isdir(args.chkpt_path), "Invalid checkpoint path."
    cur_time = int(round(time.time() * 1000))
    cur_time = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(cur_time / 1000))

    output_path = os.path.join(args.save_path, f"ppgi/{cur_time}")
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)
    print("##### Load config")

    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]
    data_info = cfg_info["DATA"]
    ppgi_info = cfg_info["PPGi"]

    keys = ppgi_info["DATA_PROJECT_ORDER"]
    tasks = ppgi_info['TASK_ORDER']
    data_dir = args.data_path
    projects = get_project_dict(data_dir)

    print("##### Load model")
    model = model_select(model_info, data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))

    project_keys = projects.keys() if keys is None else keys
    for key in tqdm(project_keys, desc="Processing Projects"):
        tqdm.write(f"... Projects {key} ...")
        current_tasks = projects[key] if tasks is None else tasks
        for task in tqdm(current_tasks, desc="Processing Tasks"):
            task_path = os.path.join(data_dir, key, task)
            #frames, fs = load_video(task, os.path.join(task_path, "video_RAW_RGBA.avi") )
            frames = read_frames(os.path.join(task_path, "pictures_ZIP_RAW_RGB"))
            fs = 30
            # test_frame = frames[:1]
            cropped_resized_frame = center_crop_faces(frames)
            detect_crop_faces = detect_and_crop_faces(cropped_resized_frame)
            pred = segment_skin(detect_crop_faces, model, batch_size=train_info['BATCH_SIZE'],
                                works=train_info['WORKERS'])
            rois = extract_roi(detect_crop_faces, pred, data_info['CLASSES'])
            if ppgi_info['ROI_OUTPUT']:
                roi_video_output(cropped_resized_frame=cropped_resized_frame, rois=rois, fs=fs,
                                 output_dir=output_path, projects=projects, key=key, task=task)
            if ppgi_info['METRICS_CAL']:
                # temporal filtering
                filtered_rois = temporal_filtering(rois, k=5)
                print("temporal filtering done!")
                # Estimate a PPGI signal
                rgbt_signal = apply_masks(cropped_resized_frame, filtered_rois)
                bvp_signal = extract_bvp_POS(rgbt_signal, fs)
                if bvp_signal is None:
                    print("BVP signal invalid，skip")
                else:
                    bvp_signal = bvp_signal.reshape(-1)
                # bvp_filtered = filter_signal(bvp_signal, fs, cutoff_freqs=[0.4, 4])

                ppg_labels, ppg_timestamps, img_timestamps = read_bvp_with_timestamps(task_path)
                # if key == "p001" and task == "v01":
                #     # output ppg wave
                #     wave_figure(bvp_signal, ppg_labels, fs, dataset_name="PURE",
                #                 img_timestamps=img_timestamps, ppg_timestamps=ppg_timestamps)

                hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                                fs_label=60, dataset_name="PURE",
                                                                                frame_ts=img_timestamps,
                                                                                gt_ts=ppg_timestamps,
                                                                                hr_method='FFT',
                                                                                win_size=ppgi_info['WIN_SIZE'],
                                                                                step=ppgi_info['STEP'])
                gt_hr_fft_all.append(hr_label_fft)
                predict_hr_fft_all.append(hr_pred_fft)
                SNR_fft_all.append(SNR_fft)

                hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                                   fs_label=60, dataset_name="PURE",
                                                                                   frame_ts=img_timestamps,
                                                                                   gt_ts=ppg_timestamps,
                                                                                   hr_method='Peak',
                                                                                   win_size=ppgi_info['WIN_SIZE'],
                                                                                   step=ppgi_info['STEP'])
                gt_hr_peak_all.append(hr_label_peak)
                predict_hr_peak_all.append(hr_pred_peak)
                SNR_peak_all.append(SNR_peak)

    if ppgi_info['METRICS_CAL']:
        calculate_resuls(gt_hr_fft_all, predict_hr_fft_all, SNR_fft_all, method="FFT")
        calculate_resuls(gt_hr_peak_all, predict_hr_peak_all, SNR_peak_all, method="Peak")