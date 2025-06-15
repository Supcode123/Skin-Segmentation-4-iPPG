import os
import shutil
import time
import torch
import yaml

from code_projects.tools.video_ppgi.face_parse import segment_skin, detect_and_crop_faces
from code_projects.tools.video_ppgi.ppgi_signal import extract_bvp_POS
from code_projects.tools.video_ppgi.roi_extraction import extract_roi, apply_masks
from models.Archi import model_select
from tools.video_ppgi.signal_processing import temporal_filtering
from tqdm import tqdm

from code_projects.tools.video_ppgi.metrics import calculate_metric_per_video, calculate_resuls
from code_projects.tools.video_ppgi.ppg_wave_figure import wave_figure
from code_projects.tools.video_ppgi.UBFC_rPPG import load_video, load_labels_time
from code_projects.tools.video_ppgi.PURE import read_frames, read_wave_with_timestamps
from code_projects.utils.before_train import parse_eval_args
from code_projects.tools.video_ppgi.video_output import roi_video_output


def get_project_dict(root_dir):
    project_dict = {}
    subjects = sorted(os.listdir(root_dir))
    for idx, subject in enumerate(subjects):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            project_dict[idx] = subject

    return project_dict


if __name__ == "__main__":
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    SNR_fft_all = list()

    args = parse_eval_args()
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
    # data_path = r'S:\XDatabase\PPGI\UBFC\DATASET_2\subject1\vid.avi'
    data = ppgi_info['DATA_NAME'] # /  "UBFC" "PURE"
    data_dir = args.data_path
    projects = get_project_dict(data_dir)

    print("##### Load model")
    model = model_select(model_info, data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))

    project_keys = projects.keys() if keys is None else keys
    for key in tqdm(project_keys, desc="Processing Projects"):

        if data == "UBFC":
            data_path = os.path.join(data_dir, projects[key], "vid.avi")
            frames, fs = load_video(projects[key], data_path)
        elif data == "PURE":
            data_path = os.path.join(data_dir, projects[key], projects[key])
            frames = read_frames(data_path)
            fs = 30

        # test_frame = frames[:1]
        cropped_resized_frame = detect_and_crop_faces(frames)
        pred = segment_skin(cropped_resized_frame, model, batch_size=train_info['BATCH_SIZE'],
                            works=train_info['WORKERS'])
        rois = extract_roi(cropped_resized_frame, pred, data_info['CLASSES'])
        if ppgi_info['ROI_OUTPUT']:
           roi_video_output(cropped_resized_frame=cropped_resized_frame, rois=rois, fs=fs,
                            output_dir=output_path, projects=projects, key=key)
        if ppgi_info['METRICS_CAL']:
            # temporal filtering
            filtered_rois = temporal_filtering(rois, k=5)
            print("temporal filtering done!")
            # Estimate a PPGI signal
            rgbt_signal = apply_masks(cropped_resized_frame, filtered_rois)
            bvp_signal = extract_bvp_POS(rgbt_signal, fs).reshape(-1)

            if data == "UBFC":
                ppg_labels, timestamps = load_labels_time(data_dir, projects[key], len(frames))
                # if key == 0:
                #     #output ppg wave
                #     wave_figure(bvp_signal, ppg_labels, fs, data,)
                hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels, frame_ts=timestamps,
                                                                                fs=fs, fs_label=fs, hr_method='FFT',
                                                                                win_size=ppgi_info['WIN_SIZE'],
                                                                                step=ppgi_info['STEP'])
                gt_hr_fft_all.append(hr_label_fft)
                predict_hr_fft_all.append(hr_pred_fft)
                SNR_fft_all.append(SNR_fft)
        #
                hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels, frame_ts=timestamps,
                                                                                fs=fs, fs_label=fs, hr_method='Peak',
                                                                                win_size=ppgi_info['WIN_SIZE'],
                                                                                step=ppgi_info['STEP'])
                gt_hr_peak_all.append(hr_label_peak)
                predict_hr_peak_all.append(hr_pred_peak)
                SNR_peak_all.append(SNR_peak)


            elif data == "PURE":
                ppg_labels, ppg_timestamps, img_timestamps = read_wave_with_timestamps(data_dir, projects[key])
                # if key == 0:
                #     # output ppg wave
                #     wave_figure(bvp_signal, ppg_labels, fs, data, img_timestamps, ppg_timestamps)
                hr_label_fft, hr_pred_fft, SNR_fft = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                                fs_label=60, dataset_name=data,
                                                                                frame_ts=img_timestamps,
                                                                                gt_ts=ppg_timestamps,
                                                                                hr_method='FFT',
                                                                                win_size=ppgi_info['WIN_SIZE'],
                                                                                step=ppgi_info['STEP'])
                gt_hr_fft_all.append(hr_label_fft)
                predict_hr_fft_all.append(hr_pred_fft)
                SNR_fft_all.append(SNR_fft)

                hr_label_peak, hr_pred_peak, SNR_peak = calculate_metric_per_video(bvp_signal, ppg_labels, fs=fs,
                                                                                   fs_label=60, dataset_name=data,
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


