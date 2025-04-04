import os

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def read_video(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)


def read_wave(bvp_file):
    """Reads a bvp signal file. UBFC Dataset"""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)


def crop_and_resize(frames):

    assert frames.ndim == 4, f"Expected frames to have 4 dimensions (T, H, W, C), but got shape {frames.shape}"
    total_frames, _, _, channels = frames.shape
    cropped_and_resized_frame = np.zeros((total_frames, 256, 256, channels), dtype=np.uint8)
    for i in range(0, total_frames):
        frame = frames[i]
        h, w, _ = frame.shape
        short_edge = min(h, w)
        if w > h:
            start_x = (w - short_edge) // 2
            cropped = frame[:, start_x:start_x + short_edge]
        else:
            start_y = (h - short_edge) // 2
            cropped = frame[start_y:start_y + short_edge, :]

        cropped_and_resized_frame[i] = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)

    return cropped_and_resized_frame


def preprocess(frames, bvps):
    """Preprocesses a pair of data.

    Args:
        frames(np.array): Frames in a video.
        bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.

    Returns:
        frame_clips(np.array): processed video data by frames
        bvps_clips(np.array): processed bvp (ppg) labels by frames
    """
    # resize frames and crop for face region
    frames = crop_and_resize(frames)
    # Check data transformation type
    data = list()  # Video data
    f_c = frames.copy()
    data.append(f_c)
    # if config_preprocess.DO_CHUNK:  # chunk data into snippets
    #     frames_clips, bvps_clips = chunk(
    #         data, bvps, config_preprocess.CHUNK_LENGTH)
    # else:
    frames_clips = np.array([data])
    bvps_clips = np.array([bvps])

    return frames_clips, bvps_clips


def preprocess_dataset_subprocess(data_dirs, i, file_list_dict):
    """ invoked by preprocess_dataset for multi_process."""
    filename = os.path.split(data_dirs[i]['path'])[-1]
    saved_filename = data_dirs[i]['index']

    # Read Frames
    frames = read_video(
        os.path.join(data_dirs[i]['path'], "vid.avi"))

    # else:
    #     raise ValueError(
    #         f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

    # Read Labels
    bvps = read_wave(
        os.path.join(data_dirs[i]['path'], "ground_truth.txt"))

    frames_clips, bvps_clips = preprocess(frames, bvps)
    input_name_list, label_name_list = save_multi_process(frames_clips, bvps_clips, saved_filename)
    file_list_dict[i] = input_name_list


def save_multi_process(frames_clips, bvps_clips, filename):
    """Save all the chunked data with multi-thread processing.

    Args:
        frames_clips(np.array): blood volumne pulse (PPG) labels.
        bvps_clips(np.array): the length of each chunk.
        filename: name the filename
    Returns:
        input_path_name_list: list of input path names
        label_path_name_list: list of label path names
    """
    if not os.path.exists(cached_path):
        os.makedirs(cached_path, exist_ok=True)
    count = 0
    input_path_name_list = []
    label_path_name_list = []
    for i in range(len(bvps_clips)):
        assert (len(inputs) == len(labels))
        input_path_name = cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
        label_path_name = cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
        input_path_name_list.append(input_path_name)
        label_path_name_list.append(label_path_name)
        np.save(input_path_name, frames_clips[i])
        np.save(label_path_name, bvps_clips[i])
        count += 1
    return input_path_name_list, label_path_name_list


def multi_process_manager(data_dirs, config_preprocess, multi_process_quota=8):
    """Allocate dataset preprocessing across multiple processes.

    Args:
        data_dirs(List[str]): a list of video_files.
        config_preprocess(Dict): a dictionary of preprocessing configurations
        multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
    Returns:
        file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
    """
    print('Preprocessing dataset...')
    file_num = len(data_dirs)
    choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    # shared data resource
    manager = mp.Manager()  # multi-process manager
    file_list_dict = manager.dict()  # dictionary for all processes to store processed files
    p_list = []  # list of processes
    running_num = 0  # number of running processes

    # in range of number of files to process
    for i in choose_range:
        process_flag = True
        while process_flag:  # ensure that every i creates a process
            if running_num < multi_process_quota:  # in case of too many processes
                # send data to be preprocessing task
                p = mp.Process(target=preprocess_dataset_subprocess,
                               args=(data_dirs, config_preprocess, i, file_list_dict))
                p.start()
                p_list.append(p)
                running_num += 1
                process_flag = False
            for p_ in p_list:
                if not p_.is_alive():
                    p_list.remove(p_)
                    p_.join()
                    running_num -= 1
                    pbar.update(1)
    # join all processes
    for p_ in p_list:
        p_.join()
        pbar.update(1)
    pbar.close()

    return file_list_dict
