from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2
from video_ppgi.face_deteck_parse import segment_skin
from video_ppgi.ppgi_algorithms import extract_bvp_POS
from video_ppgi.roi_extraction import extract_roi, apply_masks
from video_ppgi.signal_processing import filter_signal, compute_power_spectrum


def load_video(video_path):
    video_file = video_path     #VIDEO PATH (e.g. "*.mp4")
    T = 20 # seconds

    frames = list()
    cap = cv2.VideoCapture(video_file)
    fs = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"frame_width is {frame_width} \n frame_height is {frame_height}")

    n_frames = int(fs*T)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Video too short! Could only load {i}/{n_frames} frames")
            break
        # h, w, _ = frame.shape
        # crop_size = min(h, w)
        # start_x = (w - crop_size) // 2
        # start_y = (h - crop_size) // 2
        # cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    print(f"Loaded {i+1} frames.")
    return frames, fs


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


if __name__ == "__main__":
    data_path = r"D:\MA_DATA\video\vid.avi"
    frames, fs = load_video(data_path)
    #test_frame = frames[:2]
    #resized_frame = cv2.resize(frames[0], (256, 256), interpolation=cv2.INTER_LINEAR)
    pred = segment_skin(frames)
    rois = extract_roi(frames, pred)
    overlayed_roi = overlay(frames[0].astype(np.uint8), rois[0], (0, 255, 0), 0.3)
    plt.figure()
    plt.imshow(overlayed_roi)
    plt.savefig("overlayed_roi.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()

    # Estimate a PPGI signal
    rgbt_signal = apply_masks(frames, rois)

    bvp_signal = extract_bvp_POS(rgbt_signal, fs).reshape(-1)

    bvp_filtered = filter_signal(bvp_signal, fs, cutoff_freqs=[0.4, 4])

    plt.figure()
    plt.plot(np.arange(bvp_filtered.shape[0]) / fs, bvp_filtered)
    plt.xlabel("t / s")
    plt.savefig("bvp_signal.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()

    # Compute the heart rate
    F, P = compute_power_spectrum(bvp_filtered, fs)

    HR = F[np.argmax(P)] * 60
    print(f"Estimated heart rate is {HR:.2f} BPM")

    plt.figure()
    plt.plot(F[F <= 4], P[F <= 4])
    plt.vlines([HR / 60], P.min(), P.max(), 'r', linestyles='dashed')
    plt.xlabel("f / Hz")
    plt.ylabel("Power")
    plt.savefig("heart_rate.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
