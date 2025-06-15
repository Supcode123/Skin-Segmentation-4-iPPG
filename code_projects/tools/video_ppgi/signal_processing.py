from typing import List, Tuple

import numpy as np
from scipy import signal


def filter_signal(sig:np.ndarray,fs:float,cutoff_freqs :List[float],order:int=10) -> np.ndarray:
    """Applys butterworth bandpass filter to signal without phase shift (forward-backward filter)

    Args:
        sig (np.ndarray): 1D signal
        fs (float): sampling frequency (Hz)
        cutoff_freqs (List[float,float], optional): cut off frequencies (Hz). Defaults to [0.4,4].
        order (int, optional): filter order. Defaults to 10.

    Returns:
        np.ndarray: filtered signal
    """

    sos = signal.butter(order, cutoff_freqs, 'bandpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered


def temporal_filtering(segmentation_masks, k: int=5):
    """
    Temporally filter each pixel and smooth the labels using the AND operator.

    Parameters:
    segmentation_masks (list of np.array): Each element is a segmentation result of shape (height, width)
    k (int): size of the temporal window

    Returns:
    List[np.array]: smoothed segmentation result of shape (num_frames, height, width)
    """
    print(f"temporal filtering start...,window size = {k}")
    T = len(segmentation_masks)
    H, W = segmentation_masks[0].shape
    filtered_seg = [np.zeros((H, W), dtype=np.uint8) for _ in range(T)]

    for t in range(k-1, T):
        window_stack = np.stack(segmentation_masks[t - k + 1:t + 1], axis=0)  # shape: (k, H, W)
        filtered_seg[t] = np.where(np.all(window_stack == 255, axis=0), 255, 0)

    return filtered_seg


def compute_power_spectrum(input_signal:np.ndarray,fs:float,zero_pad:int=None) -> Tuple[np.ndarray,np.ndarray]:
    """Computes Power spectrum based on Welchs method and a Hamming Window

    Args:
        signal (np.ndarray): 1D signal
        fs (float): sampling frequency (Hz)

    Returns:
        np.ndarray: frequency vector,
        np.ndarray: spectral density
    """
    processed_signal=input_signal
    if zero_pad is not None:
        processed_signal=np.zeros((zero_pad))
        processed_signal[0:input_signal.shape[0]]=input_signal
    f,Pxx = signal.welch(processed_signal,fs,window='hamming',nperseg=None,nfft=zero_pad)
    return f,Pxx