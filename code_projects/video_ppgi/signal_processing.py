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