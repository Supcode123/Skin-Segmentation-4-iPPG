from copy import deepcopy
import numpy as np
import scipy
from scipy.sparse import spdiags
from scipy.signal import butter

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.6 Hz
        to 3.3 Hz.
        Ref for low_pass and high_pass filters:
        R. Cassani, A. Tiwari and T. H. Falk, "Optimal filter characterization for photoplethysmography-based pulse rate and
        pulse power spectrum estimation," 2020 IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada,
        doi: 10.1109/EMBC44109.2020.9175396.

        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR


def calculate_metric_per_video(predictions, labels, fs=30, fs_label=30, dataset_name: str = "UBFC",
                               frame_ts=None, gt_ts=None, use_bandpass=True,
                               hr_method='FFT', win_size=30, step=15):
    """Calculate video-level HR and SNR"""
    if dataset_name == "UBFC":
       assert len(predictions) == len(labels), "The length of the prediction and labels don't match "
    predictions = normalize_signal(_detrend(predictions, 100))
    labels = normalize_signal(_detrend(labels, 100))



    t_start = frame_ts[0]
   # t_end = frame_ts[-1] - win_size
    t_end = frame_ts[0] + win_size
    t = t_start

    hr_labels = []
    hr_preds = []
    snrs = []
    while t < t_end:
        t0, t1 = t, t + win_size
        vid_mask = (frame_ts >= t0) & (frame_ts < t1)
        # predicted bvp
        pred_win = predictions[vid_mask]
        # gt
        if dataset_name == "UBFC":
            label_win = labels[vid_mask] # fs = fs_label
        elif dataset_name == "PURE":
            gt_mask = (gt_ts >= t0) & (gt_ts < t1)
            label_win = labels[gt_mask]
        if hr_method == 'FFT':
            if use_bandpass:
                # bandpass filter between [0.75, 2.5] Hz, equals [45, 150] beats per min
                # bandpass filter between [0.6, 3.3] Hz, equals [36, 198] beats per min
                [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
                pred_win = scipy.signal.filtfilt(b, a, np.double(pred_win))
                [d, c] = butter(1, [0.6 / fs_label * 2, 3.3 / fs_label * 2], btype='bandpass')
                label_win = scipy.signal.filtfilt(d, c, np.double(label_win))
            hr_pred = _calculate_fft_hr(pred_win, fs=fs)
            hr_label = _calculate_fft_hr(label_win, fs=fs_label)

        elif hr_method == 'Peak':
            hr_guess = _calculate_fft_hr(pred_win, fs=fs)
            low_cutoff = max(0.4, 0.8 * hr_guess / 60)  # Hz
            high_cutoff = min(4.0, 1.2 * hr_guess / 60)  # Hz

            if use_bandpass:
                [b, a] = scipy.signal.butter(1, [low_cutoff / fs * 2, high_cutoff / fs * 2], btype='bandpass')
                pred_win = scipy.signal.filtfilt(b, a, np.double(pred_win))
                [d, c] = scipy.signal.butter(1, [low_cutoff / fs_label * 2, high_cutoff / fs_label * 2], btype='bandpass')
                label_win = scipy.signal.filtfilt(d, c, np.double(label_win))

            hr_pred = _calculate_peak_hr(pred_win, fs=fs)
            hr_label = _calculate_peak_hr(label_win, fs=fs_label)

        else:
            raise ValueError('Please use FFT or Peak to calculate your HR.')
        snr = _calculate_SNR(pred_win, hr_label, fs=fs)
        t += step
        hr_preds.append(hr_pred)
        hr_labels.append(hr_label)
        snrs.append(snr)

    return np.mean(hr_labels), np.mean(hr_preds), np.mean(snrs)


def calculate_resuls(gt_hr_all, predict_hr_all, SNR_all, method="FFT"):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    print("Calculating metrics!")

    gt_hr_all = np.array(gt_hr_all)
    predict_hr_all = np.array(predict_hr_all)
    SNR_all = np.array(SNR_all)
    num_test_samples = len(predict_hr_all)

    MAE = np.mean(np.abs(predict_hr_all - gt_hr_all))
    standard_error = np.std(np.abs(predict_hr_all - gt_hr_all)) / np.sqrt(num_test_samples)
    print(f"MAE ({method} Label): {MAE} +/- {standard_error}")

    # Calculate the squared errors, then RMSE, in order to allow
    # for a more robust and intuitive standard error that won't
    # be influenced by abnormal distributions of errors.
    squared_errors = np.square(predict_hr_all - gt_hr_all)
    RMSE = np.sqrt(np.mean(squared_errors))
    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
    print(f"RMSE ({method} Label): {RMSE} +/- {standard_error}")

    # Pearson = np.corrcoef(predict_hr_all, gt_hr_all)
    # correlation_coefficient = Pearson[0][1]
    # standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
    # print(f"Pearson ({method} Label): {correlation_coefficient} +/- {standard_error}")

    SNR = np.mean(SNR_all)
    standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
    print(f"SNR ({method} Label): {SNR} +/- {standard_error} (dB)")
