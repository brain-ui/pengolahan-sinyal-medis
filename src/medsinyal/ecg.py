"""ECG-specific processing: R-peak detection, HRV, feature extraction."""

import numpy as np
from scipy import signal as sig

from .filters import bandpass_filter, notch_filter, remove_baseline_wander


def preprocess_ecg(
    ecg: np.ndarray,
    fs: float,
    remove_baseline: bool = True,
    remove_powerline: bool = True,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
) -> np.ndarray:
    """Preprocess an ECG signal.

    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : float
        Sampling frequency (Hz).
    remove_baseline : bool
        Remove baseline wander.
    remove_powerline : bool
        Remove 50 Hz power line interference.
    bandpass : tuple or None
        Bandpass frequency range (low, high) in Hz.

    Returns
    -------
    np.ndarray
        Preprocessed ECG signal.
    """
    result = ecg.copy()

    if remove_powerline:
        result = notch_filter(result, 50.0, fs)

    if remove_baseline:
        result = remove_baseline_wander(result, fs)

    if bandpass is not None:
        result = bandpass_filter(result, bandpass[0], bandpass[1], fs)

    return result


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float,
    bandpass_range: tuple[float, float] = (5.0, 15.0),
    threshold_factor: float = 0.6,
    min_distance_s: float = 0.2,
) -> np.ndarray:
    """Detect R-peaks using a simplified Pan-Tompkins-like algorithm.

    Steps:
    1. Bandpass filter (5-15 Hz)
    2. Differentiate
    3. Square
    4. Moving average integration
    5. Adaptive thresholding
    6. Peak detection

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal (preferably preprocessed).
    fs : float
        Sampling frequency (Hz).
    bandpass_range : tuple
        Bandpass filter range for QRS enhancement.
    threshold_factor : float
        Fraction of max for threshold (0-1).
    min_distance_s : float
        Minimum distance between R-peaks in seconds.

    Returns
    -------
    np.ndarray
        Indices of detected R-peaks.
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(ecg, bandpass_range[0], bandpass_range[1], fs, order=2)

    # Step 2: Differentiate
    diff = np.diff(filtered)

    # Step 3: Square
    squared = diff ** 2

    # Step 4: Moving average integration
    window_size = int(0.15 * fs)  # 150ms window
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

    # Step 5: Thresholding and peak detection
    threshold = threshold_factor * np.max(integrated)
    min_distance = int(min_distance_s * fs)

    peaks, _ = sig.find_peaks(
        integrated, height=threshold, distance=min_distance
    )

    # Refine: find actual R-peak (maximum in original signal near each detected peak)
    refined_peaks = []
    search_window = int(0.05 * fs)  # 50ms search window
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(ecg), p + search_window)
        local_max = start + np.argmax(ecg[start:end])
        refined_peaks.append(local_max)

    return np.array(refined_peaks, dtype=int)


def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Compute RR intervals from R-peak indices.

    Parameters
    ----------
    r_peaks : np.ndarray
        Indices of R-peaks.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    np.ndarray
        RR intervals in seconds.
    """
    return np.diff(r_peaks) / fs


def compute_heart_rate(rr_intervals: np.ndarray) -> np.ndarray:
    """Compute instantaneous heart rate from RR intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    np.ndarray
        Heart rate in BPM for each interval.
    """
    return 60.0 / rr_intervals


def compute_hrv_features(rr_intervals: np.ndarray) -> dict[str, float]:
    """Compute HRV (Heart Rate Variability) features from RR intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    dict
        Time-domain HRV features:
        - mean_rr: Mean RR interval (s)
        - sdnn: Standard deviation of RR intervals (s)
        - rmssd: Root mean square of successive differences (s)
        - pnn50: Percentage of successive differences > 50ms
        - mean_hr: Mean heart rate (BPM)
        - std_hr: Standard deviation of heart rate (BPM)
    """
    rr_ms = rr_intervals * 1000  # Convert to ms

    # Successive differences
    diff_rr = np.diff(rr_ms)

    hr = 60.0 / rr_intervals

    return {
        "mean_rr": float(np.mean(rr_ms)),
        "sdnn": float(np.std(rr_ms, ddof=1)),
        "rmssd": float(np.sqrt(np.mean(diff_rr ** 2))),
        "pnn50": float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100),
        "mean_hr": float(np.mean(hr)),
        "std_hr": float(np.std(hr, ddof=1)),
    }


def extract_ecg_features(
    ecg: np.ndarray, r_peaks: np.ndarray, fs: float
) -> dict[str, float]:
    """Extract a feature vector from an ECG segment.

    Combines morphological and HRV features suitable for classification.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    r_peaks : np.ndarray
        R-peak indices.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict
        Feature dictionary.
    """
    features = {}

    # RR-based features
    if len(r_peaks) > 1:
        rr = compute_rr_intervals(r_peaks, fs)
        hrv = compute_hrv_features(rr)
        features.update(hrv)
    else:
        features.update({
            "mean_rr": 0.0, "sdnn": 0.0, "rmssd": 0.0,
            "pnn50": 0.0, "mean_hr": 0.0, "std_hr": 0.0,
        })

    # Signal amplitude features
    features["signal_mean"] = float(np.mean(ecg))
    features["signal_std"] = float(np.std(ecg))
    features["signal_max"] = float(np.max(ecg))
    features["signal_min"] = float(np.min(ecg))

    # R-peak amplitude features
    if len(r_peaks) > 0:
        r_amps = ecg[r_peaks]
        features["r_amplitude_mean"] = float(np.mean(r_amps))
        features["r_amplitude_std"] = float(np.std(r_amps))
    else:
        features["r_amplitude_mean"] = 0.0
        features["r_amplitude_std"] = 0.0

    return features
