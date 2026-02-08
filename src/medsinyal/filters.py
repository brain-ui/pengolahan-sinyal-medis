"""Digital filter implementations for biomedical signals."""

import numpy as np
from scipy import signal as sig


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, data)


def lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth lowpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    cutoff : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = sig.butter(order, cutoff / nyq, btype="low")
    return sig.filtfilt(b, a, data)


def highpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth highpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    cutoff : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = sig.butter(order, cutoff / nyq, btype="high")
    return sig.filtfilt(b, a, data)


def notch_filter(
    data: np.ndarray,
    freq: float,
    fs: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a notch (band-stop) filter to remove a specific frequency.

    Commonly used to remove 50/60 Hz power line interference.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    freq : float
        Frequency to remove (Hz).
    fs : float
        Sampling frequency (Hz).
    quality_factor : float
        Quality factor (higher = narrower notch).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    b, a = sig.iirnotch(freq, quality_factor, fs)
    return sig.filtfilt(b, a, data)


def remove_baseline_wander(
    data: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Remove baseline wander using a highpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal (typically ECG).
    fs : float
        Sampling frequency (Hz).
    cutoff : float
        Cutoff frequency for baseline removal (Hz). Default 0.5 Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Signal with baseline wander removed.
    """
    return highpass_filter(data, cutoff, fs, order)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a moving average filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    window_size : int
        Number of samples in the averaging window.

    Returns
    -------
    np.ndarray
        Smoothed signal (same length as input, edges zero-padded).
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def design_fir_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    numtaps: int = 101,
) -> np.ndarray:
    """Design a FIR bandpass filter and return coefficients.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    numtaps : int
        Number of filter taps (odd recommended).

    Returns
    -------
    np.ndarray
        FIR filter coefficients.
    """
    return sig.firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)


def apply_fir(data: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply a FIR filter using the given coefficients.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    coefficients : np.ndarray
        FIR filter coefficients.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    return sig.filtfilt(coefficients, [1.0], data)
