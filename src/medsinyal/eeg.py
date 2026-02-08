"""EEG-specific processing: band power, epoch extraction, artifact removal."""

import numpy as np
from scipy import signal as sig

from .filters import bandpass_filter

# Standard EEG frequency bands (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def extract_band(
    eeg: np.ndarray, fs: float, band: str | tuple[float, float], order: int = 4
) -> np.ndarray:
    """Extract a specific frequency band from an EEG signal.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    fs : float
        Sampling frequency (Hz).
    band : str or tuple
        Band name (e.g. "alpha") or frequency range (low, high).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Band-filtered signal.
    """
    if isinstance(band, str):
        low, high = EEG_BANDS[band]
    else:
        low, high = band

    return bandpass_filter(eeg, low, high, fs, order)


def compute_band_power(
    eeg: np.ndarray,
    fs: float,
    band: str | tuple[float, float],
    method: str = "welch",
    nperseg: int | None = None,
) -> float:
    """Compute power in a specific frequency band.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    fs : float
        Sampling frequency (Hz).
    band : str or tuple
        Band name or (low, high) frequency range.
    method : str
        "welch" for Welch's method, "bandpass" for filtered signal variance.
    nperseg : int, optional
        Segment length for Welch's method.

    Returns
    -------
    float
        Band power.
    """
    if isinstance(band, str):
        low, high = EEG_BANDS[band]
    else:
        low, high = band

    if method == "welch":
        if nperseg is None:
            nperseg = min(len(eeg), int(2 * fs))
        freqs, psd = sig.welch(eeg, fs, nperseg=nperseg)
        band_mask = (freqs >= low) & (freqs <= high)
        return float(np.trapezoid(psd[band_mask], freqs[band_mask]))
    elif method == "bandpass":
        filtered = bandpass_filter(eeg, low, high, fs)
        return float(np.var(filtered))
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_band_powers(
    eeg: np.ndarray, fs: float, **kwargs
) -> dict[str, float]:
    """Compute power for all standard EEG bands.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    fs : float
        Sampling frequency (Hz).
    **kwargs
        Additional arguments passed to compute_band_power.

    Returns
    -------
    dict
        Mapping of band name to power value.
    """
    return {
        band: compute_band_power(eeg, fs, band, **kwargs)
        for band in EEG_BANDS
    }


def compute_relative_band_powers(
    eeg: np.ndarray, fs: float, **kwargs
) -> dict[str, float]:
    """Compute relative (normalized) power for all EEG bands.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict
        Mapping of band name to relative power (0-1).
    """
    powers = compute_all_band_powers(eeg, fs, **kwargs)
    total = sum(powers.values())
    if total == 0:
        return {band: 0.0 for band in powers}
    return {band: p / total for band, p in powers.items()}


def extract_epochs(
    eeg: np.ndarray,
    fs: float,
    events: np.ndarray,
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> np.ndarray:
    """Extract epochs around event markers.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal (1D).
    fs : float
        Sampling frequency (Hz).
    events : np.ndarray
        Sample indices of events.
    tmin : float
        Start time relative to event (s), can be negative.
    tmax : float
        End time relative to event (s).

    Returns
    -------
    np.ndarray
        Epochs array of shape (n_events, n_samples_per_epoch).
    """
    n_pre = int(abs(tmin) * fs)
    n_post = int(tmax * fs)
    epoch_len = n_pre + n_post

    epochs = []
    for event_idx in events:
        start = event_idx - n_pre
        end = event_idx + n_post
        if start >= 0 and end <= len(eeg):
            epochs.append(eeg[start:end])

    return np.array(epochs)


def simple_artifact_rejection(
    epochs: np.ndarray, threshold: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Reject epochs with peak-to-peak amplitude exceeding a threshold.

    Parameters
    ----------
    epochs : np.ndarray
        Epochs array of shape (n_epochs, n_samples).
    threshold : float
        Maximum allowed peak-to-peak amplitude.

    Returns
    -------
    clean_epochs : np.ndarray
        Epochs that passed rejection.
    rejected_mask : np.ndarray
        Boolean mask (True = rejected).
    """
    ptp = np.ptp(epochs, axis=1)
    rejected_mask = ptp > threshold
    clean_epochs = epochs[~rejected_mask]
    return clean_epochs, rejected_mask
