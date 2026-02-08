"""Tests for medsinyal.ecg module."""

import numpy as np

from medsinyal.ecg import (
    compute_heart_rate,
    compute_hrv_features,
    compute_rr_intervals,
    detect_r_peaks,
    preprocess_ecg,
)


def _make_synthetic_ecg(fs=500.0, duration=5.0, heart_rate=72.0):
    """Generate a simple synthetic ECG for testing."""
    t = np.arange(0, duration, 1.0 / fs)
    beat_duration = 60.0 / heart_rate
    ecg = np.zeros_like(t)
    true_peaks = []

    for i in range(int(duration / beat_duration)):
        peak_time = i * beat_duration + 0.3
        peak_idx = int(peak_time * fs)
        if peak_idx < len(t):
            # Simple R-peak as narrow Gaussian
            ecg += 1.0 * np.exp(-((t - peak_time) ** 2) / (2 * 0.005**2))
            true_peaks.append(peak_idx)

    return t, ecg, fs, np.array(true_peaks)


def test_preprocess_ecg():
    _, ecg, fs, _ = _make_synthetic_ecg()
    # Add noise
    rng = np.random.default_rng(42)
    noisy = ecg + 0.1 * np.sin(2 * np.pi * 50 * np.arange(len(ecg)) / fs)
    noisy += rng.normal(0, 0.02, len(ecg))
    cleaned = preprocess_ecg(noisy, fs)
    assert len(cleaned) == len(noisy)


def test_detect_r_peaks():
    _, ecg, fs, true_peaks = _make_synthetic_ecg()
    detected = detect_r_peaks(ecg, fs)
    # Should find approximately the right number of peaks
    assert abs(len(detected) - len(true_peaks)) <= 2


def test_compute_rr_intervals():
    r_peaks = np.array([0, 500, 1000, 1500])
    rr = compute_rr_intervals(r_peaks, fs=500.0)
    np.testing.assert_allclose(rr, [1.0, 1.0, 1.0])


def test_compute_heart_rate():
    rr = np.array([0.8, 0.85, 0.82])
    hr = compute_heart_rate(rr)
    assert all(60 < h < 80 for h in hr)


def test_compute_hrv_features():
    rr = np.array([0.8, 0.82, 0.78, 0.81, 0.79, 0.83, 0.80])
    features = compute_hrv_features(rr)
    assert "mean_rr" in features
    assert "sdnn" in features
    assert "rmssd" in features
    assert "pnn50" in features
    assert features["mean_rr"] > 0
