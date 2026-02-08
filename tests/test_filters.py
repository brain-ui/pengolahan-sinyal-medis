"""Tests for medsinyal.filters module."""

import numpy as np

from medsinyal.filters import (
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    moving_average,
    notch_filter,
    remove_baseline_wander,
)


def _make_signal(fs=500.0, duration=2.0, freqs=None):
    """Create a test signal with known frequency components."""
    t = np.arange(0, duration, 1.0 / fs)
    if freqs is None:
        freqs = [(10.0, 1.0)]
    signal = sum(amp * np.sin(2 * np.pi * f * t) for f, amp in freqs)
    return t, signal, fs


def test_lowpass_removes_high_freq():
    _, signal, fs = _make_signal(freqs=[(5.0, 1.0), (100.0, 1.0)])
    filtered = lowpass_filter(signal, cutoff=20.0, fs=fs)
    # High-freq component should be attenuated
    assert np.std(filtered) < np.std(signal)


def test_highpass_removes_low_freq():
    _, signal, fs = _make_signal(freqs=[(0.5, 1.0), (30.0, 1.0)])
    filtered = highpass_filter(signal, cutoff=5.0, fs=fs)
    assert np.std(filtered) < np.std(signal)


def test_bandpass_isolates_band():
    _, signal, fs = _make_signal(freqs=[(2.0, 1.0), (10.0, 1.0), (80.0, 1.0)])
    filtered = bandpass_filter(signal, lowcut=5.0, highcut=15.0, fs=fs)
    # Should mostly keep the 10 Hz component
    assert np.std(filtered) < np.std(signal)
    assert np.std(filtered) > 0.3  # 10 Hz component preserved


def test_notch_removes_50hz():
    _, signal, fs = _make_signal(freqs=[(10.0, 1.0), (50.0, 0.5)])
    filtered = notch_filter(signal, freq=50.0, fs=fs)
    # 50 Hz should be reduced
    fft_orig = np.abs(np.fft.rfft(signal))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs_axis = np.fft.rfftfreq(len(signal), 1.0 / fs)
    idx_50 = np.argmin(np.abs(freqs_axis - 50.0))
    assert fft_filt[idx_50] < fft_orig[idx_50] * 0.3


def test_remove_baseline_wander():
    t, _, fs = _make_signal(duration=5.0)
    # Signal = 10 Hz sine + large DC offset / slow drift
    ecg_like = np.sin(2 * np.pi * 10 * t) + 3.0
    cleaned = remove_baseline_wander(ecg_like, fs, cutoff=0.5)
    # DC offset (mean) should be largely removed
    assert abs(np.mean(cleaned)) < abs(np.mean(ecg_like)) * 0.1


def test_moving_average():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    smoothed = moving_average(signal, window_size=3)
    assert len(smoothed) == len(signal)
    # Middle values should be averaged
    assert abs(smoothed[4] - 5.0) < 0.01
