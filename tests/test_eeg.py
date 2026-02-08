"""Tests for medsinyal.eeg module."""

import numpy as np

from medsinyal.eeg import (
    compute_all_band_powers,
    compute_band_power,
    compute_relative_band_powers,
    extract_band,
    extract_epochs,
    simple_artifact_rejection,
)


def _make_eeg(fs=256.0, duration=5.0):
    """Generate a simple synthetic EEG signal."""
    t = np.arange(0, duration, 1.0 / fs)
    # 10 Hz alpha + 25 Hz beta + noise
    rng = np.random.default_rng(42)
    eeg = (
        10.0 * np.sin(2 * np.pi * 10 * t)
        + 3.0 * np.sin(2 * np.pi * 25 * t)
        + rng.normal(0, 1.0, len(t))
    )
    return t, eeg, fs


def test_extract_band():
    _, eeg, fs = _make_eeg()
    alpha = extract_band(eeg, fs, "alpha")
    assert len(alpha) == len(eeg)
    # Alpha band should capture the 10 Hz component
    assert np.std(alpha) > 3.0


def test_compute_band_power():
    _, eeg, fs = _make_eeg()
    alpha_power = compute_band_power(eeg, fs, "alpha")
    beta_power = compute_band_power(eeg, fs, "beta")
    # Alpha should dominate (10 Hz at amplitude 10 vs 25 Hz at amplitude 3)
    assert alpha_power > beta_power


def test_compute_all_band_powers():
    _, eeg, fs = _make_eeg()
    powers = compute_all_band_powers(eeg, fs)
    assert set(powers.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert all(p >= 0 for p in powers.values())


def test_compute_relative_band_powers():
    _, eeg, fs = _make_eeg()
    rel = compute_relative_band_powers(eeg, fs)
    total = sum(rel.values())
    assert abs(total - 1.0) < 0.01


def test_extract_epochs():
    _, eeg, fs = _make_eeg(duration=10.0)
    events = np.array([int(2 * fs), int(5 * fs), int(8 * fs)])
    epochs = extract_epochs(eeg, fs, events, tmin=-0.2, tmax=0.8)
    assert epochs.shape[0] == 3
    expected_len = int(0.2 * fs) + int(0.8 * fs)
    assert epochs.shape[1] == expected_len


def test_simple_artifact_rejection():
    rng = np.random.default_rng(42)
    # 5 clean epochs, 2 with artifacts
    clean = rng.normal(0, 5, (5, 100))
    artifact = rng.normal(0, 5, (2, 100))
    artifact[:, 50] = 200  # spike
    epochs = np.vstack([clean, artifact])

    cleaned, mask = simple_artifact_rejection(epochs, threshold=100.0)
    assert cleaned.shape[0] == 5
    assert mask.sum() == 2
