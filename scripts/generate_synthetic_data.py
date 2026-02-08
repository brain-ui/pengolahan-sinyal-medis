"""Generate synthetic biomedical signals for course demos and assignments.

Usage:
    uv run python scripts/generate_synthetic_data.py [--output-dir DIR]

Generates:
    - Sine wave compositions (varying frequencies, amplitudes)
    - Synthetic ECG signal (PQRST complex)
    - Synthetic EEG signal (multi-band)
    - Noisy versions of all signals
"""

import argparse
from pathlib import Path

import numpy as np


def generate_sine_compositions(output_dir: Path, fs: float = 500.0) -> None:
    """Generate sine wave compositions with different frequency content."""
    t = np.arange(0, 5.0, 1.0 / fs)  # 5 seconds

    # Simple sine wave
    simple_sine = np.sin(2 * np.pi * 5 * t)

    # Multi-frequency composition
    multi_freq = (
        1.0 * np.sin(2 * np.pi * 1 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.3 * np.sin(2 * np.pi * 12 * t)
        + 0.2 * np.sin(2 * np.pi * 30 * t)
    )

    # Signal with 50 Hz power line interference
    clean_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    noisy_signal = clean_signal + 0.3 * np.sin(2 * np.pi * 50 * t)

    # Chirp signal (frequency sweep)
    f0, f1 = 1.0, 50.0
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * 5.0) * t**2))

    np.savez(
        output_dir / "sine_compositions.npz",
        t=t,
        fs=np.array(fs),
        simple_sine=simple_sine,
        multi_freq=multi_freq,
        clean_signal=clean_signal,
        noisy_signal=noisy_signal,
        chirp=chirp,
    )
    print(f"  Saved sine compositions ({len(t)} samples, fs={fs} Hz)")


def _synthetic_ecg_beat(t_beat: np.ndarray) -> np.ndarray:
    """Generate a single synthetic ECG beat using Gaussian functions."""
    beat = np.zeros_like(t_beat)

    # P wave
    beat += 0.15 * np.exp(-((t_beat - 0.15) ** 2) / (2 * 0.01**2))

    # Q wave
    beat -= 0.1 * np.exp(-((t_beat - 0.28) ** 2) / (2 * 0.005**2))

    # R wave
    beat += 1.0 * np.exp(-((t_beat - 0.3) ** 2) / (2 * 0.005**2))

    # S wave
    beat -= 0.2 * np.exp(-((t_beat - 0.32) ** 2) / (2 * 0.005**2))

    # T wave
    beat += 0.3 * np.exp(-((t_beat - 0.5) ** 2) / (2 * 0.02**2))

    return beat


def generate_synthetic_ecg(
    output_dir: Path, fs: float = 500.0, duration: float = 10.0, heart_rate: float = 72.0
) -> None:
    """Generate a synthetic ECG signal."""
    t = np.arange(0, duration, 1.0 / fs)
    beat_duration = 60.0 / heart_rate
    n_beats = int(duration / beat_duration) + 1

    ecg_clean = np.zeros_like(t)
    r_peak_indices = []

    for i in range(n_beats):
        beat_start = i * beat_duration
        beat_end = beat_start + beat_duration

        mask = (t >= beat_start) & (t < beat_end)
        t_beat = t[mask] - beat_start

        if len(t_beat) > 0:
            ecg_clean[mask] = _synthetic_ecg_beat(t_beat)
            # R-peak is at 0.3 into each beat
            r_idx = int((beat_start + 0.3) * fs)
            if r_idx < len(t):
                r_peak_indices.append(r_idx)

    rng = np.random.default_rng(42)

    # Add baseline wander (low frequency drift)
    baseline_wander = 0.15 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.sin(
        2 * np.pi * 0.1 * t
    )

    # Add 50 Hz power line noise
    powerline_noise = 0.05 * np.sin(2 * np.pi * 50 * t)

    # Add random noise
    random_noise = rng.normal(0, 0.02, len(t))

    ecg_noisy = ecg_clean + baseline_wander + powerline_noise + random_noise

    np.savez(
        output_dir / "synthetic_ecg.npz",
        t=t,
        fs=np.array(fs),
        heart_rate=np.array(heart_rate),
        ecg_clean=ecg_clean,
        ecg_noisy=ecg_noisy,
        baseline_wander=baseline_wander,
        r_peak_indices=np.array(r_peak_indices),
    )
    print(
        f"  Saved synthetic ECG ({duration}s, fs={fs} Hz, HR={heart_rate} bpm, "
        f"{len(r_peak_indices)} beats)"
    )


def generate_synthetic_eeg(
    output_dir: Path, fs: float = 256.0, duration: float = 30.0
) -> None:
    """Generate a synthetic EEG signal with multiple frequency bands."""
    rng = np.random.default_rng(42)
    t = np.arange(0, duration, 1.0 / fs)

    # EEG frequency bands
    bands = {
        "delta": (0.5, 4.0, 20.0),  # freq_low, freq_high, amplitude_uV
        "theta": (4.0, 8.0, 10.0),
        "alpha": (8.0, 13.0, 15.0),
        "beta": (13.0, 30.0, 5.0),
        "gamma": (30.0, 50.0, 2.0),
    }

    eeg_bands = {}
    eeg_clean = np.zeros_like(t)

    for band_name, (f_low, f_high, amp) in bands.items():
        # Sum of random-phase sinusoids within the band
        band_signal = np.zeros_like(t)
        n_components = 5
        freqs = rng.uniform(f_low, f_high, n_components)
        phases = rng.uniform(0, 2 * np.pi, n_components)
        amps = rng.uniform(0.5, 1.5, n_components) * amp / n_components

        for f, p, a in zip(freqs, phases, amps):
            band_signal += a * np.sin(2 * np.pi * f * t + p)

        eeg_bands[band_name] = band_signal
        eeg_clean += band_signal

    # Add alpha burst (eyes closed simulation around t=10-15s)
    alpha_burst_mask = (t >= 10.0) & (t < 15.0)
    alpha_burst = np.zeros_like(t)
    alpha_burst[alpha_burst_mask] = 30.0 * np.sin(
        2 * np.pi * 10.0 * t[alpha_burst_mask]
    )
    eeg_clean += alpha_burst

    # Noisy version
    eeg_noisy = eeg_clean + rng.normal(0, 3.0, len(t))

    # Add eye blink artifacts (large slow deflections)
    blink_times = [3.0, 8.0, 18.0, 25.0]
    blink_artifact = np.zeros_like(t)
    for bt in blink_times:
        blink_artifact += 100.0 * np.exp(-((t - bt) ** 2) / (2 * 0.05**2))
    eeg_with_artifacts = eeg_noisy + blink_artifact

    save_dict = {
        "t": t,
        "fs": np.array(fs),
        "eeg_clean": eeg_clean,
        "eeg_noisy": eeg_noisy,
        "eeg_with_artifacts": eeg_with_artifacts,
        "alpha_burst": alpha_burst,
        "blink_artifact": blink_artifact,
    }
    for band_name, band_signal in eeg_bands.items():
        save_dict[f"band_{band_name}"] = band_signal

    np.savez(output_dir / "synthetic_eeg.npz", **save_dict)
    print(f"  Saved synthetic EEG ({duration}s, fs={fs} Hz, {len(bands)} bands)")


def generate_sampling_demo(output_dir: Path) -> None:
    """Generate signals for demonstrating sampling and aliasing effects."""
    # High-resolution "analog" signal
    fs_analog = 10000.0
    t_analog = np.arange(0, 1.0, 1.0 / fs_analog)
    f_signal = 50.0
    analog_signal = np.sin(2 * np.pi * f_signal * t_analog)

    # Different sampling rates to show aliasing
    sampling_rates = [500.0, 200.0, 120.0, 80.0]
    sampled_signals = {}
    for fs_s in sampling_rates:
        t_sampled = np.arange(0, 1.0, 1.0 / fs_s)
        sampled = np.sin(2 * np.pi * f_signal * t_sampled)
        sampled_signals[f"fs_{int(fs_s)}"] = sampled
        sampled_signals[f"t_{int(fs_s)}"] = t_sampled

    save_dict = {
        "t_analog": t_analog,
        "analog_signal": analog_signal,
        "f_signal": np.array(f_signal),
        **sampled_signals,
    }

    np.savez(output_dir / "sampling_demo.npz", **save_dict)
    print(f"  Saved sampling demo (f={f_signal} Hz, {len(sampling_rates)} sampling rates)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic biomedical signals for course use."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory (default: data/synthetic)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data...")
    print()

    generate_sine_compositions(args.output_dir)
    generate_synthetic_ecg(args.output_dir)
    generate_synthetic_eeg(args.output_dir)
    generate_sampling_demo(args.output_dir)

    print()
    print(f"All synthetic data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
