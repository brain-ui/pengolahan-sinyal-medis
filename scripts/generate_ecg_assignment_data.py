"""Generate synthetic ECG assignment data for 11 student groups.

Each group receives one raw normal ECG and one raw abnormal ECG. Both signals
contain realistic artifacts (baseline wander, 50 Hz power-line interference,
random noise) that require preprocessing before analysis.

Usage
-----
    uv run python scripts/generate_ecg_assignment_data.py [--output-dir DIR]

Output
------
    data/ecg_assignments/ecg_group01.npz
    ...
    data/ecg_assignments/ecg_group11.npz

Each .npz file contains:
    ecg_normal   : np.ndarray  -- raw normal ECG (30 s, fs=500 Hz)
    ecg_abnormal : np.ndarray  -- raw abnormal ECG (same parameters)
    fs           : float       -- sampling frequency (500 Hz)
    duration     : float       -- signal duration in seconds (30.0)

An answer key is printed to stdout. It is NOT stored in the .npz files so
that students cannot discover the pathology by inspecting the data arrays.
"""

import argparse
from pathlib import Path

import numpy as np


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _g(t: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    """Single Gaussian component."""
    return amp * np.exp(-((t - mu) ** 2) / (2.0 * sigma ** 2))


# ── Beat morphology generators ────────────────────────────────────────────────

def _beat_normal(t: np.ndarray) -> np.ndarray:
    """Standard sinus beat (PQRST). R-peak at t ≈ 0.30 s."""
    b  = _g(t, 0.15, 0.010,  0.15)   # P
    b -= _g(t, 0.28, 0.005,  0.10)   # Q
    b += _g(t, 0.30, 0.005,  1.00)   # R
    b -= _g(t, 0.32, 0.005,  0.20)   # S
    b += _g(t, 0.50, 0.020,  0.30)   # T
    return b


def _beat_tachy(t: np.ndarray) -> np.ndarray:
    """Normal morphology compressed for high heart rates. R-peak at t ≈ 0.20 s."""
    b  = _g(t, 0.08, 0.008,  0.12)   # P
    b -= _g(t, 0.18, 0.005,  0.10)   # Q
    b += _g(t, 0.20, 0.005,  1.00)   # R
    b -= _g(t, 0.22, 0.005,  0.20)   # S
    b += _g(t, 0.32, 0.016,  0.28)   # T
    return b


def _beat_av_block(t: np.ndarray) -> np.ndarray:
    """1st-degree AV block: PR interval ≈ 0.30 s (normal < 0.20 s).
    P wave appears early in the beat window; QRS/T shifted late."""
    b  = _g(t, 0.08, 0.010,  0.15)   # P (early)
    b -= _g(t, 0.36, 0.005,  0.10)   # Q
    b += _g(t, 0.38, 0.005,  1.00)   # R
    b -= _g(t, 0.40, 0.005,  0.20)   # S
    b += _g(t, 0.60, 0.020,  0.30)   # T
    return b


def _beat_bbb(t: np.ndarray) -> np.ndarray:
    """Left bundle branch block: broad M-shaped QRS, discordant (inverted) T.
    QRS duration ≈ 0.16 s (normal < 0.12 s)."""
    b  = _g(t, 0.15, 0.010,  0.15)   # P (normal)
    b += _g(t, 0.30, 0.014,  0.75)   # R1 – first hump of M
    b -= _g(t, 0.36, 0.007,  0.22)   # notch between humps
    b += _g(t, 0.43, 0.013,  0.58)   # R2 – second hump of M
    b -= _g(t, 0.55, 0.022,  0.42)   # T (inverted, discordant)
    return b


def _beat_st_elevation(t: np.ndarray) -> np.ndarray:
    """ST elevation (STEMI): +0.22 mV plateau from QRS end to T onset."""
    b  = _g(t, 0.15, 0.010,  0.15)   # P
    b -= _g(t, 0.28, 0.005,  0.10)   # Q
    b += _g(t, 0.30, 0.005,  1.00)   # R
    b -= _g(t, 0.32, 0.005,  0.20)   # S
    b[t >= 0.33]  += 0.22             # ST plateau
    b[t >= 0.50]  -= 0.22             # plateau ends before T
    b += _g(t, 0.55, 0.025,  0.42)   # prominent T
    return b


def _beat_st_depression(t: np.ndarray) -> np.ndarray:
    """ST depression (ischemia): −0.15 mV trough after QRS, flat T."""
    b  = _g(t, 0.15, 0.010,  0.15)   # P
    b -= _g(t, 0.28, 0.005,  0.10)   # Q
    b += _g(t, 0.30, 0.005,  1.00)   # R
    b -= _g(t, 0.32, 0.005,  0.35)   # deep S
    b[t >= 0.33]  -= 0.15             # ST depression
    b[t >= 0.50]  += 0.15             # ends before T
    b += _g(t, 0.52, 0.018,  0.18)   # small, flat T
    return b


def _beat_long_qt(t: np.ndarray) -> np.ndarray:
    """Long-QT syndrome: T wave at ≈ 0.70 s, QT ≈ 0.46 s (normal ≤ 0.44 s)."""
    b  = _g(t, 0.15, 0.010,  0.15)   # P
    b -= _g(t, 0.28, 0.005,  0.10)   # Q
    b += _g(t, 0.30, 0.005,  1.00)   # R
    b -= _g(t, 0.32, 0.005,  0.20)   # S
    b += _g(t, 0.70, 0.032,  0.38)   # very late, broad T
    return b


def _beat_pvc(t: np.ndarray) -> np.ndarray:
    """PVC: no P wave, wide bizarre QRS, tall R, discordant T. R at t ≈ 0.28 s."""
    b  = -_g(t, 0.16, 0.015,  0.30)  # initial negative deflection
    b +=  _g(t, 0.28, 0.018,  1.30)  # tall wide R
    b -=  _g(t, 0.42, 0.016,  0.68)  # broad S
    b -=  _g(t, 0.58, 0.026,  0.48)  # discordant T (inverted)
    return b


def _beat_vt(t: np.ndarray) -> np.ndarray:
    """Ventricular tachycardia beat: no P, wide bizarre QRS. R at t ≈ 0.22 s."""
    b  = -_g(t, 0.10, 0.012,  0.28)
    b +=  _g(t, 0.22, 0.018,  1.15)
    b -=  _g(t, 0.34, 0.016,  0.72)
    b -=  _g(t, 0.38, 0.018,  0.38)  # discordant T
    return b


def _beat_pac(t: np.ndarray) -> np.ndarray:
    """PAC: different (smaller, earlier) P, otherwise normal QRS. R at t ≈ 0.24 s."""
    b  = _g(t, 0.08, 0.008,  0.07)   # small ectopic P
    b -= _g(t, 0.22, 0.005,  0.10)   # Q
    b += _g(t, 0.24, 0.005,  0.95)   # R
    b -= _g(t, 0.26, 0.005,  0.18)   # S
    b += _g(t, 0.40, 0.018,  0.27)   # T (earlier)
    return b


# ── ECG signal assembly ───────────────────────────────────────────────────────

def _build_regular(t: np.ndarray, fs: float, hr: float,
                   beat_fn, r_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a regular ECG from repeated beats.

    Parameters
    ----------
    t        : time vector
    fs       : sampling rate
    hr       : heart rate (bpm)
    beat_fn  : function(t_beat) -> np.ndarray
    r_offset : time within the beat where R-peak occurs (s)

    Returns
    -------
    ecg      : np.ndarray
    r_peaks  : np.ndarray (sample indices)
    """
    beat_dur = 60.0 / hr
    n_beats  = int(t[-1] / beat_dur) + 2
    ecg      = np.zeros_like(t)
    r_peaks  = []

    for i in range(n_beats):
        t0   = i * beat_dur
        t1   = t0 + beat_dur
        mask = (t >= t0) & (t < t1)
        tb   = t[mask] - t0
        if tb.size == 0:
            continue
        ecg[mask] = beat_fn(tb)
        r_idx = int((t0 + r_offset) * fs)
        if 0 <= r_idx < len(t):
            r_peaks.append(r_idx)

    return ecg, np.array(r_peaks, dtype=int)


def _build_pvc(t: np.ndarray, fs: float, hr: float,
               pvc_every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Regular sinus rhythm with PVCs every N beats (+ compensatory pause)."""
    beat_dur = 60.0 / hr
    ecg      = np.zeros_like(t)
    r_peaks  = []
    t0       = 0.0
    i        = 0

    while t0 < t[-1]:
        is_pvc = (i % pvc_every == pvc_every - 1)
        if is_pvc:
            coupling = beat_dur * 0.72          # early
            t1       = t0 + coupling
            mask     = (t >= t0) & (t < t1)
            tb       = t[mask] - t0
            if tb.size > 0:
                ecg[mask] = _beat_pvc(tb)
                r_idx = int((t0 + 0.28) * fs)
                if 0 <= r_idx < len(t):
                    r_peaks.append(r_idx)
            t0 += coupling + beat_dur           # compensatory pause
        else:
            t1   = t0 + beat_dur
            mask = (t >= t0) & (t < t1)
            tb   = t[mask] - t0
            if tb.size > 0:
                ecg[mask] = _beat_normal(tb)
                r_idx = int((t0 + 0.30) * fs)
                if 0 <= r_idx < len(t):
                    r_peaks.append(r_idx)
            t0 += beat_dur
        i += 1

    return ecg, np.array(r_peaks, dtype=int)


def _build_pac(t: np.ndarray, fs: float, hr: float,
               pac_every: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Regular sinus rhythm with PACs every N beats (non-compensatory pause)."""
    beat_dur = 60.0 / hr
    ecg      = np.zeros_like(t)
    r_peaks  = []
    t0       = 0.0
    i        = 0

    while t0 < t[-1]:
        is_pac = (i % pac_every == pac_every - 1)
        if is_pac:
            coupling = beat_dur * 0.68           # early coupling
            t1       = t0 + coupling
            mask     = (t >= t0) & (t < t1)
            tb       = t[mask] - t0
            if tb.size > 0:
                ecg[mask] = _beat_pac(tb)
                r_idx = int((t0 + 0.24) * fs)
                if 0 <= r_idx < len(t):
                    r_peaks.append(r_idx)
            t0 += coupling + beat_dur * 0.93     # non-compensatory
        else:
            t1   = t0 + beat_dur
            mask = (t >= t0) & (t < t1)
            tb   = t[mask] - t0
            if tb.size > 0:
                ecg[mask] = _beat_normal(tb)
                r_idx = int((t0 + 0.30) * fs)
                if 0 <= r_idx < len(t):
                    r_peaks.append(r_idx)
            t0 += beat_dur
        i += 1

    return ecg, np.array(r_peaks, dtype=int)


def _build_afib(t: np.ndarray, fs: float, mean_hr: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Atrial fibrillation: irregular RR intervals, no P wave, fibrillatory baseline."""
    mean_rr = 60.0 / mean_hr
    ecg     = np.zeros_like(t)
    r_peaks = []
    t0      = 0.0

    while t0 < t[-1]:
        rr   = rng.uniform(mean_rr * 0.60, mean_rr * 1.40)   # high irregularity
        t1   = t0 + rr
        mask = (t >= t0) & (t < t1)
        tb   = t[mask] - t0
        if tb.size > 0:
            # QRS + T only (no P wave)
            b  = -_g(tb, 0.08, 0.005, 0.10)
            b +=  _g(tb, 0.10, 0.005, 0.95)
            b -=  _g(tb, 0.12, 0.005, 0.18)
            b +=  _g(tb, 0.28, 0.018, 0.27)
            ecg[mask] = b
            r_idx = int((t0 + 0.10) * fs)
            if 0 <= r_idx < len(t):
                r_peaks.append(r_idx)
        t0 += rr

    # Fibrillatory baseline (f-waves, 4–8 Hz, low amplitude)
    f1 = rng.uniform(4.5, 6.5)
    f2 = rng.uniform(5.5, 8.0)
    ecg += 0.045 * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi))
    ecg += 0.025 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi))

    return ecg, np.array(r_peaks, dtype=int)


# ── Noise ─────────────────────────────────────────────────────────────────────

def _add_noise(ecg: np.ndarray, t: np.ndarray, rng: np.random.Generator,
               bw_amp: float, bw_freq: float,
               pl_amp: float, gauss_std: float) -> np.ndarray:
    """Superimpose baseline wander, 50 Hz power-line, and Gaussian noise."""
    bw  = (bw_amp       * np.sin(2 * np.pi * bw_freq * t)
           + bw_amp * 0.45 * np.sin(2 * np.pi * bw_freq * 0.38 * t
                                    + rng.uniform(0, 2 * np.pi)))
    pl  = pl_amp * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))
    wn  = rng.normal(0.0, gauss_std, len(t))
    return ecg + bw + pl + wn


# ── Group definitions ─────────────────────────────────────────────────────────
#
# Each entry: (normal_hr, builder_function, builder_kwargs, answer_key_string)
# builder_function signature: (t, fs, **builder_kwargs) -> (ecg, r_peaks)

_GROUPS = [
    # 01 – Bradycardia
    {
        "normal_hr": 72.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=44.0, beat_fn=_beat_normal, r_offset=0.30),
        "key": "Bradikardi (HR ≈ 44 bpm) — irama sinus reguler, RR interval memanjang",
    },
    # 02 – Tachycardia
    {
        "normal_hr": 68.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=152.0, beat_fn=_beat_tachy, r_offset=0.20),
        "key": "Takikardi Sinus (HR ≈ 152 bpm) — RR interval memendek, T-P overlap",
    },
    # 03 – PVC
    {
        "normal_hr": 70.0,
        "abnormal": lambda t, fs, rng: _build_pvc(t, fs, hr=70.0, pvc_every=4),
        "key": "Kontraksi Ventrikel Prematur (PVC) setiap 4 beat — QRS lebar, no P, jeda kompensatori",
    },
    # 04 – Atrial Fibrillation
    {
        "normal_hr": 75.0,
        "abnormal": lambda t, fs, rng: _build_afib(t, fs, mean_hr=92.0, rng=rng),
        "key": "Fibrilasi Atrium (AFib) — RR tidak teratur, tidak ada gelombang P, f-wave baseline",
    },
    # 05 – 1st-degree AV block
    {
        "normal_hr": 65.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=65.0, beat_fn=_beat_av_block, r_offset=0.38),
        "key": "Blok AV Derajat 1 — PR interval ≈ 0.30 s (normal < 0.20 s), morfologi normal",
    },
    # 06 – Left Bundle Branch Block
    {
        "normal_hr": 72.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=70.0, beat_fn=_beat_bbb, r_offset=0.36),
        "key": "Blok Cabang Berkas Kiri (LBBB) — QRS lebar (≈0.16 s), pola M, gelombang T inversi",
    },
    # 07 – ST Elevation (STEMI)
    {
        "normal_hr": 78.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=82.0, beat_fn=_beat_st_elevation, r_offset=0.30),
        "key": "Elevasi ST (STEMI) — segmen ST terangkat +0.22 mV, gelombang T tinggi",
    },
    # 08 – ST Depression (ischemia)
    {
        "normal_hr": 70.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=75.0, beat_fn=_beat_st_depression, r_offset=0.30),
        "key": "Depresi ST (Iskemia) — segmen ST turun −0.15 mV, gelombang T datar/kecil",
    },
    # 09 – Long QT
    {
        "normal_hr": 65.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=62.0, beat_fn=_beat_long_qt, r_offset=0.30),
        "key": "Sindrom QT Panjang (Long QT) — gelombang T sangat terlambat (≈0.70 s), QT ≈ 0.46 s",
    },
    # 10 – PAC
    {
        "normal_hr": 68.0,
        "abnormal": lambda t, fs, rng: _build_pac(t, fs, hr=68.0, pac_every=5),
        "key": "Kontraksi Atrium Prematur (PAC) setiap 5 beat — P ektopik kecil, QRS normal, jeda non-kompensatori",
    },
    # 11 – Ventricular Tachycardia
    {
        "normal_hr": 72.0,
        "abnormal": lambda t, fs, rng: _build_regular(
            t, fs, hr=162.0, beat_fn=_beat_vt, r_offset=0.22),
        "key": "Takikardi Ventrikel (VT) — HR ≈ 162 bpm, QRS lebar & aneh, tidak ada P, T diskordant",
    },
]

# Noise parameter ranges; seeded differently per group so signals look distinct
_NOISE_PARAMS = [
    # (bw_amp, bw_freq, pl_amp, gauss_std)
    (0.22, 0.28, 0.09, 0.030),
    (0.18, 0.20, 0.07, 0.025),
    (0.25, 0.32, 0.11, 0.035),
    (0.20, 0.18, 0.08, 0.028),
    (0.17, 0.25, 0.10, 0.032),
    (0.23, 0.30, 0.06, 0.022),
    (0.19, 0.22, 0.12, 0.040),
    (0.21, 0.27, 0.08, 0.026),
    (0.16, 0.35, 0.09, 0.033),
    (0.24, 0.20, 0.07, 0.029),
    (0.20, 0.24, 0.10, 0.031),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(output_dir: Path, fs: float = 500.0, duration: float = 30.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, duration, 1.0 / fs)

    print("Generating ECG assignment data")
    print("=" * 60)
    print(f"  fs       = {fs} Hz")
    print(f"  duration = {duration} s  ({int(duration * fs)} samples)")
    print(f"  output   = {output_dir}/")
    print()
    print("ANSWER KEY  (do not distribute to students)")
    print("-" * 60)

    for idx, group in enumerate(_GROUPS):
        group_num = idx + 1
        seed_normal   = 100 + group_num
        seed_abnormal = 200 + group_num
        rng_n = np.random.default_rng(seed_normal)
        rng_a = np.random.default_rng(seed_abnormal)

        bw_amp, bw_freq, pl_amp, gauss_std = _NOISE_PARAMS[idx]

        # Normal ECG
        ecg_clean_n, _ = _build_regular(
            t, fs, hr=group["normal_hr"],
            beat_fn=_beat_normal, r_offset=0.30,
        )
        ecg_normal = _add_noise(
            ecg_clean_n, t, rng_n,
            bw_amp=bw_amp, bw_freq=bw_freq,
            pl_amp=pl_amp, gauss_std=gauss_std,
        )

        # Abnormal ECG
        ecg_clean_a, _ = group["abnormal"](t, fs, rng_a)
        # Vary noise slightly for abnormal signal (different phase/seed)
        ecg_abnormal = _add_noise(
            ecg_clean_a, t, rng_a,
            bw_amp=bw_amp * 1.1, bw_freq=bw_freq * 0.9,
            pl_amp=pl_amp * 1.05, gauss_std=gauss_std * 1.1,
        )

        fname = output_dir / f"ecg_group{group_num:02d}.npz"
        np.savez(
            fname,
            ecg_normal=ecg_normal.astype(np.float32),
            ecg_abnormal=ecg_abnormal.astype(np.float32),
            fs=np.float32(fs),
            duration=np.float32(duration),
        )

        print(f"  Kelompok {group_num:>2} | normal HR = {group['normal_hr']:.0f} bpm"
              f" | abnormal: {group['key']}")
        print(f"           --> {fname.name}")

    print()
    print("Done. All files saved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ECG assignment data for 11 student groups."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ecg_assignments"),
        help="Output directory (default: data/ecg_assignments)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=500.0,
        help="Sampling frequency in Hz (default: 500)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Signal duration in seconds (default: 30)",
    )
    args = parser.parse_args()
    generate_all(args.output_dir, fs=args.fs, duration=args.duration)


if __name__ == "__main__":
    main()
