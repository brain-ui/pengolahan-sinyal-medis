"""Data loading helpers for synthetic, PhysioNet, and DICOM data."""

from pathlib import Path
from typing import Any

import numpy as np

# Project root: walk up from this file to find the repo root (contains pyproject.toml)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve a data directory path, checking both CWD-relative and project-relative."""
    p = Path(data_dir)
    if p.is_absolute() and p.exists():
        return p
    # Try CWD-relative first
    if p.exists():
        return p
    # Fall back to project-root-relative
    project_relative = _PROJECT_ROOT / p
    if project_relative.exists():
        return project_relative
    return p  # Return original; will raise FileNotFoundError downstream


def load_synthetic(name: str, data_dir: str | Path = "data/synthetic") -> dict[str, Any]:
    """Load a synthetic data file (.npz).

    Parameters
    ----------
    name : str
        Name of the file (without .npz extension), e.g. "synthetic_ecg".
    data_dir : str or Path
        Directory containing synthetic data files.

    Returns
    -------
    dict
        Dictionary with signal arrays and metadata.
    """
    resolved = _resolve_data_dir(data_dir)
    path = resolved / f"{name}.npz"
    data = np.load(path, allow_pickle=False)
    return dict(data)


def load_ecg_record(record_name: str, data_dir: str | Path = "data/physionet/mitbih"):
    """Load an ECG record from PhysioNet via wfdb.

    Parameters
    ----------
    record_name : str
        Record identifier, e.g. "100".
    data_dir : str or Path
        Directory containing the wfdb record files.

    Returns
    -------
    signal : np.ndarray
        ECG signal array, shape (n_samples, n_leads).
    fields : dict
        Record metadata (fs, units, comments, etc.).
    """
    import wfdb

    record_path = str(Path(data_dir) / record_name)
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.__dict__


def load_ecg_annotations(record_name: str, data_dir: str | Path = "data/physionet/mitbih"):
    """Load beat annotations for an ECG record.

    Parameters
    ----------
    record_name : str
        Record identifier, e.g. "100".
    data_dir : str or Path
        Directory containing the wfdb annotation files.

    Returns
    -------
    sample : np.ndarray
        Sample indices of annotations.
    symbol : list[str]
        Annotation symbols (beat types).
    """
    import wfdb

    ann_path = str(Path(data_dir) / record_name)
    ann = wfdb.rdann(ann_path, "atr")
    return ann.sample, ann.symbol


def load_eeg_edf(filepath: str | Path):
    """Load an EEG file in EDF format using MNE.

    Parameters
    ----------
    filepath : str or Path
        Path to the .edf file.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object.
    """
    import mne

    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    return raw


def load_dicom(filepath: str | Path):
    """Load a DICOM file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dcm file.

    Returns
    -------
    ds : pydicom.Dataset
        DICOM dataset.
    pixel_array : np.ndarray
        Pixel data as numpy array.
    """
    import pydicom

    ds = pydicom.dcmread(str(filepath))
    pixel_array = ds.pixel_array.astype(np.float64)
    return ds, pixel_array


def load_image(filepath: str | Path) -> np.ndarray:
    """Load a medical image (PNG, JPEG, TIFF, etc.) as a numpy array.

    Parameters
    ----------
    filepath : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image as a numpy array.
    """
    from PIL import Image

    img = Image.open(str(filepath))
    return np.array(img)
