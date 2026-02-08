"""Download PhysioNet datasets for the course.

Usage:
    uv run python data/download_physionet.py [--dataset DATASET] [--output-dir DIR]

Datasets:
    mitbih    - MIT-BIH Arrhythmia Database
    ptbdb     - PTB Diagnostic ECG Database
    eegbci    - EEG Motor Movement/Imagery Dataset
    sleepedf  - Sleep-EDF Database
    all       - Download all datasets
"""

import argparse
from pathlib import Path


def download_mitbih(output_dir: Path) -> None:
    """Download MIT-BIH Arrhythmia Database via wfdb."""
    import wfdb

    dest = output_dir / "mitbih"
    dest.mkdir(parents=True, exist_ok=True)

    # Download a subset of records for course use
    records = ["100", "101", "102", "103", "104", "105"]
    print(f"Downloading MIT-BIH records {records} to {dest}...")

    for rec in records:
        print(f"  Downloading record {rec}...")
        wfdb.dl_database("mitdb", str(dest), records=[rec])

    print(f"MIT-BIH download complete: {dest}")


def download_ptbdb(output_dir: Path) -> None:
    """Download PTB Diagnostic ECG Database (subset)."""
    import wfdb

    dest = output_dir / "ptbdb"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PTB-DB subset to {dest}...")
    # Download a few patient records
    records = [
        "patient001/s0010_re",
        "patient002/s0014_re",
        "patient003/s0021_re",
    ]
    for rec in records:
        print(f"  Downloading record {rec}...")
        wfdb.dl_database("ptbdb", str(dest), records=[rec])

    print(f"PTB-DB download complete: {dest}")


def download_eegbci(output_dir: Path) -> None:
    """Download EEG Motor Movement/Imagery Dataset via MNE."""
    import mne

    dest = output_dir / "eegbci"
    dest.mkdir(parents=True, exist_ok=True)

    subjects = [1, 2, 3]  # First 3 subjects for course use
    runs = [1, 2, 3, 4, 5, 6]  # Baseline + motor imagery runs

    print(f"Downloading EEGBCI subjects {subjects} to {dest}...")
    for subj in subjects:
        print(f"  Downloading subject {subj}...")
        mne.datasets.eegbci.load_data(subj, runs, path=str(dest))

    print(f"EEGBCI download complete: {dest}")


def download_sleepedf(output_dir: Path) -> None:
    """Download Sleep-EDF Database (subset) via MNE."""
    import mne

    dest = output_dir / "sleepedf"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Sleep-EDF subset to {dest}...")
    # Download first 2 subjects
    mne.datasets.sleep_physionet.age.fetch_data(
        subjects=[0, 1], recording=[1], path=str(dest)
    )

    print(f"Sleep-EDF download complete: {dest}")


DATASETS = {
    "mitbih": download_mitbih,
    "ptbdb": download_ptbdb,
    "eegbci": download_eegbci,
    "sleepedf": download_sleepedf,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download PhysioNet datasets for Pengolahan Sinyal Medis course."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/physionet"),
        help="Output directory (default: data/physionet)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        for name, func in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"Downloading: {name}")
            print(f"{'='*60}")
            func(args.output_dir)
    else:
        DATASETS[args.dataset](args.output_dir)

    print("\nDone! All requested datasets downloaded.")


if __name__ == "__main__":
    main()
