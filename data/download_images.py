"""Download medical image datasets for the course.

Usage:
    uv run python data/download_images.py [--dataset DATASET] [--output-dir DIR]

Datasets:
    chestxray  - Sample chest X-ray images (from scikit-image or generated)
    brainmri   - Sample brain MRI slices (synthetic/sample data)
    all        - Download all datasets

Note:
    Large clinical datasets (NIH ChestX-ray8, CheXpert, BraTS) require
    manual download due to data use agreements. This script downloads
    small sample datasets suitable for course demos.
"""

import argparse
from pathlib import Path

import numpy as np


def download_chestxray_samples(output_dir: Path) -> None:
    """Generate/download sample chest X-ray-like images for demos."""
    from PIL import Image

    dest = output_dir / "chestxray"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample chest X-ray images in {dest}...")

    rng = np.random.default_rng(42)

    for i in range(5):
        # Generate synthetic X-ray-like image (gradient + noise + structures)
        img = np.zeros((256, 256), dtype=np.float64)

        # Body outline (ellipse)
        y, x = np.ogrid[-128:128, -128:128]
        body_mask = (x**2 / 100**2 + y**2 / 120**2) < 1
        img[body_mask] = 0.3

        # Lung fields (two ellipses)
        lung_left = ((x + 40) ** 2 / 35**2 + y**2 / 70**2) < 1
        lung_right = ((x - 40) ** 2 / 35**2 + y**2 / 70**2) < 1
        img[lung_left] = 0.1
        img[lung_right] = 0.1

        # Heart shadow
        heart = ((x + 10) ** 2 / 25**2 + (y + 10) ** 2 / 30**2) < 1
        img[heart] = 0.5

        # Add noise
        img += rng.normal(0, 0.02, img.shape)
        img = np.clip(img, 0, 1)

        # Convert to 8-bit
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(dest / f"sample_xray_{i:02d}.png")

    print(f"  Generated {5} sample X-ray images")
    print(f"Chest X-ray samples complete: {dest}")


def download_brainmri_samples(output_dir: Path) -> None:
    """Generate sample brain MRI-like slices for demos."""
    from PIL import Image

    dest = output_dir / "brainmri"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample brain MRI slices in {dest}...")

    rng = np.random.default_rng(123)

    for i in range(5):
        img = np.zeros((256, 256), dtype=np.float64)

        y, x = np.ogrid[-128:128, -128:128]

        # Skull outline
        skull = (x**2 + y**2) < 110**2
        img[skull] = 0.2

        # Brain tissue
        brain = (x**2 + y**2) < 100**2
        img[brain] = 0.6

        # White matter (inner region)
        wm = (x**2 / 60**2 + y**2 / 70**2) < 1
        img[wm] = 0.8

        # Ventricles
        vent_l = ((x + 15) ** 2 / 8**2 + y**2 / 25**2) < 1
        vent_r = ((x - 15) ** 2 / 8**2 + y**2 / 25**2) < 1
        img[vent_l] = 0.2
        img[vent_r] = 0.2

        # Add tissue variation and noise
        img += rng.normal(0, 0.03, img.shape)
        img = np.clip(img, 0, 1)

        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(dest / f"sample_brain_{i:02d}.png")

    print(f"  Generated {5} sample brain MRI slices")
    print(f"Brain MRI samples complete: {dest}")


DATASETS = {
    "chestxray": download_chestxray_samples,
    "brainmri": download_brainmri_samples,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download medical image datasets for Pengolahan Sinyal Medis course."
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
        default=Path("data/images"),
        help="Output directory (default: data/images)",
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
    print("\nNote: For full clinical datasets, please download manually:")
    print("  - NIH ChestX-ray8: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("  - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/")
    print("  - BraTS: https://www.med.upenn.edu/cbica/brats/")


if __name__ == "__main__":
    main()
