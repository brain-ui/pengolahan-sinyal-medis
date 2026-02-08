"""Tests for medsinyal.imaging module."""

import numpy as np

from medsinyal.imaging import (
    apply_window,
    detect_edges,
    gaussian_smooth,
    histogram_equalization,
    morphological_clean,
    threshold_otsu,
)


def _make_image():
    """Create a simple test image."""
    rng = np.random.default_rng(42)
    img = np.zeros((100, 100), dtype=np.float64)
    img[30:70, 30:70] = 0.8  # bright square
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)


def test_apply_window():
    img = np.random.default_rng(42).uniform(-1000, 1000, (64, 64))
    windowed = apply_window(img, center=0, width=500)
    assert windowed.min() >= 0.0
    assert windowed.max() <= 1.0


def test_histogram_equalization():
    img = _make_image()
    eq = histogram_equalization(img)
    assert eq.shape == img.shape


def test_detect_edges_canny():
    img = _make_image()
    edges = detect_edges(img, method="canny")
    assert edges.shape == img.shape
    assert edges.max() <= 1.0


def test_detect_edges_sobel():
    img = _make_image()
    edges = detect_edges(img, method="sobel")
    assert edges.shape == img.shape


def test_threshold_otsu():
    img = _make_image()
    binary, thresh = threshold_otsu(img)
    assert binary.shape == img.shape
    assert 0 < thresh < 1


def test_morphological_clean():
    img = _make_image()
    binary, _ = threshold_otsu(img)
    cleaned = morphological_clean(binary.astype(bool), operation="close", disk_size=2)
    assert cleaned.shape == binary.shape


def test_gaussian_smooth():
    img = _make_image()
    smoothed = gaussian_smooth(img, sigma=2.0)
    assert smoothed.shape == img.shape
    # Smoothing should reduce variance
    assert np.std(smoothed) < np.std(img)
