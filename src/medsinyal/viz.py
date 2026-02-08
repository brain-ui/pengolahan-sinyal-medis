"""Standard plotting utilities for ECG, EEG, and medical images."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(
    t: np.ndarray,
    signal: np.ndarray,
    title: str = "Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    figsize: tuple[float, float] = (12, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single time-domain signal.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    signal : np.ndarray
        Signal array.
    title, xlabel, ylabel : str
        Plot labels.
    figsize : tuple
        Figure size (only used if ax is None).
    ax : matplotlib Axes, optional
        Axes to plot on; creates new figure if None.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(t, signal, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def plot_signals(
    t: np.ndarray,
    signals: dict[str, np.ndarray],
    title: str = "Signals",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    figsize: tuple[float, float] = (12, 3),
) -> plt.Figure:
    """Plot multiple signals as vertically stacked subplots.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    signals : dict
        Mapping of signal name to signal array.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size per subplot.

    Returns
    -------
    matplotlib Figure
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, sig) in zip(axes, signals.items()):
        ax.plot(t, sig, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def plot_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    title: str = "Frequency Spectrum",
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Magnitude",
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a frequency spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    magnitude : np.ndarray
        Magnitude array.
    xlim : tuple, optional
        Frequency axis limits.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs, magnitude, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)
    return ax


def plot_ecg(
    t: np.ndarray,
    ecg: np.ndarray,
    r_peaks: np.ndarray | None = None,
    title: str = "ECG Signal",
    figsize: tuple[float, float] = (14, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot an ECG signal with optional R-peak markers.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    ecg : np.ndarray
        ECG signal.
    r_peaks : np.ndarray, optional
        Indices of R-peaks to mark.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(t, ecg, "k-", linewidth=0.8)
    if r_peaks is not None:
        ax.plot(t[r_peaks], ecg[r_peaks], "rv", markersize=8, label="R-peaks")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_eeg_bands(
    t: np.ndarray,
    bands: dict[str, np.ndarray],
    title: str = "EEG Frequency Bands",
    figsize: tuple[float, float] = (14, 2.5),
) -> plt.Figure:
    """Plot EEG frequency bands as stacked subplots.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    bands : dict
        Mapping of band name (e.g. "alpha") to signal array.
    """
    band_colors = {
        "delta": "#1f77b4",
        "theta": "#ff7f0e",
        "alpha": "#2ca02c",
        "beta": "#d62728",
        "gamma": "#9467bd",
    }

    n = len(bands)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, sig) in zip(axes, bands.items()):
        color = band_colors.get(name, None)
        ax.plot(t, sig, linewidth=0.6, color=color)
        ax.set_ylabel(f"{name}\n(\u00b5V)")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_image(
    image: np.ndarray,
    title: str = "Medical Image",
    cmap: str = "gray",
    figsize: tuple[float, float] = (8, 8),
    ax: plt.Axes | None = None,
    colorbar: bool = False,
) -> plt.Axes:
    """Plot a medical image.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    title : str
        Plot title.
    cmap : str
        Colormap.
    colorbar : bool
        Whether to show colorbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_image_comparison(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    cmap: str = "gray",
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Plot multiple images side by side for comparison.

    Parameters
    ----------
    images : sequence of np.ndarray
        Images to compare.
    titles : sequence of str
        Title for each image.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off")

    fig.tight_layout()
    return fig
