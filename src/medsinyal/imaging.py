"""Medical image processing utilities."""

import numpy as np


def apply_window(
    image: np.ndarray, center: float, width: float
) -> np.ndarray:
    """Apply intensity windowing (common for CT/DICOM images).

    Parameters
    ----------
    image : np.ndarray
        Input image (float).
    center : float
        Window center (level).
    width : float
        Window width.

    Returns
    -------
    np.ndarray
        Windowed image, normalized to [0, 1].
    """
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance contrast.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D, uint8 or will be converted).

    Returns
    -------
    np.ndarray
        Equalized image (uint8).
    """
    from skimage import exposure

    return exposure.equalize_hist(image)


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 0.03
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    image : np.ndarray
        Input image (2D).
    clip_limit : float
        Clipping limit for CLAHE.

    Returns
    -------
    np.ndarray
        Enhanced image.
    """
    from skimage import exposure

    return exposure.equalize_adapthist(image, clip_limit=clip_limit)


def detect_edges(
    image: np.ndarray, method: str = "canny", **kwargs
) -> np.ndarray:
    """Detect edges in a medical image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D, float or uint8).
    method : str
        Edge detection method: "canny", "sobel", or "prewitt".
    **kwargs
        Additional arguments passed to the edge detector.

    Returns
    -------
    np.ndarray
        Edge map.
    """
    from skimage import feature, filters

    if method == "canny":
        sigma = kwargs.get("sigma", 1.0)
        return feature.canny(image, sigma=sigma).astype(np.float64)
    elif method == "sobel":
        return filters.sobel(image)
    elif method == "prewitt":
        return filters.prewitt(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def threshold_otsu(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Otsu's thresholding.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D).

    Returns
    -------
    binary : np.ndarray
        Binary image.
    threshold : float
        Computed threshold value.
    """
    from skimage.filters import threshold_otsu as _threshold_otsu

    thresh = _threshold_otsu(image)
    binary = image > thresh
    return binary.astype(np.float64), float(thresh)


def morphological_clean(
    binary: np.ndarray, operation: str = "close", disk_size: int = 3
) -> np.ndarray:
    """Apply morphological operation to clean a binary image.

    Parameters
    ----------
    binary : np.ndarray
        Binary input image.
    operation : str
        "open", "close", "dilate", or "erode".
    disk_size : int
        Radius of the structuring element.

    Returns
    -------
    np.ndarray
        Cleaned binary image.
    """
    from skimage.morphology import closing, dilation, disk, erosion, opening

    selem = disk(disk_size)
    ops = {
        "open": opening,
        "close": closing,
        "dilate": dilation,
        "erode": erosion,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}. Choose from {list(ops)}")
    return ops[operation](binary, selem).astype(np.float64)


def gaussian_smooth(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Smoothed image.
    """
    from skimage.filters import gaussian

    return gaussian(image, sigma=sigma)
