"""Blur test: a frame is kept when it is sharp *relative to its neighbours*."""
import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.ndimage import median_filter


def resize_long_side(image: MatLike, size: int) -> MatLike:
    """Shrink so the longer side is `size`; never enlarges."""
    scale = size / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def sharpness(image: MatLike) -> float:
    """Variance of the Laplacian at a fixed resolution."""
    gray = cv2.cvtColor(resize_long_side(image, 640), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharp_mask(scores: NDArray[np.float64], half_window: int, ratio: float) -> NDArray[np.bool_]:
    """Compare against a local median: plain walls score low even in perfect focus."""
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    return scores >= ratio * median_filter(scores, size=2 * half_window + 1, mode="nearest")
