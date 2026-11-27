"""Two-view geometry: how many matches agree, and how much parallax is left after rotation."""
import cv2
import numpy as np
from numpy.typing import NDArray

from rr.phase1.config import PairGeometry


def camera_matrix(
    width: int, height: int, focal_factor: float,
    intrinsics: NDArray[np.float64] | None = None, intrinsics_size: tuple[int, int] | None = None,
) -> NDArray[np.float64]:
    """Known intrinsics rescaled from `intrinsics_size` to (width, height), else a focal guess."""
    if intrinsics is None:
        f = focal_factor * max(width, height)
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    K = intrinsics.copy()
    if intrinsics_size is not None:
        K[0] *= width / intrinsics_size[0]
        K[1] *= height / intrinsics_size[1]
    return K


def _rays(pts: NDArray[np.float64], K: NDArray[np.float64]) -> NDArray[np.float64]:
    r = np.column_stack([pts, np.ones(len(pts))]) @ np.linalg.inv(K).T
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def pair_geometry(
    pts_1: NDArray[np.float64], pts_2: NDArray[np.float64], K: NDArray[np.float64], ransac_px: float
) -> PairGeometry:
    """Epipolar inliers, and the parallax that remains once rotation is undone."""
    if len(pts_1) < 8:
        return PairGeometry(0, 0.0)
    E, mask = cv2.findEssentialMat(pts_1, pts_2, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=ransac_px)
    if E is None or E.shape != (3, 3) or mask is None:
        return PairGeometry(0, 0.0)
    keep = mask.ravel().astype(bool)
    if keep.sum() < 8:
        return PairGeometry(int(keep.sum()), 0.0)

    r1, r2 = _rays(pts_1[keep], K), _rays(pts_2[keep], K)
    # Of the two rotations in E, take the one leaving the least parallax. Unlike a
    # cheirality vote, this stays well defined when translation is ~0.
    parallax = np.inf
    for R in cv2.decomposeEssentialMat(E)[:2]:
        cos = np.clip(np.sum((r1 @ R.T) * r2, axis=1), -1, 1)
        parallax = min(parallax, float(np.degrees(np.median(np.arccos(cos)))))
    return PairGeometry(int(keep.sum()), parallax)
