"""Sequential RANSAC plane fitting, and filling of gaps inside the fitted planes.

Lengths are relative to the scene size: monocular SfM has no metric scale.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from rr.ply import PointCloud


@dataclass
class PlaneConfig:
    max_planes: int = 12
    distance: float = 0.003          # inlier distance / scene size
    normal_deg: float = 25.0
    min_inlier_fraction: float = 0.02
    iterations: int = 2000
    voxel: float = 0.004             # RANSAC runs on one point per voxel / scene size
    cell: float = 0.005              # fill grid cell / scene size
    close: float = 0.05              # bridge gaps up to ~2x this / scene size
    max_hole: float = 0.02           # fill enclosed holes up to this area / scene size^2
    seed: int = 0


@dataclass
class Plane:
    normal: NDArray[np.float64]      # unit; normal @ x + d = 0
    d: float
    inliers: NDArray[np.int64]       # indices into the full cloud

    def distance(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.abs(points @ self.normal + self.d)


def scene_size(points: NDArray[np.float64]) -> float:
    """Robust bounding-box diagonal."""
    return float(np.linalg.norm(np.subtract(*np.percentile(points, [98, 2], axis=0))))


def _voxel_representatives(points: NDArray[np.float64], size: float) -> NDArray[np.int64]:
    keys = np.floor(points / size).astype(np.int64)
    return np.unique(keys, axis=0, return_index=True)[1]


def _least_squares(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    center = points.mean(axis=0)
    normal = np.linalg.eigh(np.cov((points - center).T))[1][:, 0]
    return normal, float(-normal @ center)


def fit_planes(cloud: PointCloud, config: PlaneConfig | None = None) -> list[Plane]:
    cfg = config or PlaneConfig()
    rng = np.random.default_rng(cfg.seed)
    size = scene_size(cloud.points)
    if len(cloud) < 50 or size <= 0:
        return []
    tol, cos_tol = cfg.distance * size, np.cos(np.radians(cfg.normal_deg))
    # One vote per voxel: area counts, not how densely textured a surface is.
    reps = _voxel_representatives(cloud.points, cfg.voxel * size)
    pts, nrm = cloud.points[reps], cloud.normals[reps]
    left = np.arange(len(reps))
    free = np.ones(len(cloud), dtype=bool)
    min_inliers = max(50, int(cfg.min_inlier_fraction * len(reps)))

    def inliers(n: NDArray, d: float, p: NDArray, pn: NDArray) -> NDArray[np.bool_]:
        return (np.abs(p @ n + d) < tol) & (np.abs(pn @ n) > cos_tol)

    planes: list[Plane] = []
    while len(planes) < cfg.max_planes and len(left) >= min_inliers:
        p, pn = pts[left], nrm[left]
        # Score hypotheses on a subset. Each is one oriented point, so few iterations suffice.
        sub = rng.choice(len(left), min(5000, len(left)), replace=False)
        sp, sn = p[sub], pn[sub]
        best_score, best = -1, 0
        for i in rng.choice(len(left), min(cfg.iterations, len(left)), replace=False):
            score = int(np.count_nonzero(inliers(pn[i], float(-pn[i] @ p[i]), sp, sn)))
            if score > best_score:
                best_score, best = score, i
        n, d = pn[best], float(-pn[best] @ p[best])
        mask = inliers(n, d, p, pn)
        for _ in range(5):  # refit
            if mask.sum() < 3:
                break
            n, d = _least_squares(p[mask])
            mask = inliers(n, d, p, pn)
        if mask.sum() < min_inliers:
            break
        full = np.flatnonzero(free & inliers(n, d, cloud.points, cloud.normals))
        planes.append(Plane(n, d, full))
        free[full] = False
        left = left[~mask]
    return planes


def _disk(radius: int) -> NDArray[np.bool_]:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def fill_plane(cloud: PointCloud, plane: Plane, size: float, config: PlaneConfig | None = None) -> PointCloud:
    """Synthetic points for gaps in a plane's support. Never extends past its outline much."""
    cfg = config or PlaneConfig()
    src = cloud[plane.inliers]
    cell, n = cfg.cell * size, plane.normal
    u = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    radius = max(1, round(cfg.close / cfg.cell))
    uv = np.column_stack([src.points @ u, src.points @ v])
    origin = uv.min(axis=0) - (radius + 1) * cell
    ij = np.floor((uv - origin) / cell).astype(int)
    occ = np.zeros(ij.max(axis=0) + radius + 2, dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    # Ignore specks, which would otherwise be bridged to the main surface.
    labels, count = ndimage.label(occ, np.ones((3, 3)))
    if count:
        keep = np.bincount(labels.ravel()) >= 10
        keep[0] = False
        occ_clean = keep[labels]
    else:
        occ_clean = occ
    filled = ndimage.binary_closing(occ_clean, _disk(radius))
    holes, _ = ndimage.label(ndimage.binary_fill_holes(filled) & ~filled)
    small = np.bincount(holes.ravel()) * cell**2 <= cfg.max_hole * size**2
    small[0] = False
    filled |= small[holes]

    new = np.argwhere(filled & ~occ)
    if len(new) == 0:
        return cloud[np.zeros(0, dtype=np.int64)]
    centers = origin + (new + 0.5) * cell
    points = centers[:, :1] * u + centers[:, 1:] * v - plane.d * n
    # Orient like the surface it extends; copy colour from the nearest supported cell.
    side = n if np.median(src.normals @ n) >= 0 else -n
    color = np.zeros((*occ.shape, 3), dtype=np.uint8)
    color[ij[:, 0], ij[:, 1]] = src.colors
    nearest = ndimage.distance_transform_edt(~occ, return_distances=False, return_indices=True)
    assert isinstance(nearest, np.ndarray)
    ni, nj = nearest[:, new[:, 0], new[:, 1]]
    return PointCloud(points, np.tile(side, (len(points), 1)), color[ni, nj])
