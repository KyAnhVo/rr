"""Sequential RANSAC plane fitting, splitting into connected segments, and gap filling.

Monocular SfM has no metric scale, so every length here is a multiple of `unit`: the typical
distance from the camera to what it films (phase 3 measures it from the depth maps). That
stays about the same in a closet and in an open-plan floor, whereas the size of the whole
scene does not, so tolerances do not loosen as the scene grows.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from rr.ply import PointCloud


@dataclass
class PlaneConfig:
    """Lengths in units of the viewing distance (about 2 m in a room); areas in units squared."""
    max_planes: int = 40             # RANSAC planes; each can split into several segments
    distance: float = 0.01           # inlier distance
    normal_deg: float = 25.0
    min_area: float = 0.4            # smallest segment kept (and smallest plane looked for)
    iterations: int = 2000
    voxel: float = 0.015             # RANSAC runs on one point per voxel
    cell: float = 0.0175             # segment / fill grid cell
    close: float = 0.175             # gaps up to ~2x this join segments and get filled
    max_hole: float = 0.25           # enclosed holes up to this area are filled too
    seed: int = 0


@dataclass
class Plane:
    """One connected segment of a plane. Coplanar segments (e.g. the same wall line in two
    rooms, or a floor split by a doorway) are separate Planes with the same equation."""
    normal: NDArray[np.float64]      # unit; normal @ x + d = 0
    d: float
    inliers: NDArray[np.int64]       # indices into the full cloud

    def distance(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.abs(points @ self.normal + self.d)


def scene_size(points: NDArray[np.float64]) -> float:
    """Robust bounding-box diagonal."""
    return float(np.linalg.norm(np.subtract(*np.percentile(points, [98, 2], axis=0))))


def plane_axes(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two unit vectors spanning the plane."""
    u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _voxel_representatives(points: NDArray[np.float64], size: float) -> NDArray[np.int64]:
    keys = np.floor(points / size).astype(np.int64)
    return np.unique(keys, axis=0, return_index=True)[1]


def _least_squares(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    center = points.mean(axis=0)
    normal = np.linalg.eigh(np.cov((points - center).T))[1][:, 0]
    return normal, float(-normal @ center)


def _disk(radius: int) -> NDArray[np.bool_]:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def split_segments(
    points: NDArray[np.float64], normal: NDArray[np.float64], unit: float, cfg: PlaneConfig
) -> list[NDArray[np.int64]]:
    """Group in-plane points into connected pieces (indices into `points`), largest first.

    Pieces closer than about 2 * close join, as filling would bridge such a gap anyway.
    Pieces smaller than min_area are dropped."""
    if len(points) == 0:
        return []
    u, v = plane_axes(normal)
    cell = cfg.cell * unit
    uv = np.column_stack([points @ u, points @ v])
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(np.int64)
    occupied = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    occupied[ij[:, 0], ij[:, 1]] = True
    grown = ndimage.binary_dilation(occupied, _disk(max(1, round(cfg.close / cfg.cell))))
    labels, count = ndimage.label(grown, np.ones((3, 3)))
    area = np.bincount(labels[occupied], minlength=count + 1) * cfg.cell**2  # in unit^2
    point_label = labels[ij[:, 0], ij[:, 1]]
    order = np.argsort(point_label, kind="stable")
    starts = np.searchsorted(point_label[order], np.arange(count + 2))
    segments = [order[starts[k]:starts[k + 1]] for k in range(1, count + 1) if area[k] >= cfg.min_area]
    return sorted(segments, key=len, reverse=True)


def fit_planes(cloud: PointCloud, unit: float, config: PlaneConfig | None = None) -> list[Plane]:
    """Plane segments, from the largest plane down to min_area."""
    cfg = config or PlaneConfig()
    rng = np.random.default_rng(cfg.seed)
    if len(cloud) < 50 or unit <= 0:
        return []
    tol, cos_tol = cfg.distance * unit, np.cos(np.radians(cfg.normal_deg))
    # One vote per voxel: area counts, not how densely textured a surface is.
    reps = _voxel_representatives(cloud.points, cfg.voxel * unit)
    pts, nrm = cloud.points[reps], cloud.normals[reps]
    left = np.arange(len(reps))
    free = np.ones(len(cloud), dtype=bool)
    min_support = max(50, int(cfg.min_area / cfg.voxel**2))

    def inliers(n: NDArray, d: float, p: NDArray, pn: NDArray) -> NDArray[np.bool_]:
        return (np.abs(p @ n + d) < tol) & (np.abs(pn @ n) > cos_tol)

    planes: list[Plane] = []
    for _ in range(cfg.max_planes):
        if len(left) < min_support:
            break
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
        if mask.sum() < min_support:
            break
        full = np.flatnonzero(free & inliers(n, d, cloud.points, cloud.normals))
        free[full] = False  # small leftover pieces stay out of later planes, too
        left = left[~mask]
        planes += [Plane(n, d, full[s]) for s in split_segments(cloud.points[full], n, unit, cfg)]
    return planes


def fill_plane(cloud: PointCloud, plane: Plane, unit: float, config: PlaneConfig | None = None) -> PointCloud:
    """Synthetic points for gaps in a plane segment's support. Never extends past its outline much."""
    cfg = config or PlaneConfig()
    src = cloud[plane.inliers]
    cell, n = cfg.cell * unit, plane.normal
    u, v = plane_axes(n)
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
    small = np.bincount(holes.ravel()) * cfg.cell**2 <= cfg.max_hole
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
