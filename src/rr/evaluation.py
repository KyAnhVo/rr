"""Scores a reconstruction against dataset ground truth (meters)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rr.ply import PointCloud
from rr.sources import Recording

Array = NDArray[np.float64]


@dataclass
class Similarity:
    scale: float
    R: Array
    t: Array

    def apply(self, points: Array) -> Array:
        return self.scale * points @ self.R.T + self.t

    def inverse(self) -> "Similarity":
        return Similarity(1 / self.scale, self.R.T, -self.R.T @ self.t / self.scale)

    def cloud(self, c: PointCloud) -> PointCloud:
        return PointCloud(self.apply(c.points), c.normals @ self.R.T, c.colors)


def umeyama(src: Array, dst: Array) -> Similarity:
    if len(src) < 3:
        raise ValueError("need at least 3 camera positions to align")
    ms, md = src.mean(0), dst.mean(0)
    U, s, Vt = np.linalg.svd((dst - md).T @ (src - ms) / len(src))
    if s[1] < 1e-6 * max(s[0], 1e-12):
        raise ValueError("camera positions are collinear; alignment is ambiguous")
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = float((s * np.diag(D)).sum() / ((src - ms) ** 2).sum(1).mean())
    return Similarity(scale, R, md - scale * R @ ms)


def align_cameras(model: Any, frame_of: dict[str, int], recording: Recording) -> tuple[Similarity, dict[str, float]]:
    src, dst = [], []
    for image in model.images.values():
        pose = recording.color_pose(frame_of[image.name])
        if pose is not None:
            src.append(image.projection_center())
            dst.append(pose[:3, 3])
    a, b = np.array(src).reshape(-1, 3), np.array(dst).reshape(-1, 3)
    sim = umeyama(a, b)
    err = np.linalg.norm(sim.apply(a) - b, axis=1)
    return sim, {
        "ate_rmse_m": float(np.sqrt((err**2).mean())), "ate_max_m": float(err.max()), "poses_compared": len(err),
    }


def read_mesh(path: Path) -> tuple[Array, NDArray[np.int64], NDArray[np.uint8]]:
    """Vertices, triangles and vertex colours of a binary triangle or quad mesh."""
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
             "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}
    with path.open("rb") as f:
        counts: dict[str, int] = {}
        fields: list[tuple[str, str]] = []
        face: list[str] = []
        current = fmt = ""
        while (words := f.readline().decode().split())[:1] != ["end_header"]:
            if not words:
                raise ValueError(f"{path}: bad PLY header")
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                current = words[1]
                counts[current] = int(words[2])
            elif words[0] == "property" and current == "vertex":
                fields.append((words[2], types[words[1]]))
            elif words[0] == "property" and current == "face":
                face.append(" ".join(words[1:4]))
        if fmt != "binary_little_endian" or len(face) != 1 or not face[0].startswith("list"):
            raise ValueError(f"{path}: unsupported mesh layout")
        v = np.fromfile(f, np.dtype(fields), counts["vertex"])
        _, n_type, i_type = face[0].split()
        start = f.tell()
        corners = int(np.fromfile(f, types[n_type], 1)[0])
        f.seek(start)
        faces = np.fromfile(f, np.dtype([("n", types[n_type]), ("i", types[i_type], corners)]), counts["face"])
    if corners not in (3, 4) or len(faces) != counts["face"] or np.any(faces["n"] != corners):
        raise ValueError(f"{path}: not a pure triangle or quad mesh")
    index = faces["i"].astype(np.int64)
    if corners == 4:
        index = np.concatenate([index[:, [0, 1, 2]], index[:, [0, 2, 3]]])
    xyz = np.column_stack([v[k] for k in "xyz"]).astype(np.float64)
    rgb = np.column_stack([v[k] for k in ("red", "green", "blue")]).astype(np.uint8)
    return xyz, index, rgb


def sample_mesh(path: Path, count: int = 2_000_000, seed: int = 0) -> PointCloud:
    """Points spread uniformly by area."""
    rng = np.random.default_rng(seed)
    v, f, rgb = read_mesh(path)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    pick = rng.choice(len(f), count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    w = np.column_stack([1 - r1, r1 * (1 - r2), r1 * r2])
    points = np.einsum("ni,nij->nj", w, v[f[pick]])
    return PointCloud(points, cross[pick] / area[pick, None], rgb[f[pick, 0]])


def visible_mask(points: Array, recording: Recording, frames: list[int], tolerance: float = 0.05) -> NDArray[np.bool_]:
    """Mesh points measured by the depth sensor in at least one of the given frames."""
    seen = np.zeros(len(points), dtype=bool)
    K = recording.depth_intrinsics[:3, :3]
    w, h = recording.depth_size
    for i in frames:
        pose = recording.depth_pose(i)
        if pose is None:
            continue
        cam_from_world = np.linalg.inv(pose)
        cam = points @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
        idx = np.flatnonzero(cam[:, 2] > 0.1)
        uv = cam[idx] @ K.T
        xy = np.round(uv[:, :2] / uv[:, 2:])
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        idx, xy = idx[ok], xy[ok].astype(int)
        z = recording.depth(i)[xy[:, 1], xy[:, 0]]
        seen[idx[(z > 0) & (np.abs(z - cam[idx, 2]) < tolerance)]] = True
    return seen


def sampling_floor(gt: Array) -> float:
    """Median distance a perfect point scores, because the mesh is represented by samples."""
    # Nearest *other* sample: the same Poisson statistics as a random surface point.
    d = cKDTree(gt).query(gt[:20_000], k=2, workers=-1)[0]
    return float(np.median(d[:, 1]))


def score(
    points: Array, gt: Array, visible: NDArray[np.bool_], threshold: float = 0.05
) -> dict[str, float | int]:
    """Accuracy/precision over the prediction; completeness/recall over the ground truth.

    `visible` covers the first len(visible) ground-truth samples, the ones used for completeness:
    they can be fewer, whereas accuracy needs dense samples.
    """
    if len(points) == 0:
        return {"points": 0}
    count = len(points)
    if count > 400_000:
        points = points[np.random.default_rng(0).choice(count, 400_000, replace=False)]
    acc = cKDTree(gt).query(points, workers=-1)[0]
    comp = cKDTree(points).query(gt[:len(visible)], workers=-1)[0]
    precision, recall = float((acc < threshold).mean()), float((comp < threshold).mean())
    result: dict[str, float | int] = {
        "points": count,
        "accuracy_mean_m": float(acc.mean()), "accuracy_median_m": float(np.median(acc)),
        "completeness_mean_m": float(comp.mean()), "completeness_median_m": float(np.median(comp)),
        "chamfer_m": float((acc.mean() + comp.mean()) / 2),
        "precision_5cm": precision, "recall_5cm": recall,
        "fscore_5cm": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    if visible.any():
        result["visible_recall_5cm"] = float((comp[visible] < threshold).mean())
        result["visible_completeness_mean_m"] = float(comp[visible].mean())
    return result
