"""Stage 5: 2D -> 3D label fusion.

Every dense point (measured, and plane fill) is projected into every keyframe where it is
visible, and the stage 4 label under it counts as a vote, weighted by how squarely that camera
sees the surface. Then:

  * plane prior: a point on a stage 3 plane whose vote is weak or missing takes the plane's
    majority label (this labels most fill points, which sit on walls no camera matched);
  * neighbours: anything still unlabelled takes the majority of its nearest labelled points.

Visibility uses the MVS depth map where it has an estimate. Elsewhere (e.g. blank walls) it
uses a coarse z-buffer of the cloud itself, so points behind furniture do not vote.
"""
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rr.colmap import load_pycolmap, read_depth_map
from rr.phase4 import read_labels, read_lookup
from rr.ply import PointCloud, concatenate, read_ply, write_ply
from rr.taxonomy import CLASSES, OBJECT, UNKNOWN, class_colors, is_structure

# How each point got its label.
FROM_VOTES, FROM_PLANE, FROM_NEIGHBOURS, UNLABELED = 0, 1, 2, 3


@dataclass
class FusionConfig:
    depth_tolerance: float = 0.03    # relative: visible if this close to the measured depth
    zbuffer_cell: int = 4            # pixels per z-buffer cell where there is no measured depth
    min_weight: float = 0.1          # floor of the |cos(view angle)| vote weight
    plane_confidence: float = 0.6    # below this vote share, a plane point takes the plane's label
    neighbours: int = 8


@dataclass
class View:
    name: str
    K: NDArray[np.float64]               # 3x3, at the label image's resolution
    cam_from_world: NDArray[np.float64]  # 3x4
    labels: NDArray[np.int8]             # coarse class per pixel, -1 = unknown
    depth: NDArray[np.float32] | None = None  # any resolution; 0 = no estimate


@dataclass
class Fusion:
    classes: NDArray[np.int8]        # final label per point
    source: NDArray[np.int8]         # FROM_VOTES / FROM_PLANE / FROM_NEIGHBOURS / UNLABELED
    confidence: NDArray[np.float32]  # winning share of the votes, 0 without votes
    votes_only: NDArray[np.int8]     # ablation: multi-view vote, no plane prior or neighbours
    single_view: NDArray[np.int8]    # ablation: label from the single most head-on view
    views_seen: NDArray[np.int16]    # views that voted for the point


@dataclass
class FusionResult:
    points: int
    views: int
    labeled_by_votes: int
    labeled_by_plane: int
    labeled_by_neighbours: int
    unlabeled: int
    structure_points: int
    object_points: int
    class_counts: dict[str, int] = field(default_factory=dict)


def visible(
    points: NDArray[np.float64], view: View, cfg: FusionConfig
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Indices of the points the view sees, their pixels (x, y) and camera-frame positions."""
    h, w = view.labels.shape
    R, t = view.cam_from_world[:, :3], view.cam_from_world[:, 3]
    cam = points @ R.T + t
    idx = np.flatnonzero(cam[:, 2] > 1e-6)
    proj = cam[idx] @ view.K.T
    xy = np.floor(proj[:, :2] / proj[:, 2:])
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    idx, xy = idx[ok], xy[ok].astype(np.int64)
    z = cam[idx, 2]

    # Nearest point per z-buffer cell; a point is in front if within tolerance of it.
    c = cfg.zbuffer_cell
    cw = (w + c - 1) // c
    cell = (xy[:, 1] // c) * cw + xy[:, 0] // c
    nearest = np.full(cw * ((h + c - 1) // c), np.inf)
    np.minimum.at(nearest, cell, z)
    seen = z <= nearest[cell] * (1 + cfg.depth_tolerance)
    if view.depth is not None:
        dh, dw = view.depth.shape
        d = view.depth[xy[:, 1] * dh // h, xy[:, 0] * dw // w]
        measured = d > 0
        seen = np.where(measured, np.abs(d - z) <= cfg.depth_tolerance * z, seen)
    return idx[seen], xy[seen], cam[idx[seen]]


def fuse(
    points: NDArray[np.float64], normals: NDArray[np.float64], views: Iterable[View],
    plane_of: NDArray[np.integer] | None = None, cfg: FusionConfig | None = None,
) -> Fusion:
    """`plane_of` is each point's stage 3 plane, -1 for none."""
    cfg = cfg or FusionConfig()
    n, C = len(points), len(CLASSES)
    votes = np.zeros(n * C, np.float64)
    best_weight = np.zeros(n)
    single_view = np.full(n, UNKNOWN, np.int8)
    views_seen = np.zeros(n, np.int16)
    has_normal = np.linalg.norm(normals, axis=1) > 0
    for view in views:
        idx, xy, cam = visible(points, view, cfg)
        label = view.labels[xy[:, 1], xy[:, 0]].astype(np.int64)
        keep = label >= 0
        idx, label, cam = idx[keep], label[keep], cam[keep]
        cos = np.abs(np.sum((normals[idx] @ view.cam_from_world[:, :3].T) * cam, axis=1))
        cos /= np.linalg.norm(cam, axis=1)
        weight = np.where(has_normal[idx], np.maximum(cos, cfg.min_weight), 1.0)
        votes += np.bincount(idx * C + label, weights=weight, minlength=n * C)
        views_seen[idx] += 1
        better = weight > best_weight[idx]
        best_weight[idx[better]] = weight[better]
        single_view[idx[better]] = label[better]

    votes = votes.reshape(n, C)
    total = votes.sum(axis=1)
    voted = total > 0
    confidence = np.where(voted, votes.max(axis=1) / np.maximum(total, 1e-12), 0.0)
    classes = np.where(voted, votes.argmax(axis=1), UNKNOWN).astype(np.int8)
    source = np.where(voted, FROM_VOTES, UNLABELED).astype(np.int8)
    votes_only = classes.copy()

    if plane_of is not None:
        for p in np.unique(plane_of[plane_of >= 0]):
            members = np.flatnonzero(plane_of == p)
            plane_votes = votes[members].sum(axis=0)
            if plane_votes.sum() == 0:
                continue
            weak = members[confidence[members] < cfg.plane_confidence]
            classes[weak] = plane_votes.argmax()
            source[weak] = FROM_PLANE

    missing = np.flatnonzero(classes == UNKNOWN)
    labeled = np.flatnonzero(classes != UNKNOWN)
    if len(missing) and len(labeled):
        k = min(cfg.neighbours, len(labeled))
        near = cKDTree(points[labeled]).query(points[missing], k=k, workers=-1)[1].reshape(len(missing), k)
        rows = np.arange(len(missing))[:, None] * C
        counts = np.bincount((rows + classes[labeled][near]).ravel(), minlength=len(missing) * C)
        classes[missing] = counts.reshape(len(missing), C).argmax(axis=1)
        source[missing] = FROM_NEIGHBOURS
    return Fusion(classes, source, confidence.astype(np.float32), votes_only, single_view, views_seen)


def workspace_views(dense_dir: Path, seg_dir: Path, pycolmap) -> Iterator[View]:
    """Views of the undistorted dense workspace, with stage 4 labels and MVS depth maps."""
    model = pycolmap.Reconstruction(dense_dir / "sparse")
    lookup = read_lookup(seg_dir)
    for image in sorted(model.images.values(), key=lambda i: i.name):
        ade = read_labels(seg_dir, image.name)
        if ade is None:
            continue
        camera = model.cameras[image.camera_id]
        K = np.array(camera.calibration_matrix())
        K[0] *= ade.shape[1] / camera.width
        K[1] *= ade.shape[0] / camera.height
        depth_path = dense_dir / "stereo" / "depth_maps" / f"{image.name}.geometric.bin"
        yield View(
            image.name, K, np.array(image.cam_from_world().matrix()), lookup[ade],
            read_depth_map(depth_path) if depth_path.exists() else None,
        )


def fuse_labels(dense_dir: Path, seg_dir: Path, output_dir: Path, config: FusionConfig | None = None) -> FusionResult:
    """Labels dense.ply (fused.ply, then plane_fill.ply). Writes output_dir/{labels.npz,
    labels.ply (coloured by class), structure.ply, objects.ply, fusion.json}."""
    cfg = config or FusionConfig()
    pycolmap = load_pycolmap()
    output_dir.mkdir(parents=True, exist_ok=True)
    fused, fill = read_ply(dense_dir / "fused.ply"), read_ply(dense_dir / "plane_fill.ply")
    cloud = concatenate([fused, fill])
    planes = np.load(dense_dir / "planes.npz")
    plane_of = np.concatenate([planes["point_plane"], planes["fill_plane"]])
    if len(plane_of) != len(cloud):
        raise ValueError(f"{dense_dir}/planes.npz does not match fused.ply + plane_fill.ply")

    used: list[str] = []

    def views() -> Iterator[View]:  # one at a time: depth maps of every view would not fit in RAM
        for view in workspace_views(dense_dir, seg_dir, pycolmap):
            used.append(view.name)
            yield view

    f = fuse(cloud.points, cloud.normals, views(), plane_of, cfg)
    if not used:
        raise ValueError(f"no keyframe in {dense_dir} has a stage 4 label image in {seg_dir}")

    np.savez_compressed(output_dir / "labels.npz", num_fused=len(fused), **asdict(f))
    structure = is_structure(f.classes)
    write_ply(output_dir / "labels.ply", PointCloud(cloud.points, cloud.normals, class_colors(f.classes)))
    write_ply(output_dir / "structure.ply", cloud[structure])
    write_ply(output_dir / "objects.ply", cloud[f.classes == OBJECT])
    result = FusionResult(
        len(cloud), len(used),
        *(int(np.sum(f.source == s)) for s in (FROM_VOTES, FROM_PLANE, FROM_NEIGHBOURS, UNLABELED)),
        int(structure.sum()), int(np.sum(f.classes == OBJECT)),
        {c: int(np.sum(f.classes == i)) for i, c in enumerate(CLASSES)},
    )
    (output_dir / "fusion.json").write_text(json.dumps(asdict(result) | {"config": asdict(cfg)}, indent=2))
    return result
