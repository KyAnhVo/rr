"""Stage 3: dense reconstruction (PatchMatch MVS) plus RANSAC plane segments and gap filling."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rr.colmap import load_pycolmap, read_depth_map
from rr.planes import PlaneConfig, fill_plane, fit_planes, scene_size
from rr.ply import PointCloud, concatenate, read_ply, write_ply


@dataclass
class DenseConfig:
    max_image_size: int = 1200         # fits comfortably in 8 GB of VRAM
    cache_gb: float = 2.0              # host RAM per cache
    free_space_tolerance: float = 0.03 # relative depth margin for "seen through"
    free_space_views: int = 2
    planes: PlaneConfig = field(default_factory=PlaneConfig)


@dataclass
class DenseResult:
    num_dense_points: int
    num_planes: int                    # plane segments
    num_plane_fill_points: int
    unit: float = 0.0                  # typical viewing distance, the planes' length unit
    scene_size: float = 0.0            # robust bounding-box diagonal, for comparison


def viewing_distance(dense_dir: Path, views: int = 25) -> float:
    """Median camera-to-surface distance, from a spread of MVS depth maps (model units)."""
    paths = sorted((dense_dir / "stereo" / "depth_maps").glob("*.geometric.bin"))
    if not paths:
        raise RuntimeError(f"no depth maps in {dense_dir}")
    medians = []
    for path in paths[:: max(1, len(paths) // views)]:
        depth = read_depth_map(path)
        if (depth > 0).any():
            medians.append(float(np.median(depth[depth > 0])))
    if not medians:
        raise RuntimeError(f"every depth map in {dense_dir} is empty")
    return float(np.median(medians))


def seen_through(
    points: NDArray[np.float64], K: NDArray[np.float64], cam_from_world: NDArray[np.float64],
    depth: NDArray[np.float32], tolerance: float,
) -> NDArray[np.bool_]:
    """True where the camera measured a surface clearly *behind* the point."""
    cam = points @ cam_from_world[:, :3].T + cam_from_world[:, 3]
    z = cam[:, 2]
    result = np.zeros(len(points), dtype=bool)
    front = np.flatnonzero(z > 1e-6)
    proj = cam[front] @ K.T
    xy = np.floor(proj[:, :2] / proj[:, 2:])
    h, w = depth.shape
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    idx, xy = front[ok], xy[ok].astype(int)
    result[idx] = depth[xy[:, 1], xy[:, 0]] > z[idx] * (1 + tolerance)
    return result


def _remove_free_space(points: PointCloud, dense_dir: Path, pycolmap: Any, cfg: DenseConfig) -> PointCloud:
    """Drop fill points that would close off a doorway or other observed empty space."""
    model = pycolmap.Reconstruction(dense_dir / "sparse")
    votes = np.zeros(len(points), dtype=int)
    for image in model.images.values():
        path = dense_dir / "stereo" / "depth_maps" / f"{image.name}.geometric.bin"
        if not path.exists():
            continue
        depth = read_depth_map(path)
        camera = model.cameras[image.camera_id]
        K = np.array(camera.calibration_matrix())
        # Depth maps may be smaller than the undistorted image.
        K[0] *= depth.shape[1] / camera.width
        K[1] *= depth.shape[0] / camera.height
        votes += seen_through(
            points.points, K, np.array(image.cam_from_world().matrix()), depth, cfg.free_space_tolerance
        )
    return points[votes < cfg.free_space_views]


def reconstruct_dense(
    image_dir: Path, model_dir: Path, output_dir: Path, config: DenseConfig | None = None
) -> DenseResult:
    """Writes fused.ply (measured), plane_fill.ply (synthetic), planes.ply (coloured by plane),
    dense.ply (measured + synthetic) and planes.npz / dense.json."""
    cfg = config or DenseConfig()
    pycolmap = load_pycolmap()
    if not pycolmap.has_cuda:
        raise RuntimeError("PatchMatch stereo needs a CUDA build of pycolmap (pycolmap-cuda12)")
    if (output_dir / "stereo").exists():
        raise FileExistsError(f"{output_dir} already holds a dense workspace; use a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    undistort = pycolmap.UndistortCameraOptions()
    undistort.max_image_size = cfg.max_image_size
    pycolmap.undistort_images(output_dir, model_dir, image_dir, undistort_options=undistort)
    stereo = pycolmap.PatchMatchOptions()
    stereo.max_image_size = cfg.max_image_size
    stereo.cache_size = cfg.cache_gb
    pycolmap.patch_match_stereo(output_dir, options=stereo)
    fusion = pycolmap.StereoFusionOptions()
    fusion.max_image_size = cfg.max_image_size
    fusion.use_cache = True
    fusion.cache_size = cfg.cache_gb
    fused_path = output_dir / "fused.ply"
    pycolmap.stereo_fusion(fused_path, output_dir, options=fusion, output_type="PLY")

    cloud = read_ply(fused_path)
    if len(cloud) == 0:
        raise RuntimeError("dense fusion produced no points")
    unit = viewing_distance(output_dir)
    planes = fit_planes(cloud, unit, cfg.planes)
    fills = [
        _remove_free_space(fill_plane(cloud, p, unit, cfg.planes), output_dir, pycolmap, cfg) for p in planes
    ]
    empty = cloud[np.zeros(0, dtype=np.int64)]
    fill = concatenate([empty, *fills])
    write_ply(output_dir / "plane_fill.ply", fill)
    write_ply(output_dir / "dense.ply", concatenate([cloud, fill]))

    palette = np.random.default_rng(1).integers(60, 256, (max(1, len(planes)), 3)).astype(np.uint8)
    colored = []
    point_plane = np.full(len(cloud), -1, dtype=np.int32)
    for i, plane in enumerate(planes):
        part = cloud[plane.inliers]
        part.colors = np.tile(palette[i], (len(part), 1))
        colored.append(part)
        point_plane[plane.inliers] = i
    write_ply(output_dir / "planes.ply", concatenate([empty, *colored]))
    # For stages 4-5: the plane of every fused point (-1 = none) and every synthetic point.
    np.savez_compressed(
        output_dir / "planes.npz",
        equations=np.array([[*p.normal, p.d] for p in planes]).reshape(-1, 4),
        point_plane=point_plane,
        fill_plane=np.concatenate([np.full(len(f), i, np.int32) for i, f in enumerate(fills)] or [np.zeros(0, np.int32)]),
    )
    result = DenseResult(len(cloud), len(planes), len(fill), unit, scene_size(cloud.points))
    (output_dir / "dense.json").write_text(json.dumps(asdict(result) | {"config": asdict(cfg)}, indent=2))
    return result
