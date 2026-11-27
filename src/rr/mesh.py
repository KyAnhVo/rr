"""Triangle meshes of a finished run, by Poisson surface reconstruction of its point clouds.

    python -m rr.mesh <output> [--depth 10]

Writes <output>/mesh/:
  scene.ply      dense.ply (measured + plane fill), true colours       (stage 3)
  labels.ply     the same points coloured by structure/object class     (stage 5)
  structure.ply  walls, floor, ceiling, doors, windows                  (stage 5)
  objects.ply    everything else                                        (stage 5)
  frame.json     the 4x4 transform from the reconstruction to the mesh frame

COLMAP's frame has no "up", so the meshes are levelled: z up, the floor at z = 0, the largest
wall along x, centred on the room. Units stay the reconstruction's (no metric scale). Clouds of
stages that did not run are skipped.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rr.colmap import load_pycolmap
from rr.evaluation import Similarity
from rr.ply import PointCloud, read_ply, write_ply
from rr.visualize import gravity_frame

SOURCES = {
    "scene": "dense/dense.ply",
    "labels": "semantic/labels.ply",
    "structure": "semantic/structure.ply",
    "objects": "semantic/objects.ply",
}
SNAP_DEG = 15.0  # a plane this close to level (or upright) counts as floor/ceiling (or wall)


def level_frame(normals: NDArray[np.float64], support: NDArray[np.int64], camera_up: NDArray[np.float64],
                points: NDArray[np.float64]) -> Similarity:
    """z-up frame. `normals`/`support` are the stage 3 planes and their inlier counts.

    The cameras' average 'up' is only roughly vertical (people tilt the camera), so it is
    snapped to the best supported floor/ceiling plane, and x to the best supported wall.
    """
    up = camera_up / np.linalg.norm(camera_up)
    cos = normals @ up
    level = np.flatnonzero(np.abs(cos) > np.cos(np.radians(SNAP_DEG)))
    if len(level):
        best = level[np.argmax(support[level])]
        up = normals[best] * np.sign(cos[best])
    upright = np.flatnonzero(np.abs(normals @ up) < np.sin(np.radians(SNAP_DEG)))
    if len(upright):
        x = normals[upright[np.argmax(support[upright])]]
    else:
        x = np.eye(3)[np.argmin(np.abs(up))]
    x = x - (x @ up) * up
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(up, x), up])
    rotated = points @ R.T
    # Floor at z = 0 (a low percentile, so stray points below it do not count); centred in x, y.
    t = -np.array([*np.median(rotated[:, :2], axis=0), np.percentile(rotated[:, 2], 0.5)])
    return Similarity(1.0, R, t)


def run_frame(output: Path) -> Similarity:
    pycolmap = load_pycolmap()
    model = pycolmap.Reconstruction(output / "dense" / "sparse")
    camera_up = gravity_frame(model).R[2]
    points = read_ply(output / "dense" / "fused.ply").points
    planes = np.load(output / "dense" / "planes.npz")
    plane_of = planes["point_plane"]
    support = np.bincount(plane_of[plane_of >= 0], minlength=len(planes["equations"]))
    return level_frame(planes["equations"][:, :3], support, camera_up, points[:: max(1, len(points) // 200_000)])


def write_meshes(output: Path, depth: int = 10, trim: float = 10.0) -> list[Path]:
    """`depth` is the octree depth: each step up roughly quadruples triangles and time
    (10 = about 2M triangles and half a minute for a room). `trim` drops faces in sparsely
    sampled areas, so unseen space is not closed over with invented surface."""
    pycolmap = load_pycolmap()
    options = pycolmap.PoissonMeshingOptions()
    options.depth = depth
    options.trim = trim
    mesh_dir = output / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    frame = run_frame(output)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = frame.scale * frame.R, frame.t
    (mesh_dir / "frame.json").write_text(json.dumps({
        "mesh_from_reconstruction": matrix.round(9).tolist(),
        "note": "applies to dense/, semantic/ and sparse/ coordinates: x_mesh = M @ [x, 1]",
    }, indent=2))

    written = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in SOURCES.items():
            path = output / source
            if not path.exists():
                continue
            # Level the points first, so the mesh comes out levelled.
            levelled = Path(tmp) / f"{name}.ply"
            write_ply(levelled, frame.cloud(read_ply(path)))
            target = mesh_dir / f"{name}.ply"
            pycolmap.poisson_meshing(levelled, target, options)
            if not target.exists():
                raise RuntimeError(f"Poisson meshing of {path} produced nothing")
            written.append(target)
    if not written:
        raise FileNotFoundError(f"no point clouds under {output}; run at least stage 3 first")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr-mesh", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="output directory of an rr run")
    parser.add_argument("--depth", type=int, default=10, help="Poisson octree depth (detail vs. size)")
    parser.add_argument("--trim", type=float, default=10.0, help="drop faces below this sample density")
    args = parser.parse_args()
    for path in write_meshes(args.output, args.depth, args.trim):
        print(path)


if __name__ == "__main__":
    main()
