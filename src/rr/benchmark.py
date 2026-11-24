"""Runs stages 1-3 on Replica (or ScanNet) scenes, scores them and writes pictures.

    rr-benchmark --data room-data --out results
"""
import argparse
import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rr.colmap import load_pycolmap
from rr.evaluation import align_cameras, sample_mesh, sampling_floor, score, visible_mask
from rr.phase1 import KeyframeConfig, KeyframeSelector
from rr.phase2 import SparseConfig, reconstruct_sparse
from rr.phase3 import DenseConfig, reconstruct_dense
from rr.ply import read_ply
from rr.sources import Recording, open_source
from rr.visualize import facing_cameras, frame_comparisons, gravity_frame, keyframe_sheet, original_frames, outside_views

MODES = ("estimated", "known")  # camera intrinsics: from RGB only / from the dataset


@dataclass
class Scene:
    name: str
    recording: Path
    mesh: Path


def find_scenes(data: Path) -> list[Scene]:
    """Scenes with both a recording and a ground-truth mesh, wherever they sit under `data`."""
    scenes = [
        Scene(traj.parent.name, traj.parent, mesh)
        for traj in sorted(data.rglob("traj.txt"))
        if (mesh := traj.parent.with_name(f"{traj.parent.name}_mesh.ply")).exists()
    ]
    for sens in sorted(data.rglob("scene*.sens")):
        mesh = sens.with_name(f"{sens.stem}_vh_clean_2.ply")
        if mesh.exists():
            scenes.append(Scene(sens.stem, sens, mesh))
    return scenes


def run_scene(scene: Scene, mode: str, out: Path, dense_size: int = 1200, matcher: Any = "sequential") -> dict[str, Any]:
    """Pictures are written after each stage, so a later failure keeps the earlier ones."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if out.exists():
        raise FileExistsError(out)
    pycolmap = load_pycolmap()
    recording: Recording = open_source(scene.recording)  # type: ignore[assignment]
    pictures = out / "pictures"
    metrics: dict[str, Any] = {"scene": scene.name, "mode": mode, "frames": len(recording)}

    def save() -> None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def timed(stage: str, started: float) -> None:
        metrics.setdefault("seconds", {})[stage] = round(time.monotonic() - started, 1)

    original_frames(recording, pictures / "original")
    intrinsics = recording.color_intrinsics[:3, :3] if mode == "known" else None
    started = time.monotonic()
    keyframes = KeyframeSelector(KeyframeConfig(), intrinsics).select(recording, out)
    timed("keyframes", started)
    metrics["keyframes"] = len(keyframes)
    keyframe_sheet(out / "images", [k.name for k in keyframes], pictures / "keyframes.jpg")
    save()

    started = time.monotonic()
    sparse = reconstruct_sparse(
        out / "images", out / "sparse", SparseConfig(matcher=matcher), intrinsics, recording.color_size
    )
    timed("sparse", started)
    metrics["sparse"] = asdict(sparse)
    metrics["registered_fraction"] = sparse.images_registered / sparse.images_total
    save()

    model = pycolmap.Reconstruction(sparse.model_dir)
    frame_of = {k.name: k.frame_index for k in keyframes}
    gt = sample_mesh(scene.mesh)
    try:
        world, metrics["trajectory"] = align_cameras(model, frame_of, recording)
        aligned = True
    except ValueError as e:
        world, aligned = gravity_frame(model), False
        metrics["alignment_error"] = str(e)
    some = gt[np.arange(min(400_000, len(gt)))]
    visible = visible_mask(some.points, recording, [k.frame_index for k in keyframes])
    metrics["gt_visible_fraction"] = float(visible.mean())
    metrics["accuracy_floor_m"] = sampling_floor(gt.points)

    def evaluate(name: str, cloud) -> None:
        if aligned:
            metrics.setdefault("clouds", {})[name] = score(world.apply(cloud.points), gt.points, visible)
        save()

    sparse_cloud = read_ply(out / "sparse" / "sparse.ply")
    evaluate("sparse", sparse_cloud)
    shown_sparse = world.cloud(facing_cameras(sparse_cloud, model))
    # Only with an alignment can the target be drawn from the estimated cameras.
    outside = {"target": (gt, 1)} if aligned else {}
    outside["sparse"] = (shown_sparse, 2)
    inside = {"target mesh": (world.inverse().cloud(gt), 1)} if aligned else {}
    inside["sparse"] = (sparse_cloud, 2)
    outside_views(outside, pictures / "outside_sparse")
    frame_comparisons(model, out / "images", inside, pictures / "inside_sparse")

    started = time.monotonic()
    metrics["dense"] = asdict(reconstruct_dense(
        out / "images", Path(sparse.model_dir), out / "dense", DenseConfig(max_image_size=dense_size)
    ))
    timed("dense", started)
    dense = {n: read_ply(out / "dense" / f"{n}.ply") for n in ("fused", "plane_fill", "dense", "planes")}
    for name in ("fused", "plane_fill", "dense"):
        evaluate(name, dense[name])
    outside["dense"] = (world.cloud(dense["dense"]), 1)
    outside["planes"] = (world.cloud(dense["planes"]), 1)
    inside["dense"] = (dense["dense"], 1)
    outside_views(outside, pictures / "outside")
    frame_comparisons(model, out / "images", inside, pictures / "inside")
    save()
    return metrics


def write_summary(root: Path) -> None:
    """summary.csv with one row per scene, mode and cloud, for every run under `root`."""
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        m = json.loads(path.read_text())
        shared = {k: m.get(k) for k in ("scene", "mode", "keyframes", "registered_fraction", "gt_visible_fraction", "accuracy_floor_m")}
        shared |= m.get("trajectory", {})
        rows += [shared | {"cloud": name} | c for name, c in m.get("clouds", {}).items()]
    if rows:
        with (root / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr-benchmark", description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("room-data"))
    parser.add_argument("--out", type=Path, default=Path("results") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--scenes", nargs="*", help="default: every scene found")
    parser.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--dense-size", type=int, default=1200)
    args = parser.parse_args()
    scenes = find_scenes(args.data)
    if args.scenes:
        missing = set(args.scenes) - {s.name for s in scenes}
        if missing:
            parser.error(f"not found under {args.data}: {', '.join(sorted(missing))}")
        scenes = [s for s in scenes if s.name in args.scenes]
    if not scenes:
        parser.error(f"no scenes under {args.data}; run scripts/download_replica.py first")
    failures = 0
    for scene in scenes:
        for mode in args.modes:
            out = args.out / scene.name / mode
            try:
                print(json.dumps(run_scene(scene, mode, out, args.dense_size, args.matcher), indent=2), flush=True)
            except Exception:  # keep going; a failed scene is itself a result
                failures += 1
                out.mkdir(parents=True, exist_ok=True)
                (out / "FAILED.txt").write_text(traceback.format_exc())
                print(f"FAILED {scene.name} {mode}; see {out / 'FAILED.txt'}", flush=True)
    write_summary(args.out)
    raise SystemExit(1 if failures else 0)
