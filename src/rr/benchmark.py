"""Runs stages 1-5 on Replica scenes, scores them and writes pictures.

    rr-benchmark --data room-data --out results

Pictures go to <run>/pictures/ as described in rr.pictures, plus ground-truth charts:
p2_trajectory.png, p3_scores.png and p5_scores.png.
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

from rr.charts import cloud_scores, label_scores, trajectory
from rr.colmap import load_pycolmap
from rr.evaluation import align_cameras, camera_positions, sample_mesh, sampling_floor, score, visible_mask
from rr.phase1 import KeyframeConfig, KeyframeSelector
from rr.phase2 import SparseConfig, reconstruct_sparse
from rr.phase3 import DenseConfig, reconstruct_dense
from rr.phase4 import SegmentationConfig, segment_in_subprocess
from rr.phase5 import fuse_labels
from rr.pictures import Pictures
from rr.ply import PointCloud, concatenate, read_ply
from rr.semantic_gt import frame_offset, sample_labeled, semantic_files
from rr.semantic_gt import label_scores as score_labels
from rr.sources import Recording, open_source
from rr.taxonomy import class_colors

MODES = ("estimated", "known")  # camera intrinsics: from RGB only / from the dataset


@dataclass
class Scene:
    name: str
    recording: Path
    mesh: Path


def find_scenes(data: Path) -> list[Scene]:
    """Scenes with both a recording and a ground-truth mesh, wherever they sit under `data`."""
    return [
        Scene(traj.parent.name, traj.parent, mesh)
        for traj in sorted(data.rglob("traj.txt"))
        if (mesh := traj.parent.with_name(f"{traj.parent.name}_mesh.ply")).exists()
    ]


def run_scene(
    scene: Scene, mode: str, out: Path, dense_size: int = 1200, matcher: Any = "sequential",
    through: int = 5, segmentation: SegmentationConfig | None = None,
) -> dict[str, Any]:
    """Runs stages 1..`through`. metrics.json is rewritten after every stage, with "completed"
    the last finished stage, and pictures are written as soon as their stage is done, so a later
    failure keeps everything before it."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if out.exists():
        raise FileExistsError(out)
    pycolmap = load_pycolmap()
    recording: Recording = open_source(scene.recording)  # type: ignore[assignment]
    pictures = Pictures(out)
    charts = out / "pictures"
    metrics: dict[str, Any] = {"scene": scene.name, "mode": mode, "frames": len(recording), "completed": 0}

    def save() -> None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def timed(stage: str, started: float) -> None:
        metrics.setdefault("seconds", {})[stage] = round(time.monotonic() - started, 1)

    def done(stage: int) -> bool:
        metrics["completed"] = stage
        save()
        return stage >= through

    # Phase 1: keyframes.
    intrinsics = recording.color_intrinsics[:3, :3] if mode == "known" else None
    started = time.monotonic()
    keyframes = KeyframeSelector(KeyframeConfig(), intrinsics).select(recording, out)
    timed("keyframes", started)
    metrics["keyframes"] = len(keyframes)
    pictures.phase1(recording)
    if done(1):
        return metrics

    # Phase 2: sparse SfM.
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
        pictures.set_ground_truth(world, gt)
        estimated, truth = camera_positions(model, frame_of, recording)
        trajectory(world.apply(estimated), truth, charts / "p2_trajectory.png")
    except ValueError as e:  # pictures then fall back to a levelled frame without the target
        aligned = False
        metrics["alignment_error"] = str(e)
    some = gt[np.arange(min(400_000, len(gt)))]
    visible = visible_mask(some.points, recording, [k.frame_index for k in keyframes])
    metrics["gt_visible_fraction"] = float(visible.mean())
    metrics["accuracy_floor_m"] = sampling_floor(gt.points)

    def evaluate(name: str, cloud) -> None:
        if aligned:
            metrics.setdefault("clouds", {})[name] = score(world.apply(cloud.points), gt.points, visible)
        save()

    evaluate("sparse", read_ply(out / "sparse" / "sparse.ply"))
    pictures.phase2(model)
    if done(2):
        return metrics

    # Phase 3: dense MVS and planes.
    started = time.monotonic()
    metrics["dense"] = asdict(reconstruct_dense(
        out / "images", Path(sparse.model_dir), out / "dense", DenseConfig(max_image_size=dense_size)
    ))
    timed("dense", started)
    dense = {n: read_ply(out / "dense" / f"{n}.ply") for n in ("fused", "plane_fill", "dense")}
    for name, cloud in dense.items():
        evaluate(name, cloud)
    if aligned:
        cloud_scores(metrics["clouds"], charts / "p3_scores.png")
    pictures.phase3(model)
    if done(3):
        return metrics

    # Phase 4: 2D segmentation of the undistorted keyframes that phase 3 used.
    started = time.monotonic()
    seg = segment_in_subprocess(out / "dense" / "images", out / "segmentation", segmentation)
    timed("segmentation", started)
    metrics["segmentation"] = asdict(seg)
    pictures.phase4()
    if done(4):
        return metrics

    # Phase 5: label fusion.
    started = time.monotonic()
    metrics["fusion"] = asdict(fuse_labels(out / "dense", out / "segmentation", out / "semantic"))
    timed("fusion", started)
    z = np.load(out / "semantic" / "labels.npz")
    cloud = concatenate([dense["fused"], dense["plane_fill"]])
    target_labels: PointCloud | None = None
    files = semantic_files(scene.recording)
    if files is None:
        metrics["labels_skipped"] = "no semantic ground truth; run scripts/download_replica_semantics.py"
    elif not aligned:
        metrics["labels_skipped"] = "no camera alignment"
    else:
        gt_points, gt_normals, gt_classes = sample_labeled(*files)
        metrics["semantic_frame_offset_m"] = offset = frame_offset(gt_points, gt.points)
        if offset > 0.02:
            metrics["labels_skipped"] = f"semantic mesh is {offset:.3f} m off the geometry mesh: different frame?"
        else:
            target_labels = PointCloud(gt_points, gt_normals, class_colors(gt_classes))
            metrics["labels"] = score_labels(world.apply(cloud.points), {
                "single view": z["single_view"], "multi-view votes": z["votes_only"],
                "final (+ plane prior, neighbours)": z["classes"],
            }, gt_points, gt_classes)
            label_scores(metrics["labels"], charts / "p5_scores.png")
    save()
    pictures.phase5(pycolmap.Reconstruction(out / "dense" / "sparse"), target_labels)
    done(5)
    return metrics


def write_summary(root: Path) -> None:
    """summary.csv (one row per scene, mode and cloud) and labels.csv (per label variant)."""
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        m = json.loads(path.read_text())
        shared = {k: m.get(k) for k in ("scene", "mode", "keyframes", "registered_fraction", "gt_visible_fraction", "accuracy_floor_m")}
        shared |= m.get("trajectory", {})
        rows += [shared | {"cloud": name} | c for name, c in m.get("clouds", {}).items()]
    _write_csv(root / "summary.csv", rows)
    labels = []
    for path in sorted(root.glob("*/*/metrics.json")):
        m = json.loads(path.read_text())
        labels += [{"scene": m.get("scene"), "mode": m.get("mode"), "variant": v} | s for v, s in m.get("labels", {}).items()]
    _write_csv(root / "labels.csv", labels)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="") as f:
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
    parser.add_argument("--through", type=int, choices=range(1, 6), default=5, help="last stage to run")
    parser.add_argument("--seg-model", default=SegmentationConfig.model)
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
                metrics = run_scene(scene, mode, out, args.dense_size, args.matcher, args.through,
                                    SegmentationConfig(model=args.seg_model))
                print(json.dumps(metrics, indent=2), flush=True)
            except Exception:  # keep going; a failed scene is itself a result
                failures += 1
                out.mkdir(parents=True, exist_ok=True)
                (out / "FAILED.txt").write_text(traceback.format_exc())
                print(f"FAILED {scene.name} {mode}; see {out / 'FAILED.txt'}", flush=True)
    write_summary(args.out)
    raise SystemExit(1 if failures else 0)
