"""Runs stages 1-3 on ScanNet scenes, scores them and writes pictures.

    rr-scannet --data room-data --out results
"""
import argparse
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
from rr.sens import SensReader
from rr.visualize import frame_comparisons, gravity_frame, keyframe_sheet, original_frames, outside_views

MODES = ("estimated", "known")  # camera intrinsics: from RGB only / from the .sens header


@dataclass
class Scene:
    name: str
    sens: Path
    mesh: Path


def find_scenes(data: Path) -> list[Scene]:
    """Scenes with both a recording and a ground-truth mesh, wherever they sit under `data`."""
    scenes = []
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
    sens = SensReader(scene.sens)
    pictures = out / "pictures"
    metrics: dict[str, Any] = {"scene": scene.name, "mode": mode, "frames": len(sens)}

    def save() -> None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def timed(stage: str, started: float) -> None:
        metrics.setdefault("seconds", {})[stage] = round(time.monotonic() - started, 1)

    original_frames(sens, pictures / "original")
    intrinsics = sens.color_intrinsics[:3, :3] if mode == "known" else None
    started = time.monotonic()
    keyframes = KeyframeSelector(KeyframeConfig(), intrinsics).select(sens, out)
    timed("keyframes", started)
    metrics["keyframes"] = len(keyframes)
    keyframe_sheet(out / "images", [k.name for k in keyframes], pictures / "keyframes.jpg")
    save()

    started = time.monotonic()
    sparse = reconstruct_sparse(
        out / "images", out / "sparse", SparseConfig(matcher=matcher), intrinsics, sens.color_size
    )
    timed("sparse", started)
    metrics["sparse"] = asdict(sparse)
    metrics["registered_fraction"] = sparse.images_registered / sparse.images_total
    save()

    model = pycolmap.Reconstruction(sparse.model_dir)
    frame_of = {k.name: k.frame_index for k in keyframes}
    gt = sample_mesh(scene.mesh)
    try:
        world, metrics["trajectory"] = align_cameras(model, frame_of, sens)
        aligned = True
    except ValueError as e:
        world, aligned = gravity_frame(model), False
        metrics["alignment_error"] = str(e)
    some = gt[np.arange(min(400_000, len(gt)))]
    visible = visible_mask(some.points, sens, [k.frame_index for k in keyframes])
    metrics["gt_visible_fraction"] = float(visible.mean())
    metrics["accuracy_floor_m"] = sampling_floor(gt.points)

    def evaluate(name: str, cloud) -> None:
        if aligned:
            metrics.setdefault("clouds", {})[name] = score(world.apply(cloud.points), gt.points, visible)
        save()

    sparse_cloud = read_ply(out / "sparse" / "sparse.ply")
    evaluate("sparse", sparse_cloud)
    outside_views(some, pictures / "ground_truth", "ground truth", 1)
    outside_views(world.cloud(sparse_cloud), pictures / "sparse", "sparse", 2)
    frame_comparisons(model, out / "images", {"sparse": (sparse_cloud, 2)}, pictures / "compare_sparse")

    started = time.monotonic()
    metrics["dense"] = asdict(reconstruct_dense(
        out / "images", Path(sparse.model_dir), out / "dense", DenseConfig(max_image_size=dense_size)
    ))
    timed("dense", started)
    dense = {n: read_ply(out / "dense" / f"{n}.ply") for n in ("fused", "plane_fill", "dense", "planes")}
    for name in ("fused", "plane_fill", "dense"):
        evaluate(name, dense[name])
    for name in ("dense", "planes"):
        if len(dense[name]):
            outside_views(world.cloud(dense[name]), pictures / name, name, 1)
    frame_comparisons(
        model, out / "images", {"sparse": (sparse_cloud, 2), "dense": (dense["dense"], 1)}, pictures / "compare_dense"
    )
    save()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr-scannet", description=__doc__)
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
        parser.error(f"no scenes under {args.data}; run scripts/download_scannet.sh first")
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
    raise SystemExit(1 if failures else 0)
