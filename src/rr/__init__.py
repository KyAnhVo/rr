import argparse
import traceback
from collections.abc import Callable
from pathlib import Path


def _pictures(phase: int, draw: Callable[[], None]) -> None:
    """A failed picture is reported, but never costs the run."""
    try:
        draw()
    except Exception:
        print(f"warning: phase {phase} pictures failed:\n{traceback.format_exc()}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr", description="Room reconstruction and structure/object separation, stages 1-5")
    parser.add_argument("video", type=Path, help="video or Replica scene directory")
    parser.add_argument("output", type=Path, help="fresh output directory")
    parser.add_argument("--through", type=int, choices=range(1, 6), default=5, help="last stage to run")
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential",
                        help="exhaustive matches every keyframe pair: slow, but finds every loop")
    parser.add_argument("--loops", choices=("auto", "on", "off"), default="auto",
                        help="loop detection for sequential matching; auto = only for long videos")
    parser.add_argument("--vocab-tree", help="vocabulary tree file for loop detection (default: COLMAP's)")
    parser.add_argument("--focal-factor", type=float, default=0.8, help="focal guess / long image side")
    parser.add_argument("--min-parallax", type=float, default=1.5, help="degrees between keyframes")
    parser.add_argument("--dense-size", type=int, default=1200, help="long image side for MVS")
    parser.add_argument("--seg-model", default=None, help="Hugging Face segmentation checkpoint (stage 4)")
    parser.add_argument("--no-pictures", action="store_true", help="skip <output>/pictures/")
    args = parser.parse_args()

    from rr.colmap import load_pycolmap
    from rr.phase1 import KeyframeConfig, KeyframeSelector
    from rr.phase2 import SparseConfig, reconstruct_sparse
    from rr.phase3 import DenseConfig, reconstruct_dense
    from rr.phase4 import SegmentationConfig, segment_in_subprocess
    from rr.phase5 import fuse_labels
    from rr.pictures import Pictures
    from rr.sources import open_source

    out: Path = args.output
    source = open_source(args.video)
    pictures = Pictures(out)
    draw = (lambda phase, f: None) if args.no_pictures else _pictures

    keyframes = KeyframeSelector(
        KeyframeConfig(focal_factor=args.focal_factor, min_parallax_deg=args.min_parallax)
    ).select(source, out)
    print(f"stage 1: {len(keyframes)} keyframes")
    draw(1, lambda: pictures.phase1(source))
    if args.through == 1:
        return

    sparse = reconstruct_sparse(out / "images", out / "sparse", SparseConfig(
        matcher=args.matcher, focal_factor=args.focal_factor, loops=args.loops, vocab_tree=args.vocab_tree,
    ))
    print(f"stage 2: {sparse}")
    if sparse.num_models > 1 or sparse.images_registered < sparse.images_total:
        print("warning: some keyframes were not registered in the chosen model")
    pycolmap = load_pycolmap()
    model = pycolmap.Reconstruction(sparse.model_dir)
    draw(2, lambda: pictures.phase2(model))
    if args.through == 2:
        return

    dense = reconstruct_dense(
        out / "images", Path(sparse.model_dir), out / "dense", DenseConfig(max_image_size=args.dense_size)
    )
    print(f"stage 3: {dense}")
    draw(3, lambda: pictures.phase3(model))
    if args.through == 3:
        return

    # Labels the undistorted keyframes, which match the MVS depth maps that stage 5 uses.
    seg_config = SegmentationConfig(model=args.seg_model) if args.seg_model else SegmentationConfig()
    seg = segment_in_subprocess(out / "dense" / "images", out / "segmentation", seg_config)
    print(f"stage 4: {seg}")
    draw(4, pictures.phase4)
    if args.through == 4:
        return

    fusion = fuse_labels(out / "dense", out / "segmentation", out / "semantic")
    print(f"stage 5: {fusion}")
    draw(5, lambda: pictures.phase5(pycolmap.Reconstruction(out / "dense" / "sparse")))
    print(f"pictures: {out / 'pictures'}" if not args.no_pictures else "")
